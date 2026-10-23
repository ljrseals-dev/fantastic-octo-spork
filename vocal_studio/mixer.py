"""Vocal mixer: per-track level/pan/EQ/de-ess/comp/saturation + reverb & echo sends, stems and mix export."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, fftconvolve
from .model import Session, Track
from .renderer import VocalRenderer, ParametricRenderer


def db(x: float) -> float:
    return 10 ** (x / 20)


def _eq(y, sr):  # high-pass 90 Hz, gentle presence shelf-ish lift
    y = sosfilt(butter(2, 90, "hp", fs=sr, output="sos"), y)
    return y + 0.25 * sosfilt(butter(2, [2500, 6000], "bp", fs=sr, output="sos"), y)


def _deess(y, sr, thresh=0.08):
    band = sosfilt(butter(2, [5500, 9500], "bp", fs=sr, output="sos"), y)
    env = np.convolve(np.abs(band), np.ones(int(0.005 * sr)) / int(0.005 * sr), "same")
    gain = np.where(env > thresh, thresh / np.maximum(env, 1e-9), 1.0)
    return y - band * (1 - gain)


def _comp(y, thr=0.25, ratio=3.0):
    m = np.abs(y)
    over = m > thr
    out = y.copy()
    out[over] = np.sign(y[over]) * (thr + (m[over] - thr) / ratio)
    return out


def _sat(y, drive=1.3):
    return np.tanh(y * drive) / np.tanh(drive)


def _reverb_ir(sr, seconds=1.8, seed=7):
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (sr * seconds / 5))
    ir = sosfilt(butter(1, 6000, "lp", fs=sr, output="sos"), ir)
    return (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)


def _echo(y, sr, ms, fb=0.4, taps=4):
    out = np.zeros(len(y) + int(taps * ms * sr / 1000), np.float32)
    for k in range(1, taps + 1):
        d = int(k * ms * sr / 1000)
        out[d:d + len(y)] += y * (fb ** k)
    return out


def _pan(y, pan):
    a = (pan + 1) * np.pi / 4
    return np.stack([y * np.cos(a), y * np.sin(a)], 1)


class VocalMixer:
    def __init__(self, session: Session, renderer: VocalRenderer | None = None):
        self.s, self.renderer = session, renderer or ParametricRenderer()
        self._ir = _reverb_ir(session.sr)
        self._cache: dict[str, np.ndarray] = {}

    def _dry(self, t: Track) -> np.ndarray:
        key = repr((t.notes, t.settings))
        if t.name not in self._cache or self._cache[t.name][0] != key:
            self._cache[t.name] = (key, self.renderer.render_track(t, self.s.sr, self.s.tempo))
        return self._cache[t.name][1]

    def process_track(self, t: Track) -> np.ndarray:
        """Return the stereo, fully processed track (dry + its own effect sends)."""
        sr, st = self.s.sr, t.settings
        y = self._dry(t)
        if len(y) == 0:
            return np.zeros((0, 2), np.float32)
        y = _sat(_comp(_deess(_eq(y, sr), sr)))
        y = y * db(st.level_db)
        wet = np.zeros(len(y) + len(self._ir), np.float32)
        wet[:len(y)] += y
        if st.reverb > 0:
            rv = fftconvolve(y, self._ir).astype(np.float32)
            wet[:len(rv)] += st.reverb * 0.6 * rv[:len(wet)]
        if st.echo > 0:
            ec = _echo(y, sr, st.echo_ms)
            if len(ec) > len(wet):
                wet = np.pad(wet, (0, len(ec) - len(wet)))
            wet[:len(ec)] += st.echo * ec
        return _pan(wet, st.pan).astype(np.float32)

    @staticmethod
    def _sum(stereos: list[np.ndarray]) -> np.ndarray:
        if not stereos:
            return np.zeros((1, 2), np.float32)
        n = max(len(x) for x in stereos)
        mix = np.zeros((n, 2), np.float32)
        for x in stereos:
            mix[:len(x)] += x
        return mix

    def mix(self) -> np.ndarray:
        m = self._sum([self.process_track(t) for t in self.s.active_tracks()])
        m = np.tanh(m * 1.1) / np.tanh(1.1)                       # master soft limiter
        return (m / max(np.max(np.abs(m)), 1e-9) * db(-1.0)).astype(np.float32)

    def export(self, outdir: str | Path, stems: bool = True) -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = {}
        full = self.mix()
        n = len(full)
        if stems:
            for t in self.s.tracks:           # every track, so stems line up in a DAW
                st = self.process_track(t)
                st = np.pad(st, ((0, max(0, n - len(st))), (0, 0)))[:n]
                p = outdir / f"stem_{t.name.replace(' ', '_')}.wav"
                wavfile.write(p, self.s.sr, (np.clip(st, -1, 1) * 32767).astype(np.int16))
                paths[t.name] = p
        p = outdir / "vocal_mix.wav"
        wavfile.write(p, self.s.sr, (full * 32767).astype(np.int16))
        paths["mix"] = p
        return paths
