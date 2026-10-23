"""Replaceable vocal rendering interface.

The arrangement engine only talks to `VocalRenderer.render_track`. Swap in a
better singing synth or an authorized voice-conversion backend without touching
harmony, mixing or export. `ParametricRenderer` is the built-in formant
synthesizer; it is an approximation of timbre, not a reconstruction of anyone's voice.
"""
from __future__ import annotations
import random
from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import lfilter
from .model import Track, Note

VOWELS = {  # (F1, F2, F3) Hz, adult-ish averages
    "a": (800, 1200, 2600), "e": (500, 1900, 2600), "i": (300, 2300, 3000),
    "o": (500, 900, 2500), "u": (320, 800, 2400),
}


def _vowel_of(lyric: str) -> str:
    for ch in lyric.lower():
        if ch in VOWELS:
            return ch
    return "a"


def _resonator(f: float, bw: float, sr: int):
    r = np.exp(-np.pi * bw / sr)
    th = 2 * np.pi * f / sr
    return [1 - r], [1, -2 * r * np.cos(th), r * r]


class VocalRenderer(ABC):
    @abstractmethod
    def render_track(self, track: Track, sr: int, tempo: float) -> np.ndarray:
        """Return a mono float32 signal for the whole track (timeline starts at 0 s)."""


class ParametricRenderer(VocalRenderer):
    def __init__(self, profile: dict | None = None):
        """`profile` (optional, from a consented enrollment) may supply
        formant_scale, vibrato_rate_hz, vibrato_depth_cents, breathiness."""
        self.profile = profile or {}

    def render_track(self, track: Track, sr: int, tempo: float) -> np.ndarray:
        s = track.settings
        rng = random.Random(s.seed)
        nrng = np.random.default_rng(s.seed)
        fscale = float(self.profile.get("formant_scale", 1.0))
        vrate = float(self.profile.get("vibrato_rate_hz", s.vibrato_rate_hz))
        vdepth = float(self.profile.get("vibrato_depth_cents", s.vibrato_depth_cents))
        breath = float(self.profile.get("breathiness", s.breath))
        if not track.notes:
            return np.zeros(0, np.float32)
        total = max(n.end for n in track.notes) + 1.0 + abs(s.delay_ms) / 1000
        out = np.zeros(int(total * sr) + sr, np.float32)
        for n in track.notes:
            t0 = n.start + s.delay_ms / 1000 + rng.gauss(0, s.timing_ms / 1000)
            t0 = max(0.0, t0)
            cents = rng.gauss(0, s.tuning_cents)
            sig = self._note(n, sr, vrate, vdepth, breath, cents, fscale, s.brightness, nrng, rng)
            i = int(t0 * sr)
            out[i:i + len(sig)] += sig[: len(out) - i]
        return out

    def _note(self, n: Note, sr, vrate, vdepth, breath, cents, fscale, bright, nrng, rng):
        dur = n.dur + 0.08
        N = max(int(dur * sr), 16)
        t = np.arange(N) / sr
        onset = np.clip((t - 0.15) / 0.25, 0, 1)                       # vibrato fades in
        vib = vdepth * onset * np.sin(2 * np.pi * vrate * t + rng.random() * 6.28)
        f0 = 440 * 2 ** ((n.pitch - 69 + (cents + vib) / 100) / 12)
        phase = np.cumsum(f0 / sr)
        src = 2 * (phase % 1.0) - 1                                    # sawtooth glottal-ish source
        src = lfilter([1 - 0.55 * (1 - bright)], [1, -0.55 * (1 - bright)], src)  # tilt
        v = VOWELS[_vowel_of(n.lyric)]
        voiced = sum(lfilter(*_resonator(f * fscale, bw, sr), src) * g
                     for f, bw, g in zip(v, (90, 110, 160), (1.0, 0.6, 0.3)))
        noise = lfilter(*_resonator(3500, 2500, sr), nrng.standard_normal(N)) * breath * 6
        sig = voiced + noise
        a, r = int(0.03 * sr), int(0.08 * sr)
        env = np.ones(N)
        env[:a] = np.linspace(0, 1, a)
        env[-r:] = np.linspace(1, 0, r)
        sig = sig * env
        peak = np.max(np.abs(sig)) or 1.0
        return (sig / peak * (0.15 + 0.5 * n.vel)).astype(np.float32)
