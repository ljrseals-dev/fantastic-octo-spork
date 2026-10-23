"""Multi-track vocal data model. Every part is an independent, editable track."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Note:
    pitch: float          # MIDI note number (60 = C4); fractional = cents
    start: float          # seconds
    dur: float            # seconds
    vel: float = 0.8      # 0..1
    lyric: str = "ah"     # syllable; vowel picks the formant shape

    @property
    def end(self) -> float:
        return self.start + self.dur


@dataclass
class TrackSettings:
    level_db: float = 0.0
    pan: float = 0.0              # -1 (L) .. +1 (R)
    delay_ms: float = 0.0         # timing offset of the whole part
    reverb: float = 0.2           # 0..1 send
    echo: float = 0.0             # 0..1 send
    echo_ms: float = 280.0
    mute: bool = False
    solo: bool = False
    # humanization (0 = machine-tight)
    timing_ms: float = 0.0
    tuning_cents: float = 0.0
    seed: int = 0                 # change to "regenerate" a take
    # voice shaping
    vibrato_depth_cents: float = 25.0
    vibrato_rate_hz: float = 5.6
    breath: float = 0.04
    brightness: float = 0.5


@dataclass
class Track:
    name: str
    role: str                     # lead | harmony | double | background | adlib
    notes: list[Note] = field(default_factory=list)
    settings: TrackSettings = field(default_factory=TrackSettings)
    interval: str = "unison"      # how it was derived, for display / regeneration


@dataclass
class Session:
    tempo: float = 100.0
    key_tonic: int = 0            # pitch class, 0 = C
    key_mode: str = "major"
    sr: int = 44100
    tracks: list[Track] = field(default_factory=list)

    def track(self, name: str) -> Track:
        for t in self.tracks:
            if t.name == name:
                return t
        raise KeyError(name)

    def active_tracks(self) -> list[Track]:
        solos = [t for t in self.tracks if t.settings.solo]
        if solos:
            return solos
        return [t for t in self.tracks if not t.settings.mute]

    def reorder(self, names: list[str]) -> None:
        self.tracks = [self.track(n) for n in names]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        d = json.loads(Path(path).read_text())
        tracks = [Track(name=t["name"], role=t["role"], interval=t.get("interval", "unison"),
                        notes=[Note(**n) for n in t["notes"]],
                        settings=TrackSettings(**t["settings"])) for t in d.pop("tracks")]
        return cls(tracks=tracks, **d)
