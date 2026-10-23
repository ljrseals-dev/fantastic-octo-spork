# AOC Magic — Vocal Studio (milestone 1)

Multi-track vocal arrangement engine. Drop the `vocal_studio/` folder next to `aoc_magic.py`.
Needs only `numpy` and `scipy`.

    lead notes (list of Note, or MIDI you convert)  ->  harmony / doubles / backgrounds
        -> VocalRenderer (replaceable)  ->  VocalMixer (EQ, de-ess, comp, sat, pan, reverb, echo)
        -> stems + vocal_mix.wav + session.json

Run `python demo.py` for an example (writes to `out/`). Tests: `python -m pytest tests` or run the file directly.

## Status
Done: track data model, scale-aware harmony (2/3/4/6 parts), doubles, backgrounds and ad-libs,
mute/solo/reorder, per-track timing/tuning humanization, regenerate-by-seed, stems + mix export, session save/load.
Not done: pitch tracking from a guide vocal (needs librosa/pyin), MIDI file import, GUI wiring,
enrolled-voice rendering. `ParametricRenderer` is a basic formant synth for previewing arrangements, not a realistic voice.
Implement `VocalRenderer.render_track` to plug in a better singing backend or an authorized voice-conversion model.
