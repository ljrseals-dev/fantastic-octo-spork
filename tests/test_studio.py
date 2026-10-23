import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.io import wavfile
from vocal_studio import *

def melody():
    return [Note(p, i*0.6, 0.5, 0.8, "la") for i, p in enumerate([60, 62, 64, 65, 67])]

def test_diatonic_thirds_in_c_major():
    k = Key(0, "major")
    up = [n.pitch for n in harmonize(melody(), k, "3rd_up")]
    assert up == [64, 65, 67, 69, 71], up        # E F G A B  (major/minor thirds as the scale dictates)
    low = [n.pitch for n in harmonize(melody(), k, "3rd_low")]
    assert low == [57, 59, 60, 62, 64], low      # A B C D E

def test_minor_key_and_harmony_stays_in_scale():
    k = Key(9, "minor")                         # A minor
    lead = [Note(p, i, 0.5) for i, p in enumerate([69, 71, 72, 74, 76])]
    for iv in ("3rd_up", "5th_up", "3rd_low", "6th_up"):
        assert all(k.in_scale(n.pitch) for n in harmonize(lead, k, iv)), iv

def test_octave_and_range_clamp():
    k = Key(0)
    hi = harmonize([Note(80, 0, 1)], k, "octave_up", hi=84)
    assert hi[0].pitch == 80          # 92 is out of range, folded down an octave back to 80

def test_chromatic_note_keeps_inflection():
    k = Key(0)
    assert abs(k.shift(61, 2) - 65) < 1e-9     # C# treated as C+1: third up is E+1, inflection preserved

def test_stack_sizes_and_independence():
    for parts in (2, 3, 4, 6):
        tr = build_stack(melody(), Key(), parts)
        assert len(tr) == parts and tr[0].role == "lead"
    tr = build_stack(melody(), Key(), 4)
    tr[1].notes[0].pitch += 1
    assert tr[0].notes[0].pitch == 60            # editing one part never touches another

def test_bad_inputs():
    for fn in (lambda: build_stack(melody(), Key(), 5), lambda: Key(0, "lydian-ish")):
        try: fn()
        except ValueError: continue
        raise AssertionError("expected ValueError")

def test_render_mix_export_and_mute_solo():
    s = Session()
    s.tracks = build_stack(melody(), Key(), 3) + build_backgrounds(melody(), Key(), "soulful")
    m = VocalMixer(s)
    full = m.mix()
    assert full.ndim == 2 and full.shape[1] == 2 and np.max(np.abs(full)) <= 1.0 and np.max(np.abs(full)) > 0.1
    s.track("Harm High").settings.mute = True
    assert len(s.active_tracks()) == len(s.tracks) - 1
    s.track("Lead").settings.solo = True
    assert [t.name for t in s.active_tracks()] == ["Lead"]
    s.track("Lead").settings.solo = False
    with tempfile.TemporaryDirectory() as d:
        paths = m.export(d)
        sr, mix = wavfile.read(paths["mix"])
        assert sr == 44100 and mix.dtype == np.int16
        lens = {wavfile.read(p)[1].shape[0] for k, p in paths.items()}
        assert len(lens) == 1                    # all stems and the mix share one length

def test_pitch_is_correct_in_rendered_audio():
    s = Session(); t = Track("Lead", "lead", [Note(69, 0.0, 1.0)], TrackSettings(vibrato_depth_cents=0))
    y = ParametricRenderer().render_track(t, 44100, 100)
    seg = y[4410:4410 + 8820]                     # 0.2 s steady state
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    lag = np.argmax(ac[40:400]) + 40              # autocorrelation peak = true fundamental period
    assert abs(44100 / lag - 440) < 6, 44100 / lag
