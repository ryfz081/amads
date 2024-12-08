from pytest import approx, fixture
from musmart.algorithm.slice import get_timepoints, salami_slice
from musmart.core.basics import Note
from musmart.io.pt_midi_import import partitura_midi_import
from musmart.music import example


def test_timepoints_twochan(twochan_notes):
    timepoints = get_timepoints(twochan_notes, time_n_digits=6)
    assert len([t for t in timepoints if len(t.note_ons) > 0]) == 16


def test_salami_slice_twochan(twochan_score):
    chords = salami_slice(twochan_score)
    assert len(chords) == 16

    pitches = [[int(p.keynum) for p in c] for c in chords]
    assert pitches == [
        [43, 67],
        [43, 64],
        [43, 67],
        [43, 65],
        [45, 67],
        [45, 64],
        [43, 67],
        [43, 71],
        [40, 69],
        [40, 71],
        [38, 74],
        [38, 71],
        [43, 69],
        [43, 67],
        [47, 64],
        [47, 67]
    ]

    for chord in chords:
        assert chord.duration == approx(1.0, abs=0.01)
