from musmart.algorithm.slider import Timeline, salami_slice
from musmart.io.pt_midi_import import partitura_midi_import
from musmart.music import example


def test_chordify_twochan():
    midi_file = example.fullpath("midi/twochan.mid")
    score = partitura_midi_import(midi_file, ptprint=False)

    timeline = Timeline.from_score(score)
    assert len([t for t in timeline if len(t.note_ons) > 0]) == 16

    chords = salami_slice(score)

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
