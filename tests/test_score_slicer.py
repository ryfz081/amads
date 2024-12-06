from musmart.algorithm.slider import Chordifier, chordify
from musmart.io.pt_midi_import import partitura_midi_import
from musmart.music import example


def test_chordify():
    midi_file = example.fullpath("midi/twochan.mid")
    score = partitura_midi_import(midi_file, ptprint=False)

    timepoints = Chordifier().get_timepoints(score)

    assert len([t for t in timepoints if len(t.note_ons) > 0]) == 16

    chords = chordify(score)
    
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
