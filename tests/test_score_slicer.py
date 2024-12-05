

from musmart.algorithm.slider import chordify
from musmart.io.pt_midi_import import partitura_midi_import
from musmart.music import example


def test_chordify():
    midi_file = example.fullpath("midi/twochan.mid")
    score = partitura_midi_import(midi_file, ptprint=False)
    chords = chordify(score)
    
    assert len(chords) == 16
    pitches = [[int(p.keynum) for p in c] for c in chords]
    assert pitches == [
        [43, 67], 
        [43, 64], 
        [43, 67], 
        [43, 65], 
        [45, 67], 
        [64], 
        [43, 67], 
        [71], 
        [40, 69], 
        [71], 
        [38, 74], 
        [71], 
        [43, 69], 
        [67], 
        [47, 64], 
        [67],
    ]
