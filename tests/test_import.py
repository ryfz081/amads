
import pytest

import pretty_midi

from musmart.core.basics import Note, Score
from musmart.io.pt_midi_import import partitura_midi_import
from musmart.music import example



def test_import_midi():
    """
    Test MIDI import by comparing the results with pretty_midi.
    """
    midi_file = example.fullpath("midi/sarabande.mid")
    score = partitura_midi_import(midi_file, ptprint=False)
    assert isinstance(score, Score)

    score_notes = list(score.find_all(Note))

    pm = pretty_midi.PrettyMIDI(str(midi_file))
    pm_notes = [note for instrument in pm.instruments for note in instrument.notes]

    assert len(score_notes) == len(pm_notes)

    flattened_score = score.flatten()
    flattened_notes = list(flattened_score.find_all(Note))

    assert len(flattened_notes) == len(pm_notes)

    flattened_notes.sort(key=lambda x: (x.offset, x.pitch.keynum))
    pm_notes.sort(key=lambda x: (x.start, x.pitch))

    quarter_note_duration = pm_notes[1].start / flattened_notes[1].offset

    for score_note, pm_note in zip(flattened_notes, pm_notes):
        assert score_note.pitch.keynum == pm_note.pitch
        assert score_note.offset == pytest.approx(pm_note.start / quarter_note_duration)
        assert score_note.dur == pytest.approx(pm_note.end / quarter_note_duration - pm_note.start / quarter_note_duration)
