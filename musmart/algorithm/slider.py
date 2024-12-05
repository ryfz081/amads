"""

Author: Peter Harrison
"""

from typing import List
from musmart.core.basics import Note, Score


EndOfSequence = object()

class Slider:
    def slide_over(self, score: Score):
        score = score.flatten().collapse_parts()
        notes = list(score.find_all(Note))
        notes.sort(key=lambda n: n.offset)
        notes.append(EndOfSequence)
        
        slice = []
        last_offset = None

        for note in notes:
            assert last_offset is None or note.offset >= last_offset

            if self.slice_ready(slice, next_note=note):
                yield self.finalize_slice(slice, next_note=note)
                slice = []

            slice.append(note)
            

    def slice_ready(self, slice: List[Note]):
        raise NotImplementedError
    
    def finalize_slice(self, slice: List[Note], next_note: Note):
        raise NotImplementedError


class ChordifySlider(Slider):
    def slice_ready(self, slice: List[Note], next_note: Note):
        if len(slice) == 0:
            return False
        elif next_note == EndOfSequence:
            return True
        else:
            return slice[0].offset < next_note.offset
    
    def finalize_slice(self, slice: List[Note], next_note: Note):
        return [
            self.update_note(note, next_note)
            for note in slice
        ]
    
    def update_note(self, note: Note, next_note: Note):
        note = note.copy()
        if next_note != EndOfSequence:
            note.dur = next_note.offset - note.offset
        return note
