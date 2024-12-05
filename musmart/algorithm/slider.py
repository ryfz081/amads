"""

Author: Peter Harrison
"""

from typing import List
from musmart.core.basics import Note, Score


EndOfSequence = object()

class ScoreSlicer:
    def slide_over(self, score: Score):
        score = score.flatten().collapse_parts()
        notes = list(score.find_all(Note))
        notes.sort(key=lambda n: (n.offset, n.keynum))
        notes.append(EndOfSequence)
        
        # slice = []
        last_offset = None

        for note in notes:
            assert last_offset is None or note.offset >= last_offset

            self.consume_note(note)

            # if self.slice_ready(slice, next_note=note):
            #     yield self.finalize_slice(slice, next_note=note)
            #     slice = []

            # slice.append(note)

        
            

    # def slice_ready(self, slice: List[Note]):
    #     raise NotImplementedError
    
    # def finalize_slice(self, slice: List[Note], next_note: Note):
    #     raise NotImplementedError


class Chordifier(ScoreSlicer):
    def __init__(self):
        self.sounding_notes = []
        self.pending_slice = []
        self.slices = []

    def consume_note(self, note: Note):
        assert note.tie is None

        if self.should_start_new_slice(note):
            self.finalize_slice(self.pending_slice, note)  # todo - make slice a class
            self.slices.append(self.pending_slice)
            self.pending_slice = []

        self.pending_slice.append(note)






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


def chordify(score: Score):
    chordifier = Chordifier()
    return list(chordifier.slide_over(score))
