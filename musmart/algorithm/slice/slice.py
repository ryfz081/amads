from typing import List

from ...core.basics import Note

class Slice:
    def __init__(
            self,
            notes: List[Note],
            original_notes: List[Note],
            start: float,
            end: float,
        ):
        self.notes = notes
        self.original_notes = original_notes
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(self.notes)

    def __len__(self):
        return len(self.notes)

    @property
    def duration(self):
        return self.end - self.start

    @property
    def is_empty(self):
        return len(self.notes) == 0
