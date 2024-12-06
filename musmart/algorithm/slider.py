"""

Author: Peter Harrison
"""

from typing import List
from musmart.core.basics import Note, Score


EndOfSequence = object()

from dataclasses import dataclass

@dataclass
class Timepoint:
    id: int
    time: float 
    note_ons: list[Note]
    note_offs: list[Note]
    sounding_notes: set[Note]


class ScoreSlicer:
    time_n_decimals = 6

    def slice(self, score: Score):
        timepoints = self.get_timepoints(score)
        return self._slice(timepoints)
    
    def _slice(self, timepoints: List[Timepoint]):
        raise NotImplementedError

    def get_timepoints(self, score: Score) -> List[Timepoint]:
        notes = self.get_notes(score)

        from collections import defaultdict

        note_ons = defaultdict(list)
        note_offs = defaultdict(list)
        
        for note in notes:
            note_on = round(note.offset, self.time_n_decimals)
            note_off = round(note.offset + note.dur, self.time_n_decimals)

            note_ons[note_on].append(note)
            note_offs[note_off].append(note)

        times = sorted(set(note_ons.keys()) | set(note_offs.keys()))

        timepoints = []
        sounding_notes = set()
        
        for i, time in enumerate(times):
            for note in note_offs[time]:
                sounding_notes.discard(note)

            for note in note_ons[time]:
                sounding_notes.add(note)

            timepoints.append(Timepoint(
                id=i, 
                time=time, 
                note_ons=note_ons[time], 
                note_offs=note_offs[time],
                sounding_notes=sorted(list(sounding_notes), key=lambda n: n.keynum),
            ))

        return timepoints

    def get_notes(self, score: Score):
        notes = list(score.flatten().collapse_parts().find_all(Note))
        notes.sort(key=lambda n: (n.offset, n.keynum))
        return notes
    

class Chordifier(ScoreSlicer):
    def __init__(self, slice_on_note_start: bool = True, slice_on_note_end: bool = False):
        assert slice_on_note_start or slice_on_note_end
        self.slice_on_note_start = slice_on_note_start
        self.slice_on_note_end = slice_on_note_end

    def _slice(self, timepoints: List[Timepoint]):
        slices = []
        for timepoint in timepoints:
            if (
                (self.slice_on_note_start and len(timepoint.note_ons) > 0)
                or (self.slice_on_note_end and len(timepoint.note_offs) > 0)
            ):
                # TODO: decide whether to edit the offsets of the notes
                # I think the answer is yes
                slices.append(timepoint.sounding_notes)

        return slices
        