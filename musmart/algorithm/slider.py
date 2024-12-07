"""

Author: Peter Harrison
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List
from musmart.core.basics import Note, Score


EndOfSequence = object()


@dataclass
class Timepoint:
    id: int
    time: float
    note_ons: list[Note]
    note_offs: list[Note]
    sounding_notes: set[Note]

    @property
    def last_note_end(self):
        return max(n.offset + n.dur for n in self.sounding_notes)


@dataclass
class Timeline:
    timepoints: List[Timepoint]
    time_n_decimals: int

    def __len__(self):
        return len(self.timepoints)

    def __getitem__(self, index: int) -> Timepoint:
        return self.timepoints[index]

    def __iter__(self):
        return iter(self.timepoints)

    @classmethod
    def from_score(cls, score: Score, time_n_decimals: int = 6) -> "Timeline":
        notes = cls.get_notes(score)

        note_ons = defaultdict(list)
        note_offs = defaultdict(list)

        for note in notes:
            note_on = round(note.offset, time_n_decimals)
            note_off = round(note.offset + note.dur, time_n_decimals)

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

        return cls(timepoints, time_n_decimals)

    @classmethod
    def get_notes(cls, score: Score) -> List[Note]:
        notes = list(score.flatten().collapse_parts().find_all(Note))
        notes.sort(key=lambda n: (n.offset, n.keynum))
        return notes


@dataclass
class Slice:
    notes: List[Note]
    original_notes: List[Note]
    start: float
    end: float

    def __iter__(self):
        return iter(self.notes)

    def __len__(self):
        return len(self.notes)


# TODO:
def salami_slice(
        score: Score,
        min_slice_duration: float = 0.0,
        include_empty_slices: bool = True,
        slice_on_note_start: bool = True,
        slice_on_note_end: bool = True,
) -> List[Slice]:
    timeline = Timeline.from_score(score)
    slices = []

    for i, timepoint in enumerate(timeline):
        if (
            (slice_on_note_start and len(timepoint.note_ons) > 0)
            or (slice_on_note_end and len(timepoint.note_offs) > 0)
        ):

            assert len(timepoint.sounding_notes) > 0

            try:
                next_timepoint = timeline[i + 1]
            except IndexError:
                next_timepoint = None

            slice_start = timepoint.time

            if next_timepoint is None:
                slice_end = timepoint.last_note_end
            else:
                slice_end = next_timepoint.time

            slice_duration = slice_end - slice_start

            if slice_duration < min_chord_duration:
                continue

            notes = [note.copy() for note in timepoint.sounding_notes]
            for note in notes:
                note.offset = slice_start
                note.dur = slice_end - slice_start

            slices.append(Slice(
                notes=notes,
                original_notes=timepoint.sounding_notes,
                start=slice_start,
                end=slice_end,
            ))

    return slices
