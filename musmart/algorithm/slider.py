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


# TODO: refactor to use a generator
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



def salami_slice(
        score: Score,
        remove_duplicated_pitches: bool = True,
        include_empty_slices: bool = False,
        include_note_end_slices: bool = True,
        min_duration_for_note_end_slices: float = 0.01,
) -> List[Slice]:
    timeline = Timeline.from_score(score)
    slices = []

    for i, timepoint in enumerate(timeline):
        if (
            len(timepoint.note_ons) > 0
            or (include_note_end_slices and len(timepoint.note_offs) > 0)
        ):
            try:
                next_timepoint = timeline[i + 1]
            except IndexError:
                next_timepoint = None

            is_last_timepoint = next_timepoint is None
            is_empty_slice = len(timepoint.sounding_notes) == 0
            is_note_end_slice = len(timepoint.note_ons) == 0 and len(timepoint.note_offs) > 0

            if is_empty_slice:
                if not include_empty_slices:
                    continue
                if is_last_timepoint:
                    # Don't include empty slices at the end of the score
                    continue

            slice_start = timepoint.time

            if next_timepoint is None:
                if len(timepoint.sounding_notes) == 0:
                    continue
                else:
                    slice_end = timepoint.last_note_end
            else:
                slice_end = next_timepoint.time

            slice_duration = slice_end - slice_start

            if is_note_end_slice and slice_duration < min_duration_for_note_end_slices:
                continue

            pitches = [note.pitch for note in timepoint.sounding_notes]
            if remove_duplicated_pitches:
                pitches = sorted(set(pitches))

            notes = [
                Note(
                    offset=slice_start,
                    dur=slice_duration,
                    pitch=pitch,
                )
                for pitch in pitches
            ]

            slices.append(Slice(
                notes=notes,
                original_notes=timepoint.sounding_notes,
                start=slice_start,
                end=slice_end,
            ))

    return slices
