"""

Author: Peter Harrison
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union
from musmart.core.basics import Note, Score


@dataclass
class Timepoint:
    time: float
    note_ons: list[Note] = field(default_factory=list)
    note_offs: list[Note] = field(default_factory=list)
    sounding_notes: set[Note] = field(default_factory=set)

    @property
    def last_note_end(self):
        return max(n.offset + n.dur for n in self.sounding_notes)


def get_timepoints(notes: List[Note], time_n_digits: Optional[int] = None) -> List[Timepoint]:
    note_ons = defaultdict(list)
    note_offs = defaultdict(list)

    for note in notes:
        note_on = note.offset
        note_off = note.offset + note.dur

        if time_n_digits is not None:
            note_on = round(note_on, time_n_digits)
            note_off = round(note_off, time_n_digits)

        note_ons[note_on].append(note)
        note_offs[note_off].append(note)

    times = sorted(set(note_ons.keys()) | set(note_offs.keys()))

    timepoints = []
    sounding_notes = set()

    for time in times:
        for note in note_offs[time]:
            sounding_notes.discard(note)

        for note in note_ons[time]:
            sounding_notes.add(note)

        timepoints.append(Timepoint(
            time=time,
            note_ons=note_ons[time],
            note_offs=note_offs[time],
            sounding_notes=sorted(list(sounding_notes), key=lambda n: n.keynum),
        ))

    return timepoints


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

    @property
    def duration(self):
        return self.end - self.start



def salami_slice(
        passage: Union[Score, Iterable[Note]],
        remove_duplicated_pitches: bool = True,
        include_empty_slices: bool = False,
        include_note_end_slices: bool = True,
        min_slice_duration: float = 0.01,
) -> List[Slice]:
    if isinstance(passage, Score):
        notes = passage.flatten(collapse=True).find_all(Note)
    else:
        notes = passage

    timepoints = get_timepoints(notes)

    slices = []

    for i, timepoint in enumerate(timepoints):
        if (
            len(timepoint.note_ons) > 0
            or (include_note_end_slices and len(timepoint.note_offs) > 0)
        ):
            try:
                next_timepoint = timepoints[i + 1]
            except IndexError:
                next_timepoint = None

            is_last_timepoint = next_timepoint is None
            is_empty_slice = len(timepoint.sounding_notes) == 0

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

            if slice_duration < min_slice_duration:
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
