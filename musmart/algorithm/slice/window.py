from typing import Iterable, Iterator, Optional, Union

from .slice import Slice
from ...core.basics import Note, Score
from ...utils import float_range


class Window(Slice):
    def __init__(
            self,
            time: float,
            size: float,
            align: str,
            candidate_notes: Iterable[Note],
            skip: int = 0,
    ):
        # TODO: document that candidate_notes must be ordered by offset and pitch

        match align:
            case "left":
                start = time
            case "center":
                start = time - size / 2
            case "right":
                start = time - size
            case _:
                raise ValueError(f"Invalid value passed to `align`: {align}")

        end = start + size

        self.time = time
        self.size = size
        self.align = align

        original_notes = []
        notes = []

        candidate_notes = list(candidate_notes)

        for i in range(skip, len(candidate_notes)):
            note = candidate_notes[i]

            if note.end_offset < start:
                # The note finished before the window started.
                # It'll definitely finish before future windows start,
                # because they'll be even later, so we can skip it then too.
                skip = i
                continue

            if note.offset > end:
                # The note starts after the window finishes.
                # All the remaining notes in candidate_notes will have even later offsets,
                # so we don't need to check them for this window.
                # They might be caught by future windows though.
                break

            original_notes.append(note)

            # We use copy instead of creating a new Note because we want to
            # preserve any other attributes that might be useful in downstream tasks.
            note = note.copy()
            note.offset = max(note.offset, start)
            note.dur = min(note.dur, end - note.offset)

            notes.append(note)

        # The next window can look at this attribute to know which candidates can be skipped.
        self.skip = skip

        super().__init__(notes=notes, original_notes=original_notes, start=start, end=end)


def window_slice(
        passage: Union[Score, Iterable[Note]],
        size: float,
        step: float = 1.0,
        align: str = "right",
        start: float = 0.0,
        end: Optional[float] = None,
        times: Optional[Iterable[float]] = None,
) -> Iterator[Window]:
    """
    Slice a score into (possibly overlapping) slices of a given size.

    Parameters
    ----------

    passage :
        The passage to slice.

    size :
        The size of each slice (time units).

    step :
        The step size to to take between slices (time units).
        For example, if step is 0.1, then a given slice will start 0.1 time units
        after the previous slice started. Note that if step is smaller than size,
        successive slices will overlap.

    align :
        Each generated window has a `time` property that points to a
        particular timepoint in the musical passage. The `align` parameter determines
        how the window is aligned to this timepoint.

        - "left" : the window starts at ``slice.time``
        - "center" : ``window.time`` corresponds to the midpoint of the window
        - "right" : the window finishes at ``slice.time``

    start :
        The desired time of the first slice (defaults to 0.0).

    end :
        If set, the windowing will stop once the end time is reached.
        Following the behaviour of Python's built-in range function,
        ``end`` is not treated inclusively, i.e. the last slice will
        not include ``end``.

    times :
        Optional iterable of times to generate slices for. If provided,
        `start` and `end` are ignored.
    """
    if isinstance(passage, Score):
        if not passage.is_flattened_and_collapsed():
            raise NotImplementedError(
                "Currently this function only supports flattened and collapsed scores. "
                "You can flatten a score using `score.flatten(collapse=True)`."
            )
        notes = passage.find_all(Note)
    else:
        notes = passage

    notes = list(notes)
    notes.sort(key=lambda n: (n.offset, n.pitch))

    if times is None:
        window_times = float_range(start, end, step)
    else:
        for par, default in [("start", 0.0), ("end", None), ("step", 1.0)]:
            provided = globals()[par]
            if provided != default:
                raise ValueError(f"`{par}` was set to {provided} but `times` was also provided")

        window_times = times

    skip = 0

    for time in window_times:
        window = Window(time, size, align, notes, skip)

        yield window

        skip = window.skip

        if skip + 1 == len(notes):
            break
