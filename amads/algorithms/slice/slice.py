from typing import List

from ...core.basics import Note


class Slice:
    """A slice of a musical score between two timepoints.

    This is the base class for different slicing algorithms like salami slicing and
    windowing. A slice contains a list of notes that are sounding between its start
    and end times, as well as references to the original notes from which these
    were derived.

    Parameters
    ----------
    notes : List[Note]
        The notes in this slice, with durations truncated to fit within the slice
        boundaries if necessary
    original_notes : List[Note]
        The original unmodified notes from which the slice notes were derived
    start : float
        The start time of the slice
    end : float
        The end time of the slice
    """

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
        """Iterate over the notes in this slice.

        Returns
        -------
        Iterator[Note]
            Iterator over the notes
        """
        return iter(self.notes)

    def __len__(self):
        """Get the number of notes in this slice.

        Returns
        -------
        int
            Number of notes
        """
        return len(self.notes)

    @property
    def duration(self):
        """Get the duration of this slice.

        Returns
        -------
        float
            Duration in time units
        """
        return self.end - self.start

    @property
    def is_empty(self):
        """Check if this slice contains any notes.

        Returns
        -------
        bool
            True if the slice contains no notes
        """
        return len(self.notes) == 0
