from typing import Iterable

from pytest import approx

from musmart.core.basics import Score, Note
from musmart.algorithm.slice.salami import window_slice


def test_window_slice(twochan_score: Score, twochan_notes: Iterable[Note]):
    size = 4.0  # one bar
    step = 0.01

    score = twochan_score.flatten(collapse=True)

    last_note = twochan_notes[-1]
    last_note_off = last_note.offset + last_note.dur
    assert last_note_off == approx(16.0)

    windows = window_slice(
        score,
        size=size,
        step=step,
        align="right",
    )
    windows = list(windows)

    first_slice = windows[0]
    assert first_slice.start == approx(-4.0)
    assert first_slice.end == approx(0.0)
    assert first_slice.is_empty

    last_slice = windows[-1]
    assert last_slice.start == approx(last_note_off, abs=step * 3)
    assert last_slice.is_empty

    # The second slice should include the score's opening two notes
    assert len(windows[1].notes) == 2
    assert windows[1].notes[0].keynum == 67 - 2 * 12
    assert windows[1].notes[1].keynum == 67
