from typing import Optional, Iterator


def float_range(start: float, end: Optional[float], step: float) -> Iterator[float]:
    """Generate a range of floats.

    Similar to Python's built-in range() function but supports floating point numbers.
    If end is None, generates an infinite sequence.

    Parameters
    ----------
    start : float
        The starting value of the range
    end : float or None
        The end value of the range (exclusive). If None, generates an infinite sequence
    step : float
        The increment between values

    Yields
    ------
    float
        The next value in the sequence
    """
    curr = start
    while end is None or curr < end:
        yield curr
        curr += step
