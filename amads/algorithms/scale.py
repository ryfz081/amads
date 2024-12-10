"""

Original doc: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=6e06906ca1ba0bf0ac8f2cb1a929f3be95eeadfa#page=88

Scales note data in given dimension (offset, dur, or both)

Input arguments:
SCORE = score object/other EventGroup object (Note: This method MODIFIES the score object input)
FACTOR = amount of scale (must be > 0)
DIM = dimension ('offset', 'dur','all')
    - 'offset': scales the offset of ALL EVENTS
    - 'dur': scales the durations of ALL NON-EVENTGROUP EVENTS (Note, Rest)
    - 'all': scales both the offset and the duration

Output:
SCORE = the scaled version of the input object

Examples:
scaled_score = scale(score.copy(),'offset',2); % scales offset by a factor of 2
scaled_score = scale(score.copy(),'dur',0.5); % shortens durations by a factor of 2

"""

from ..core.basics import EventGroup


def scale(score, factor=2, dim="all"):
    if dim == "all":
        scale(score, factor, "dur")
        scale(score, factor, "offset")
        return score
    for elem in score.content:
        if isinstance(elem, EventGroup):
            scale(elem, factor, dim)
            if dim == "offset":
                elem.offset *= 2
        else:
            if dim == "dur":
                elem.dur *= factor
            elif dim == "offset":
                elem.offset *= 2
        score.dur = max(score.dur, elem.end_offset)
    return score
