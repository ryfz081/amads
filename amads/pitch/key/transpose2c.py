"""
Tranposes a score to C major or minor according to Krumhansl-Kessler
algorithm.
Date: [2025-03-13]
Description:
    Returns a new score that is tranposed to C major or minor
    according to Krumhansl-Kessler algorithm.
Usage:
    [Add basic usage examples or import statements]
Original doc: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=6e06906ca1ba0bf0ac8f2cb1a929f3be95eeadfa#page=93
Reference(s):
    [see kkcc.py]
"""

import profiles as prof
from kkcc import kkcc

from ...core.basics import Note, Score


def transpose2c(score: Score, standardProfile: dict = prof.KrumhanslKessler):
    """
    Define a pitch fingerprint of a score as the unweighted pitch distribution
    of the score.
    Define a standard pitch fingerprint as a reference pitch histogram
    for a standard major and minor key (typically C major and minor)
    that is obtained through experimentation or data analysis.
    In the case of Krumhansl and Kessler (1982), the standard pitch fingerprint
    was obtained by having a controlled group of somewhat trained musicians
    rating key likeness of a standardized set of slightly altered scores
    (more details in the paper). Note that, in the experiment,
    the scores are presented through playback, so the ratings were done through
    musician's ears.

    One thing of note in the implementation is that: I am too lazy to not use
    numpy and write some of these stats functions by hand.
    So a lot of the implementation is reminiscent of what the matlab
    implementation does, but in numpy.

    Parameters
    ----------
    score
        Input score (class from core.basics for storing music scores)
        Note that there are no restrictions for the score here
    standardProfile
        the profile dictionary that contains the standard pitch fingerprint.
        Structure of this dictionary abides to the dictionaries in profiles.py,
        mainly the major, major_sum, minor, and minor_sum.
    Returns
    -------
    A tranposed score to C major or C minor

    Notes
    -----
    Implementation based on the original MATLAB code from:
    https://github.com/miditoolbox/1.1/blob/master/miditoolbox/transpose2c.m
    Examples
    --------
    [Unfortunately, still having problem resolving score construction]
    """
    kkccResult = kkcc(score, standardProfile)
    transposeOffset = kkccResult.index(max(kkccResult)) % 12
    transposedScore = score.deep_copy()
    for note in transposedScore.find_all(Note):
        note.pitch.keynum -= transposeOffset
    return transposedScore
