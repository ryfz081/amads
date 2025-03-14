"""
Correlations of pitch-class distribution with Krumhansl-Kessler tonal
hierarchies
Date: [2025-03-11]
Description:
    Computes correlation coefficients of a score's pitch distribution
    to a specified standard pitch histogram that is key transposed
    over all 24 standard keys
Usage:
    [Add basic usage examples or import statements]
Original doc: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=6e06906ca1ba0bf0ac8f2cb1a929f3be95eeadfa#page=68
Reference(s):
    Albrecht, J., & Shanahan, D. (2013). The Use of Large Corpora to
        Train a New Type of Key-Finding Algorithm. Music Perception: An
        Interdisciplinary Journal, 31(1), 59-67.

    Krumhansl, C. L. (1990). Cognitive Foundations of Musical Pitch.
        New York: Oxford University Press.

    Huron, D., & Parncutt, R. (1993). An improved model of tonality
        perception incorporating pitch salience and echoic memory.
        Psychomusicology, 12, 152-169.

    Temperley, D. (1999). What's key for key? The Krumhansl-Schmuckler
        key-finding algorithm reconsidered. Music Perception: An Interdisciplinary
        Journal, 17(1), 65-100.
"""

from collections import deque

import numpy as np
import profiles as prof

from ...core.basics import Score
from ..pcdist1 import pcdist1


def constructFingerprintMatrix(standardProfile: dict = prof.KrumhanslKessler):
    """
    Constructs a 12x24 numpy matrix denoting the histograms
    for each key.

    Parameters
    ----------
    standardProfile
        the profile dictionary that contains the standard pitch fingerprint.
        Structure of this dictionary abides to the dictionaries in profiles.py,
        mainly the major, major_sum, minor, and minor_sum.

    Returns
    -------
        Constructed 12x24 numpy matrix that denote the histograms
    """
    majorDeque = deque(standardProfile["major"])
    majorMat = np.zeros((12, 12))
    for i in range(majorMat.shape[0]):
        majorMat[i] = majorDeque
        majorDeque.rotate(1)
    minorDeque = deque(standardProfile["minor"])
    minorMat = np.zeros((12, 12))
    for i in range(minorMat.shape[0]):
        minorMat[i] = minorDeque
        minorDeque.rotate(1)
    return np.concatenate((majorMat, minorMat), axis=0)


def kkcc(
    score: Score,
    standardProfile: dict = prof.KrumhanslKessler,
    salienceFlag: bool = False,
):
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

    Parameters
    ----------
    score
        Input score (class from core.basics for storing music scores)
        Note that there are no restrictions for the score here
    standardProfile
        the profile dictionary that contains the standard pitch fingerprint.
        Structure of this dictionary abides to the dictionaries in profiles.py,
        mainly the major, major_sum, minor, and minor_sum.
    salienceFlag
        Whether or not we need to abide to the Huron and Parncutt (1988)
        key-finding algorithm.
        Default when false is the Krumhansl and Kessler (1982) algorithm.
    Returns
    -------
    List of correlation coefficients computed by the Pearson correlation
    coefficient formula

    Notes
    -----
    Implementation based on the original MATLAB code from:
    https://github.com/miditoolbox/1.1/blob/master/miditoolbox/kkcc.m
    Examples
    --------
    [Unfortunately, still having problem resolving score construction]
    """
    pcd = np.array([pcdist1(score, False)])

    if salienceFlag:
        sal = deque([1, 0, 0.25, 0, 0, 0.5, 0, 0, 0.33, 0.17, 0.2, 0])
        salm = np.zeros((12, 12))
        for i in range(salm.shape[0]):
            salm[i] = sal
            sal.rotate(1)
        pcd = np.matmul(pcd, salm)

    profileMat = constructFingerprintMatrix(standardProfile)

    resultArray = np.corrcoef(np.concatenate((pcd, profileMat), axis=0))
    # this should be a 24 element array containing all the
    # key distance coefficients...
    result = list(resultArray[0, 1:])
    assert len(result) == 24
    return result
