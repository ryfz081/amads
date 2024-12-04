#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements various functions related to linear tempo modeling e.g., `tempo_slope`, `tempo_drift`"""

import numpy as np

from scipy.stats import linregress


def tempo_slope(beats: np.array) -> float:
    """Return the tempo slope for an array of beats.

    The tempo slope is the signed overall tempo change per second within a performance, equivalent
    to the slope of a linear regression of instantaneous tempo against beat onset time
    such that a negative slope implies deceleration over time and a positive slope acceleration [1].

    Arguments:
        beats (np.array): an array of beat timestamps in seconds corresponding to e.g., quarter notes

    Returns:
        float: the tempo slope value

    Examples:
        >>> my_beats = np.array([1., 2., 3., 4.])    # stable performance, slope == 0
        >>> tempo_slope(my_beats) == 0.
        True

        >>> my_beats = np.array([1., 1.9, 2.7, 3.5])    # accelerating performance, slope > 0
        >>> tempo_slope(my_beats) > 0
        True

        >>> my_beats = np.array([1., 2.1, 3.3, 4.5])    # decelerating performance, slope < 0
        >>> tempo_slope(my_beats) < 0
        True

    Raises:
        ValueError: when linear regression cannot be calculated

    References:
        [1]: Cheston H., Cross, I., Harrison, P. (2024). Trade-offs in Coordination Strategies for
        Duet Jazz Performances Subject to Network Delay and Jitter. Music Perception, 42/1 (pp. 48–72).

    """

    # Dependent variable: BPM measurements
    y = 60 / np.diff(beats)
    # Predictor variable: the onset time
    x = beats[1:]
    # Handling NaNs: remove missing values from both arrays
    nan_idxs = np.isnan(y)
    x, y = x[~nan_idxs], y[~nan_idxs]
    # Compute linear regression and return slope
    return linregress(x, y).slope


def tempo_drift(beats: np.array) -> float:
    """Return the tempo drift for an array of beats.

    The tempo drift is the gradient of the slope of signed overall tempo change in a performance,
    such that larger values imply a greater departure from a linear tempo slope (i.e., not just
    accelerating or decelerating).

    Arguments:
        beats (np.array): an array of beat timestamps in seconds corresponding to e.g., quarter notes

    Returns:
        float: the tempo slope value

    Examples:
        >>> stable = np.array([1., 2., 3., 4.])
        >>> unstable = np.array([1., 2.1, 2.9, 3.5])
        >>> tempo_drift(stable) < tempo_drift(unstable)

    Raises:
        ValueError: when linear regression cannot be calculated

    """

    # Dependent variable: BPM measurements
    y = 60 / np.diff(beats)
    # Predictor variable: the onset time
    x = beats[1:]
    # Handling NaNs: remove missing values from both arrays
    nan_idxs = np.isnan(y)
    x, y = x[~nan_idxs], y[~nan_idxs]
    # Compute linear regression and return slope
    return linregress(x, y).stderr


def tempo_fluctuation(beats: np.array) -> float:
    """Calculates the percentage fluctuation about the overall tempo of provided `beats`.

    Tempo fluctuation can be calculated as the standard deviation of the tempo of a
    performance normalized by the mean tempo [1].

    Arguments:
        beats (np.array): an array of beat timestamps in seconds corresponding to e.g., quarter note positions

    Returns:
        float: the tempo fluctuation value

    Examples:
        >>> my_beats = np.array([1., 2., 3., 4.])    # stable performance

    References:
        [1] Cheston, H., Schlichting, J.L., Cross, I. and Harrison, P.M.C. (2024) Jazz Trio Database:
        Automated Annotation of Jazz Piano Trio Recordings Processed Using Audio Source Separation.
        Transactions of the International Society for Music Information Retrieval, 7/1 (pp. 144–158).

    """

    inter_beat_intervals = np.diff(beats)
    return np.nanstd(inter_beat_intervals) / np.nanmean(inter_beat_intervals)

