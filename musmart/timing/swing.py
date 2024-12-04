#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements various functions useful for analyzing swing in jazz, e.g. `beat_upbeat_ratio`."""

from math import log2

import numpy as np


__all__ = ["beat_upbeat_ratio"]


def beat_upbeat_ratio(
        beats: np.array,
        upbeats: np.array,
        use_log2_burs: bool = False,
        allow_multiple_burs_per_beat: bool = False
) -> np.array:
    """Extracts beat-upbeat ratio (BUR) values from an array of onsets.

    The beat-upbeat ratio is introduced in [1] as a concept for analyzing the individual amount of 'swing' in two
    consecutive eighth note beat durations. It is calculated simply by dividing the duration of the first, 'long'
    eighth note beat by the second, 'short' beat. A BUR value of 2 indicates 'perfect' swing, i.e. a triplet quarter
    note followed by a triplet eighth note, while a BUR of 1 indicates 'even' eighth note durations.

    By default, it is expected that there will be exactly one upbeat between every two consecutive beats. If this is
    not the case, this function will only return a BUR for the first value following each beat. To bypass this, pass
    `allow_multiple_burs_per_beat=True`, which will calculate a BUR for every upbeat between two beats.

    Arguments:
        beats (np.array): an array of quarter note beat timestamps
        upbeats (np.array): an array of upbeat timestamps (i.e., approximate eighth notes).
        use_log2_burs (bool, optional): whether to use the log^2 of inter-onset intervals to calculate BURs,
            as employed in [2]. Defaults to False.
        allow_multiple_burs_per_beat (bool, optional): if multiple upbeats are found between two consecutive beats and
            this value is True, will return a beat-upbeat ratio for every upbeat. Otherwise, will return a value only
            for the first upbeat following a single beat

    Returns:
        np.array: the calculated BUR values

    Examples:
        >>> my_upbeats = np.array([0.6, 1.4])
        >>> my_beats = np.array([0., 1., 2.])
        >>> beat_upbeat_ratio(my_beats, my_upbeats)
        np.array([0.66666667, 1.5])

        >>> my_upbeats = np.array([0.6, 1.4])
        >>> my_beats = np.array([0., 1., 2.])
        >>> beat_upbeat_ratio(my_beats, my_upbeats, use_log2_burs=True)
        np.array([-0.5849625, 0.5849625])

        >>> my_upbeats = np.array([0.6, 0.7, 1.4])
        >>> my_beats = np.array([0., 1., 2.])
        >>> beat_upbeat_ratio(my_beats, my_upbeats, allow_multiple_burs_per_beat=False)
        Traceback (most recent call last):
            ...
        ValueError: Expected only one upbeat between 0.0 and 1.0, but got [0.6 0.7]

        >>> my_upbeats = np.array([0.6, 0.7, 1.4])
        >>> my_beats = np.array([0., 1., 2.])
        >>> beat_upbeat_ratio(my_beats, my_upbeats, allow_multiple_burs_per_beat=True)
        np.array([0.66666667, 0.42857143, 1.5])

    References:
        [1]: Benadon, F. (2006). Slicing the Beat: Jazz Eighth-Notes as Expressive Microrhythm. Ethnomusicology,
            50/1 (pp. 73-98).
        [2]: Corcoran, C., & Frieler, K. (2021). Playing It Straight: Analyzing Jazz Soloists’ Swing Eighth-Note
            Distributions with the Weimar Jazz Database. Music Perception, 38(4), 372–385.

    """

    def _bur(beat1: float, beat2: float) -> np.array:
        """Internal BUR calculation function. Takes in two consecutive beats and gets BURs."""
        # Beat 2 should always be later than beat 1
        assert beat2 > beat1
        # Find all upbeats between these two beats
        upbeats_bw = upbeats[(beat1 <= upbeats) & (upbeats <= beat2)]
        # If we've found multiple upbeats, and we only want to get one BUR per beat
        if len(upbeats_bw) > 1 and not allow_multiple_burs_per_beat:
            # Just use the first upbeat we've found
            upbeats_bw = np.array([upbeats_bw[0]])    # keep as an array to allow iteration
        # Iterate over all upbeats found between the two beats
        found_burs = []
        for upbeat_bw in upbeats_bw:
            # Compute the inter-onset intervals
            ioi1 = upbeat_bw - beat1
            ioi2 = beat2 - upbeat_bw
            # Calculate the beat-upbeat ratio
            bur = ioi2 / ioi1
            # Express with base-2 log if required
            if use_log2_burs:
                bur = log2(bur)
            found_burs.append(bur)
        return np.array(found_burs)

    # Get consecutive pairs of (non-missing) beats
    consecutive_beats = np.array(
        [(b1, b2) for b1, b2 in zip(beats, beats[1:]) if not any((np.isnan(b1), np.isnan(b2)))]
    )
    # Calculate the BUR for upbeats between consecutive beats
    return np.concatenate([_bur(b1, b2) for b1, b2 in consecutive_beats])
