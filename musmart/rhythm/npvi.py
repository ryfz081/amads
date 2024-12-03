#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the `normalised_pairwise_variability_index` function."""

from typing import Union

import numpy as np


__all__ = ["normalised_pairwise_variability_index"]


def _npvi(data: np.array) -> float:
    """Calculates nPVI for a single input"""
    return (
        sum([abs((k - k1) / ((k + k1) / 2)) for (k, k1) in zip(data, data[1:])]) * 100 / (sum(1 for _ in data) - 1)
    )


def normalised_pairwise_variability_index(sequence: np.array) -> Union[float, np.array]:
    """Extracts the normalised pairwise variability index (nPVI).

    The nPVI is a measure of variability between successive elements in a sequence, commonly used
    in music analysis to quantify rhythmic variability [1].

    If the input is a 1D array, returns a single nPVI value. If the input is a 2D array, returns
    nPVI values for every element in the array along dimension 1.

    Arguments:
        sequence (np.array): the data to calculate the nPVI for. Either 1D or 2D arrays are supported.

    Returns:
        float | np.array: the extracted nPVI value or values.

    Examples:
        >>> normalised_pairwise_variability_index(np.array([1, 2, 3]))
        53.333333333333336
        >>> normalised_pairwise_variability_index(np.array([[1, 2, 3], [3, 3, 6], [4, 6, 8]]))
        np.array([53.33333333 33.33333333 34.28571429])

    Raises:
        ValueError: in cases where the input is not a 1D or 2D array.

    References:
        [1]: Daniele, J. R., & Patel, A. D. (2013). An Empirical Study of Historical Patterns in Musical Rhythm. Music
            Perception, 31/1 (pp. 10-18).
    """

    # For a 1D array, return a single floating point
    if len(sequence.shape) == 1:
        return _npvi(sequence)
    # Otherwise, return a floating point for each element
    elif len(sequence.shape) == 2:
        return np.apply_along_axis(_npvi, axis=1, arr=sequence)
    else:
        raise ValueError("`sequence` must be either a 1D or 2D array")
