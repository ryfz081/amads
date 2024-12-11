"""
Calculates the Step Contour of a melody, along with related features, as implemented
in the FANTASTIC toolbox of Müllensiefen (2009).
"""

__author__ = "David Whyatt"

import numpy as np


step_contour_length = 64


def _normalize_durations(durations: list[float]) -> list[float]:
    """Helper function to normalize note durations to fit within 4 bars of 4/4 time
    (64 tatums total).

    Parameters
    ----------
    durations : list[float]
        List of duration values measured in tatums

    Returns
    -------
    list[float]
        List of normalized duration values
    """
    total_duration = sum(durations)
    if total_duration == 0:
        return durations

    # Apply normalization formula: normalized_duration = 64 * (duration/sum(durations))
    normalized = [step_contour_length * (duration / total_duration) for duration in durations]
    return normalized


def _expand_to_vector(pitches: list[int], normalized_durations: list[float]) -> list[int]:
    """Helper function to create a vector of length 64 by repeating each pitch value
    proportionally to its normalized duration.

    Parameters
    ----------
    pitches : list[int]
        List of pitch values
    normalized_durations : list[float]
        List of normalized duration values (should sum to 64)

    Returns
    -------
    list[int]
        List of length 64 containing repeated pitch values
    """
    result = []
    for pitch, duration in zip(pitches, normalized_durations):
        # Round the duration to nearest integer and repeat the pitch that many times
        repetitions = round(duration)
        result.extend([pitch] * repetitions)

    # Ensure the result is exactly 64 elements long
    if len(result) > step_contour_length:
        result = result[:step_contour_length]
    elif len(result) < step_contour_length:
        # Pad with the last pitch if necessary
        result.extend([result[-1]] * (step_contour_length - len(result)))

    return result

def step_contour(pitches: list[int], durations: list[float]) -> list[int]:
    """Calculates the Step Contour of a melody, as according to Müllensiefen (2009) [1].

    Parameters
    ----------
    pitches : list[int]
        List of pitch values
    durations : list[float]
        List of duration values measured in tatums

    Returns
    -------
    list[int]
        List of length 64 containing the step contour of the melody

    Raises
    ------
    ValueError
        If the length of pitches is not equal to the length of durations

    Examples
    --------
    Calculate the step contour for an input list of pitches and durations (in tatums)
    >>> step_contour([60, 62, 63, 65, 72], [1, 1, 4, 4, 8])
    [60, 60, 60, 60, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
     63, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 72, 72, 72, 72, 72, 72,
     72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72,
     72]

    Throws a ValueError where input lists are different in length
    >>> step_contour([60, 62, 63, 65, 72], [1, 1, 4, 4])
    Traceback (most recent call last):
      ...
    ValueError: The length of pitches (currently 5) must be equal to the length of durations (currently 4)

    References
    ----------
    [1] Müllensiefen, D. (2009). FANTASTIC: Feature ANalysis Technology Accessing
    STatistics (In a Corpus): Technical Report v1.5
    https://www.doc.gold.ac.uk/isms/m4s/FANTASTIC_docs.pdf
    """
    if len(pitches) != len(durations):
        raise ValueError(f"The length of pitches (currently {len(pitches)}) must be equal to the length of durations (currently {len(durations)})")

    normalized_durations = _normalize_durations(durations)
    contour = _expand_to_vector(pitches, normalized_durations)
    return contour

def step_contour_global_variation(contour: list[int]) -> float:
    """Calculates the global variation of a step contour.

    Parameters
    ----------
    contour : list[int]
        List of length 64 containing the step contour of a melody

    Returns
    -------
    float
        Float value representing the global variation of the step contour

    Raises
    ------
    ValueError
        If the length of the step contour is not 64

    Examples
    --------
    Calculate the global variation of a step contour
    >>> contour = [60, 60, 60, 60, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    ...           63, 63, 63, 63, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
    ...           72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72,
    ...           72, 72, 72, 72, 72, 72, 72, 72, 72, 72]
    >>> step_contour_global_variation(contour)
    4.46339...

    Create step contour by adding lists
    >>> descending = [72] * 16 + [69] * 16 + [65] * 16 + [60] * 16
    >>> step_contour_global_variation(descending)
    4.5

    Calculate global variation for a constant melody
    >>> constant = [60] * 64
    >>> step_contour_global_variation(constant)
    0.0

    Throws a ValueError when contour length is not 64
    >>> step_contour_global_variation([60, 62, 63]) # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: The length of the step contour must be 64.
    """
    if len(contour) != step_contour_length:
        raise ValueError("The length of the step contour must be 64.")

    return float(np.nanstd(contour))


def step_contour_global_direction(contour: list[int]) -> float:
    """Calculates the global direction of a step contour by correlating the step
    contour with a vector of [1, 2, 3, ..., 64].

    Parameters
    ----------
    contour : list[int]
        List of length 64 containing the step contour of a melody

    Returns
    -------
    float
        Float value representing the global direction of the step contour
        Returns 0.0 if the contour is flat

    Raises
    ------
    ValueError
        If the length of the step contour is not 64

    Examples
    --------
    Calculate the global direction of a step contour - this example is clearly upwards
    >>> contour = [60, 60, 60, 60, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    ...           63, 63, 63, 63, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
    ...           72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72,
    ...           72, 72, 72, 72, 72, 72, 72, 72, 72, 72]
    >>> step_contour_global_direction(contour)
    0.9316...

    Calculate the global direction of a step contour - this example is clearly downwards
    >>> descending = [72] * 16 + [69] * 16 + [65] * 16 + [60] * 16
    >>> step_contour_global_direction(descending)
    -0.9623...

    Flat contour
    >>> flat = [60] * 64
    >>> step_contour_global_direction(flat)
    0.0
    """
    if len(contour) != step_contour_length:
        raise ValueError("The length of the step contour must be 64.")
    
    # First check whether all values in the contour are identical (e.g. [60,60,60,...])
    # In this case, correlation is undefined (0/0) so we return 0.0 to indicate no direction
    if len(set(contour)) == 1:
        return 0.0
        
    # Otherwise calculate correlation as normal
    try:
        corr = np.corrcoef(contour, np.arange(step_contour_length))[0, 1]
        return float(corr)
    
    except:
        # Handle any other numerical errors in correlation calculation
        # This should never happen with valid input, so return None to indicate error
        return None


def step_contour_local_variation(contour: list[int]) -> float:
    """Calculates the local variation of a step contour.

    Parameters
    ----------
    contour : list[int]
        List of length 64 containing the step contour of a melody

    Returns
    -------
    float
        Float value representing the local variation of the step contour

    Raises
    ------
    ValueError
        If the length of the step contour is not 64

    Examples
    --------
    Calculate the local variation of a step contour
    >>> contour = [60, 60, 60, 60, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    ...           63, 63, 63, 63, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
    ...           72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72,
    ...           72, 72, 72, 72, 72, 72, 72, 72, 72, 72]
    >>> step_contour_local_variation(contour)
    0.1904...
    """
    if len(contour) != step_contour_length:
        raise ValueError("The length of the step contour must be 64.")

    # Calculate the mean absolute difference between adjacent values
    pairs = list(zip(contour, contour[1:]))
    local_variation = sum(abs(c2 - c1) for c1, c2 in pairs) / len(pairs)

    return local_variation
