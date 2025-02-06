__author__ = "Yiwen Zhao"

import numpy as np
from ..core.basics import Score, Note, Part

def calculate_melodic_accent(score: Score) -> list:
    """
    Calculate melodic accent salience according to Thomassen's model.
    
    Melodic accent is determined by analyzing patterns of three consecutive notes.
    The model assigns different accent values based on the following patterns:
    - Peak accent (up-down pattern): 1.0
    - Valley accent (down-up pattern): 0.6
    - Continuous ascending: 0.4
    - Continuous descending: 0.4
    - Other patterns (including repeated notes): 0.0
    
    Parameters
    ----------
    score : Score
        A Score object containing the melody to analyze. The score will be
        flattened and collapsed into a single sequence of notes ordered by
        onset time.
    
    Returns
    -------
    list
        A list of accent values between 0 and 1 for each note. First and last
        notes always have accent value 0 as they require three notes for
        accent calculation.
    
    Examples
    --------
    >>> score = Score()
    >>> part = Part()
    >>> notes = [Note(pitch=60), Note(pitch=64), Note(pitch=62)]  # up-down pattern
    >>> for note in notes:
    ...     part.append(note)
    >>> score.append(part)
    >>> accent_values = calculate_melodic_accent(score)
    >>> print(accent_values)
    [0.0, 1.0, 0.0]  # Peak accent in the middle
    """
    # Flatten and collapse the score into a single sequence of notes
    flattened_score = score.flatten(collapse=True)
    
    # Handle empty score
    if len(flattened_score.content) == 0:
        return []
        
    # Get all notes from the flattened score
    notes = list(flattened_score.find_all(Note))
    
    # Need at least 3 notes to calculate accents
    if len(notes) < 3:
        return [0] * len(notes)
    
    # Initialize accent values array
    accents = [0] * len(notes)
    
    # Analyze each three-note window
    for i in range(1, len(notes) - 1):
        # Calculate intervals between adjacent notes
        prev_interval = notes[i].keynum - notes[i-1].keynum
        next_interval = notes[i+1].keynum - notes[i].keynum
        
        # Assign accent values based on Thomassen's model:
        if prev_interval > 0 and next_interval < 0:
            # Peak accent (up-down pattern)
            accents[i] = 1.0
        elif prev_interval < 0 and next_interval > 0:
            # Valley accent (down-up pattern)
            accents[i] = 0.6
        elif prev_interval > 0 and next_interval > 0:
            # Continuous ascending pattern
            accents[i] = 0.4
        elif prev_interval < 0 and next_interval < 0:
            # Continuous descending pattern
            accents[i] = 0.4
        else:
            # Other patterns (including repeated notes)
            accents[i] = 0.0
    
    return accents


if __name__ == "__main__":
    # Create a test score
    score = Score()
    part = Part()
    
    # Test case 1: Peak accent pattern
    print("\nTest Case 1: Peak accent pattern (up-down)")
    notes = [
        Note(pitch=60),  # C4
        Note(pitch=64),  # E4 (up)
        Note(pitch=62)   # D4 (down)
    ]
    
    for note in notes:
        part.append(note)
    score.append(part)
    
    accent_values = calculate_melodic_accent(score)
    print("Notes:", [note.name_with_octave for note in notes])
    print("Accent values:", accent_values)
    
    # Test case 2: Valley accent pattern
    print("\nTest Case 2: Valley accent pattern (down-up)")
    score2 = Score()
    part2 = Part()
    notes2 = [
        Note(pitch=64),  # E4
        Note(pitch=60),  # C4 (down)
        Note(pitch=65)   # F4 (up)
    ]
    
    for note in notes2:
        part2.append(note)
    score2.append(part2)
    
    accent_values2 = calculate_melodic_accent(score2)
    print("Notes:", [note.name_with_octave for note in notes2])
    print("Accent values:", accent_values2)
    
    # Test case 3: Empty score
    print("\nTest Case 3: Empty score")
    empty_score = Score()
    print("Empty score accents:", calculate_melodic_accent(empty_score))