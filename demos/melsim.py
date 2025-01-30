"""
Example usage of the melsim module.
"""

from amads.core.basics import Note, Part, Score
from amads.melody.similarity import melsim

# Uses @pmcharrison's create_score function in PR #58
# TODO: Update after PR #58 is merged to call from basics.py instead


def create_score(melody):
    """Create a Score object from a melody sequence.

    Args:
        melody: List of (pitch, duration) tuples representing a melody

    Returns:
        A Score object containing a single Part of the melody
    """
    score = Score()
    part = Part()

    # Keep track of current time position
    current_time = 0

    # Create notes from melody sequence
    for pitch, duration in melody:
        note = Note(duration=duration, pitch=pitch, delta=current_time)
        part.insert(note)
        current_time += duration

    # Set part duration to total duration
    part.duration = current_time

    # Add part to score
    score.insert(part)
    score.duration = current_time

    return score


# Create a C major scale melody as (pitch, duration) tuples
# Using MIDI note numbers: C4=60, D4=62, E4=64, F4=65, G4=67, A4=69, B4=71, C5=72
c_major_scale = [
    (60, 1.0),  # C4
    (62, 1.0),  # D4
    (64, 1.0),  # E4
    (65, 1.0),  # F4
    (67, 1.0),  # G4
    (69, 1.0),  # A4
    (71, 1.0),  # B4
    (72, 1.0),  # C5
]

# Create score object containing the C major scale
c_major_score = create_score(c_major_scale)
# Create a second melody with an altered fourth note (F4->F#4)
altered_scale = [
    (60, 1.0),  # C4
    (62, 1.0),  # D4
    (64, 1.0),  # E4
    (66, 1.0),  # F#4 (altered from F4)
    (67, 1.0),  # G4
    (69, 1.0),  # A4
    (71, 1.0),  # B4
    (72, 1.0),  # C5
]

# Create score object containing the altered scale
second_score = create_score(altered_scale)

# Create a third melody with two altered notes (F4->F#4 and A4->Ab4)
third_scale = [
    (60, 1.0),  # C4
    (62, 1.0),  # D4
    (64, 1.0),  # E4
    (66, 1.0),  # F#4 (altered from F4)
    (67, 1.0),  # G4
    (68, 1.0),  # Ab4 (altered from A4)
    (71, 1.0),  # B4
    (72, 1.0),  # C5
]

# Create score object containing the third scale
third_score = create_score(third_scale)
# Create a fourth melody with three altered notes (F4->F#4, A4->Ab4, and B4->Bb4)
fourth_scale = [
    (60, 1.0),  # C4
    (62, 1.0),  # D4
    (64, 1.0),  # E4
    (66, 1.0),  # F#4 (altered from F4)
    (67, 1.0),  # G4
    (68, 1.0),  # Ab4 (altered from A4)
    (70, 1.0),  # Bb4 (altered from B4)
    (72, 1.0),  # C5
]

# Create score object containing the fourth scale
fourth_score = create_score(fourth_scale)


melodies = [c_major_score, second_score, third_score, fourth_score]

# Get similarity between the two melodies using the 'cosine' and 'Simpson' similarity measures
# Remember, the names of the similarity measures are case-sensitive, and can be found in the melsim
# documentation string

cosine_similarity = melsim.get_similarity(melodies, "cosine")
simpson_similarity = melsim.get_similarity(melodies, "Simpson")

print(f"Cosine similarity between melodies: {cosine_similarity}")
print(f"Simpson similarity between melodies: {simpson_similarity}")
