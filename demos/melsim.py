"""
Example usage of the melsim module.
"""

from amads.core.basics import Score
from amads.melody.similarity import melsim
from amads.melody.similarity.melsim import check_packages_installed
from amads.utils import check_python_package


def test_check_dependencies():
    check_python_package("rpy2")
    check_packages_installed()


# Create a C major scale melody as (pitch, duration) tuples
# Using MIDI note numbers: C4=60, D4=62, E4=64, F4=65, G4=67, A4=69, B4=71, C5=72

c_major_scale = Score.from_melody(
    pitches=[60, 62, 64, 65, 67, 69, 71, 72], durations=1.0  # C4 to C5  # quarter notes
)

# Create a second melody with an altered fourth note (F4->F#4)
modified_scale = Score.from_melody(
    pitches=[60, 62, 64, 66, 67, 71, 72], durations=1.0  # C4 to C5  # quarter notes
)

# Create a third melody with two altered notes (F4->F#4 and A4->Ab4)
third_scale = Score.from_melody(
    pitches=[60, 62, 64, 66, 67, 68, 71, 72], durations=1.0  # C4 to C5  # quarter notes
)

# Create a fourth melody with three altered notes (F4->F#4, A4->Ab4, and B4->Bb4)
fourth_scale = Score.from_melody(
    pitches=[60, 62, 64, 66, 67, 68, 70, 72], durations=1.0  # C4 to C5  # quarter notes
)

melodies = [c_major_scale, modified_scale, third_scale, fourth_scale]

# Get similarity between the two melodies using the 'cosine' and 'Simpson' similarity measures
# Remember, the names of the similarity measures are case-sensitive, and can be found in the melsim
# documentation string

cosine_similarity = melsim.get_similarity(melodies, "cosine")
simpson_similarity = melsim.get_similarity(melodies, "Simpson")

print(f"Cosine similarity between melodies: {cosine_similarity}\n")
print(f"Simpson similarity between melodies: {simpson_similarity}\n")
