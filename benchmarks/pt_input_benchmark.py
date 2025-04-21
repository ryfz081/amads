# pt_input_benchmark.py -- experiments around partitura midi file input

__author__ = "Roger B. Dannenberg"

# As a baseline, let's see how long it takes to create a score with 1000 notes:

from amads.core.basics import Measure, Note, Part, Score, Staff


def create_score(n: int) -> Score:
    """create a score with n Notes - use quarter notes in 4/4 time
    """
    staff = Staff()
    score = Score(Part(staff))
    for i in range(n):
        if i % 4 == 0:  # time to create a new measure?
            measure = Measure(parent=staff, onset=n, duration=4)
        pitch = i % 24 + 48  # 2-octave chromatic scales
        Note(parent=measure, onset=n, duration=1, pitch=pitch)
    return score


create_score(1000).show()
