"""
Begin rambling:
There is some trouble for deciding the output for combcontour
Namely, we have a similar problem to when we implemented boundary
and segment_gestalt. 
We don't have a total ordering in our score representation, so
a matrix representation is probably not going to work (?) unless
I finesse a key into the return value.

combcontour(score) -> notes, contour_matrix

Reference:
    Marvin, E. W. & Laprade, P. A. (1987). Relating music contours: 
        Extensions of a theory for contour. Journal of Music Theory,
        31(2), 225-267.

Continue Rambling:
Observe that we are currently dealing with monophonic melodies,
so we can use offsets to work this.

Basically always half of the matrix square will be 1.
We are forced to store a dense map for this... *sighs*
That or, it might be prudent just to return a double iterator.
Probably not particularly prudent to do that.
Ah well, might as well store the matrix in a numpy array

However, whether or not we can hide this behemoth amount of
unnecessary data still depends on what this matrix is used for...

I would much much much prefer that not all of such regular data can be used,
so we can probably hide the computation somewhere...
Probably not, because the entire point of combcontour is to give a contour
representation via a matrix form.

Okay, we *have* to use a matrix or something similar, because contour here
is defined as pitch comparisons between 2 notes that are i notes apart
when sorted by onset.
This is such a cute realization
"""
from ...core.basics import Note, Part, Score
from ...pitch.ismonophonic import ismonophonic
import numpy as np


def combcontour(score: Score):
    """
    Given a score, we return two things:
    (1) 2-d numpy array with len(notes) x len(notes) dimension, recording
    a boolean on whether the ith note's pitch is larger than the jth note's pitch
    (2) list of notes (as a reference for which index refers to which note 
    in the 2-d array)
    """
    # ripped from boundary, because once again, we're doing melodic analysis
    if not ismonophonic(score):
        raise ValueError("Score must be monophonic")
    # make a flattened and collapsed copy of the original score
    flattened_score = score.flatten(collapse=True)

    # extracting note references here from our flattened score
    notes = list(flattened_score.find_all(Note))

    # sort the notes
    notes.sort(key=lambda note: (note.start, -note.pitch.keynum))
    if not notes:
        # probably better to do it earlier
        raise ValueError("Score is empty")

    pitch_array = np.array([note.keynum for note in notes])
    
    contour_mtx = np.full((len(notes), len(notes)), False, dtype = bool)

    for i in range(len(notes)):
        contour_mtx[i:, i] = pitch_array[i] > pitch_array[i:]

    return notes, contour_mtx