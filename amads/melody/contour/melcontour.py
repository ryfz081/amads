"""
Begin rambling:
We cannot add options to toggle information on and off when the information
does not exist in score. 

melcontour(score, res (int describing step size)) -> contour 
where contour is list((sampling_tick (in beats), pitch))

Reference(s):
    Eerola, T., Himberg, T., Toiviainen, P., & Louhivuori, J. (2006). 
        Perceived complexity of Western and African folk melodies by Western 
        and African listeners. Psychology of Music, 34(3), 341-375.

Reference(s) notes:
Keep in mind this reference does not reference a particular algorithm,
but rather a series of experiments that were done.
This implementation was ported from the matlab precursor to this library,
which was written by the same author as in the paper.

Back to my rambling:
Output problems... how do we get time in seconds...
Since we can't obtain said information from score, it is prudent for us to
introduce output postprocessing that introduce other data structures of information
that are readily made to handle these features (that score cannot).

Timemap doesn't actually do anything, except transcribes rhythmic information
of a score onto a bpm setting.
To fulfill getting the autocorrelation function of the original melcontour, 
we add the following postprocessing function:
correlatecontour(melcontour output) -> autocorrelation list.

If we want more useful functions for postprocessing, we can also implement this:
contourbeatstotime(contour, bpm) -> contour with time onsets instead
However, the original matlab implementation did not document this, so it might 
just be a vestige of an earlier version.

While we're at it, how about preprocessing? Probably not (for now).
"""

import numpy as np
import math

from ...core.basics import Note, Part, Score
from ...pitch.ismonophonic import ismonophonic

def melcontour(score: Score, res: float) -> list[tuple[float, float]]:
    """
    Given a score, and a sampling resolution (step size in beats), 
    returns a list of (sampling_tick, pitch) pairs of notes sorted by sampling_tick.

    Note that the last note before a sampling tick is chosen to be sampled.
    Also note that, unlike the matlab implementation where the onsets have the onset
    of the first note removed, the onsets here correspond to the note onsets 
    in the original score.
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

    # Note that the "tick" only starts on the onset of the first note
    sampling_tick = 0.0
    # while floats are not a particularly good idea in terms of accuracy if
    # we use very large numbers,
    # let's be honest, a singular score won't last for thousands of years
    # in practice
    sampling_end = float(notes[-1].end - notes[0].start)

    sampled_note_idx = 0
    # we can add a canary value to notes, for easy pickings
    notes.append(Note(1, None, None, None, float("infinity")))

    contour_list = []
    while (sampling_tick <= sampling_end):
        # we want to find the closest note with an onset closest to the current
        # sampling tick
        current_note_range = range(sampled_note_idx, len(notes))
        next_note_range = range(sampled_note_idx + 1, len(notes))
        for note_idx, next_note_idx in zip(current_note_range, next_note_range):
            # note that the canary value will always satisfy the second condition,
            # also note that the canary value will never be selected so we don't
            # have to worry about wrong values here
            if notes[note_idx].start <= sampling_tick + notes[0].start and \
                notes[next_note_idx].start > sampling_tick + notes[0].start:
                sampled_note_idx = note_idx
                break
            assert(notes[note_idx].start <= sampling_tick + notes[0].start)

        # assert that we didn't accidentally select the canary
        assert(notes[sampled_note_idx].start != float("infinity"))
        contour_list.append((sampling_tick, notes[sampled_note_idx].keynum))
        sampling_tick += res

    return contour_list

def autocorrelatecontour(contour_output: list[tuple[float, int]]):
    """
    Given vanilla contour output from melcontour,
    return a list of 2-tuples with the following elements:
    (1) the lag for the given contour output (a multiple of the original resolution
    called by melcontour)
    (2) the autocorrelation value corresponding to said contour output and lag

    Note that this is a postprocessing function for the output of melcontour, 
    hence why it is organized in the same file as melcontour.
    """
    # unpack output of melcontour
    sample_tuple, pitch_tuple = zip(*contour_output)
    # adjust pitch information based off of matlab implementation
    pitch_array = np.array(pitch_tuple, dtype = float)
    pitch_array -= np.mean(pitch_array)
    pitch_array /= math.sqrt(np.dot(pitch_array, pitch_array))
    # note that with the "full" option, np.correlate gives the exact same output
    # as matlab's xcorr function
    correlation_array = np.correlate(pitch_array, pitch_array, "full")
    # essentially, we want all values 
    # [-sampling_end, -sampling_end + res ... 0 ... sampling_end]
    # with a step size of res here
    # Note the slight abuse of notation, where sampling_end is truncated towards
    # the nearest multiple of res
    sample_array = np.array(sample_tuple, dtype = float)
    lag_array = np.concatenate((-sample_array[:0:-1], sample_array))
    assert(lag_array.shape == correlation_array.shape)
    # repack correspondence
    return list(zip(list(lag_array), list(correlation_array)))