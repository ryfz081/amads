import numpy as np
from ..core.basics import Note, Score
from ..algorithms.nnotes import nnotes

def concur(score: Score):
    """

    
    Args:
        score (Score): The musical score to analyze
        float (Threshold): (optional) value for threshold for concurrent
                            onsets default value is +/- 0.2 beats

    Returns:
        int (c): integer displaying the proportion of concurrent onsets
        
        
    Example:
        concur(score, 0.25)
    """
    
    a = nnotes(Score)
    #if score is empty, return
    if a == 0: return; end
    
    #if threshold is not provided, set it to 0.2
    if threshold is None: threshold = 0.2;

    #get unique onset times
    
    onsets = [n._convert_to_quarters.onset for n in Score.find_all(Note)]
    u = np.unique(onsets);

    #calculate the proportion of concurrent onsets
    c = 1 - len(u)/len(onsets); 
    return c;
    
