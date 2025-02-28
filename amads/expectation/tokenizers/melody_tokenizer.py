from typing import List
from amads.core.basics import Score, Note
from amads.expectation.tokenizer import Tokenizer, Token
from amads.pitch.ismonophonic import ismonophonic

class MelodyIntervalTokenizer(Tokenizer):
    """Tokenizes a melody into pitch intervals between successive notes."""
    def tokenize(self, x) -> List[Token]:
        assert isinstance(x, Score)
        assert ismonophonic(x)

        flat_score = x.flatten(collapse=True)
        notes = list(flat_score.find_all(Note))
        
        if len(notes) < 2:
            return []
            
        tokens = []
        for prev_note, current_note in zip(notes[:-1], notes[1:]):
            # Calculate pitch interval in semitones
            interval = current_note.keynum - prev_note.keynum
            tokens.append(Token(interval, prev_note.onset, current_note.onset))

        return tokens

class AudioTokenizer(Tokenizer):
    pass

class IOITokenizer(Tokenizer):
    """Tokenizes a melody into Inter-Onset Intervals (time between successive note starts)."""
    def tokenize(self, x) -> List[Token]:
        assert isinstance(x, Score)
        assert ismonophonic(x)

        flat_score = x.flatten(collapse=True)
        notes = list(flat_score.find_all(Note))
        
        if len(notes) < 2:
            return []
            
        tokens = []
        for prev_note, current_note in zip(notes[:-1], notes[1:]):
            # Calculate time between note onsets
            ioi = current_note.onset - prev_note.onset
            assert ioi >= 0, f"Invalid negative IOI found: {ioi}"
            tokens.append(Token(ioi, prev_note.onset, current_note.onset))

        return tokens
