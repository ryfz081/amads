"""
Tokenization schemes for MIDI data in the AMADS library using miditok implementations.
"""

from typing import List, Optional, Dict, Any, Union
from amads.core.basics import Score, Note, Part
from amads.expectation.tokenizer import Token, Tokenizer
from amads.utils import check_python_package_installed


class MIDITokenizer(Tokenizer):
    """Base class for all MIDI tokenization schemes."""
    
    def tokenize(self, score: Score) -> List[Token]:
        """Convert a Score object into a sequence of Tokens.

        Parameters
        ----------
        score : Score
            The musical score to tokenize

        Returns
        -------
        List[Token]
            Sequence of tokens representing the score
        """
        raise NotImplementedError
    
    def decode(self, tokens: List[Token]) -> Optional[Score]:
        """Convert tokens back to a Score. Optional method.

        Parameters
        ----------
        tokens : List[Token]
            Sequence of tokens to decode

        Returns
        -------
        Optional[Score]
            The reconstructed musical score, if decoding is supported
        """
        return None


# Common default parameters for all miditok tokenizers
DEFAULT_MIDITOK_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,
    "tempo_range": (40, 250)
}

def score_to_symusic(score: Score) -> 'symusic.Score':
    """Convert our Score format to symusic Score format directly.
    
    Parameters
    ----------
    score : Score
        Our internal score representation
        
    Returns
    -------
    symusic.Score
        Score representation that miditok can use
    """
    check_python_package_installed('symusic')
    import symusic
    
    # First flatten the score to simplify note handling
    flat_score = score.flatten(collapse=True)
    
    # Constants for conversion (assuming 120 BPM)
    TICKS_PER_QUARTER = 480
    SECONDS_PER_QUARTER = 0.5  # at 120 BPM
    TICKS_PER_SECOND = int(TICKS_PER_QUARTER / SECONDS_PER_QUARTER)
    
    # Create notes list
    notes = []
    for note in flat_score.find_all(Note):
        symusic_note = symusic.Note(
            time=int(note.onset * TICKS_PER_SECOND),      # Convert seconds to ticks
            duration=int(note.duration * TICKS_PER_SECOND), # Convert seconds to ticks
            pitch=int(note.pitch.keynum),
            velocity=64  # Default velocity
        )
        notes.append(symusic_note)
    
    # Create a track with the notes
    track = symusic.Track(notes=notes)
    
    # Create a symusic Score
    symusic_score = symusic.Score()
    symusic_score.tracks = [track]
    
    return symusic_score

class SymusicTokenizer(MIDITokenizer):
    """Base class for tokenizers that use symusic/miditok implementations."""
    
    def tokenize(self, input_data: Union[Score, str, 'symusic.Score']) -> List[Token]:
        """Convert input into a sequence of Tokens.

        Parameters
        ----------
        input_data : Union[Score, str, symusic.Score]
            Either:
            - A Score object
            - Path to a MIDI file
            - A symusic Score object

        Returns
        -------
        List[Token]
            Sequence of tokens representing the music
        """
        check_python_package_installed('symusic')
        import symusic
        
        # Convert input to symusic Score format
        if isinstance(input_data, Score):
            symusic_score = score_to_symusic(input_data)
        elif isinstance(input_data, str):
            symusic_score = symusic.Score(input_data)  # Load MIDI file
        elif isinstance(input_data, symusic.Score):
            symusic_score = input_data
        else:
            raise TypeError(f"Expected Score, path to MIDI file, or symusic.Score, got {type(input_data)}")
            
        # Use the concrete tokenizer's implementation
        miditok_tokens = self._tokenizer.encode(symusic_score)[0]  # [0] to get first sequence
        return [Token(value=tok) for tok in miditok_tokens]

class REMITokenizer(SymusicTokenizer):
    """REMI tokenizer using miditok's implementation."""
    
    def __init__(self, **params):
        check_python_package_installed('miditok')
        from miditok import REMI, TokenizerConfig
        
        default_params = DEFAULT_MIDITOK_PARAMS.copy()
        default_params.update({
            "use_chords": True,
            "use_rests": False
        })
        default_params.update(params)
        
        config = TokenizerConfig(**default_params)
        self._tokenizer = REMI(config)
    
    def decode(self, tokens: List[Token]) -> Score:
        miditok_tokens = [t.value for t in tokens]
        midi_data = self._tokenizer.tokens_to_midi(miditok_tokens)
        return midi_to_score(midi_data)

class TSDTokenizer(SymusicTokenizer):
    """Time-Shift-Duration (TSD) tokenizer using miditok's implementation."""
    
    def __init__(self, **params):
        check_python_package_installed('miditok')
        from miditok import TSD, TokenizerConfig
        
        default_params = DEFAULT_MIDITOK_PARAMS.copy()
        default_params.update({
            "use_chords": False,
            "use_rests": True
        })
        default_params.update(params)
        
        config = TokenizerConfig(**default_params)
        self._tokenizer = TSD(config)
    
    def decode(self, tokens: List[Token]) -> Score:
        miditok_tokens = [t.value for t in tokens]
        midi_data = self._tokenizer.tokens_to_midi(miditok_tokens)
        return midi_to_score(midi_data)

class MIDILikeTokenizer(SymusicTokenizer):
    """MIDI-Like tokenizer using miditok's implementation."""
    
    def __init__(self, **params):
        check_python_package_installed('miditok')
        from miditok import MIDILike, TokenizerConfig
        
        default_params = DEFAULT_MIDITOK_PARAMS.copy()
        default_params.update({
            "use_chords": False,
            "use_rests": False
        })
        default_params.update(params)
        
        config = TokenizerConfig(**default_params)
        self._tokenizer = MIDILike(config)
    
    def decode(self, tokens: List[Token]) -> Score:
        miditok_tokens = [t.value for t in tokens]
        midi_data = self._tokenizer.tokens_to_midi(miditok_tokens)
        return midi_to_score(midi_data)
