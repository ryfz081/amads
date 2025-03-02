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
    
    def _convert_to_symusic(self, input_data: Union[Score, str, 'symusic.Score']) -> 'symusic.Score':
        """Convert various input types to symusic Score format."""
        check_python_package_installed('symusic')
        import symusic
        
        if isinstance(input_data, Score):
            return score_to_symusic(input_data)
        elif isinstance(input_data, str):
            return symusic.Score(input_data)
        elif isinstance(input_data, symusic.Score):
            return input_data
        else:
            raise TypeError(f"Expected Score, path to MIDI file, or symusic.Score, got {type(input_data)}")

    def _encode_tokens(self, symusic_score: 'symusic.Score') -> List[Any]:
        """Get raw tokens from the tokenizer."""
        return self._tokenizer.encode(symusic_score)[0]

    def _add_timing(self, raw_tokens: List[Any]) -> List[Token]:
        """Add REMI-specific timing to tokens."""
        tokens = []
        current_time = None  # Start as None until we see first Position
        current_tempo = 120.0  # Default tempo in BPM
        vocab = {v: k for k, v in self._tokenizer.vocab.items()}
        
        for tok in raw_tokens:
            token_str = vocab.get(tok, str(tok))
            
            try:
                if token_str.startswith('Position_'):
                    position = int(token_str.split('_')[1])
                    current_time = position * (60.0 / current_tempo)
                elif token_str.startswith('Tempo_'):
                    current_tempo = float(token_str.split('_')[1])
            except Exception as e:
                pass
            
            # Only assign timing to Position and Pitch tokens
            if current_time is not None and (
                token_str.startswith('Position_') or 
                token_str.startswith('Pitch_')
            ):
                tokens.append(Token(value=tok, start_time=current_time))
            else:
                tokens.append(Token(value=tok))
                
        return tokens

    def tokenize(self, input_data: Union[Score, str, 'symusic.Score']) -> List[Token]:
        """Convert input into a sequence of Tokens."""
        symusic_score = self._convert_to_symusic(input_data)
        raw_tokens = self._encode_tokens(symusic_score)
        return self._add_timing(raw_tokens)

    def decode(self, tokens: List[Token]) -> Score:
        """Convert tokens back to a Score."""
        miditok_tokens = [t.value for t in tokens]
        midi_data = self._tokenizer.tokens_to_midi(miditok_tokens)
        return midi_to_score(midi_data)

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
    
    def _add_timing(self, raw_tokens: List[Any]) -> List[Token]:
        """Add REMI-specific timing to tokens."""
        tokens = []
        current_time = None  # Start as None until we see first Position
        current_tempo = 120.0  # Default tempo in BPM
        vocab = {v: k for k, v in self._tokenizer.vocab.items()}
        
        for tok in raw_tokens:
            token_str = vocab.get(tok, str(tok))
            
            try:
                if token_str.startswith('Position_'):
                    position = int(token_str.split('_')[1])
                    current_time = position * (60.0 / current_tempo)
                elif token_str.startswith('Tempo_'):
                    current_tempo = float(token_str.split('_')[1])
            except Exception as e:
                pass
            
            # Only assign timing to Position and Pitch tokens
            if current_time is not None and (
                token_str.startswith('Position_') or 
                token_str.startswith('Pitch_')
            ):
                tokens.append(Token(value=tok, start_time=current_time))
            else:
                tokens.append(Token(value=tok))
                
        return tokens

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
    
    def _add_timing(self, raw_tokens: List[Any]) -> List[Token]:
        """Add TSD-specific timing to tokens."""
        raise NotImplementedError("TSD timing logic not yet implemented")

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
    
    def _add_timing(self, raw_tokens: List[Any]) -> List[Token]:
        """Add MIDI-Like specific timing to tokens."""
        raise NotImplementedError("MIDI-Like timing logic not yet implemented")