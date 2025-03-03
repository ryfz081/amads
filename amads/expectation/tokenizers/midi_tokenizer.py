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
    
    def __init__(self, tokenizer_class: str, **params):
        """Initialize the tokenizer.
        
        Parameters
        ----------
        tokenizer_class : str
            Name of the miditok tokenizer class to use ('REMI', 'TSD', or 'MIDILike')
        params : dict
            Additional parameters to pass to TokenizerConfig
        """
        check_python_package_installed('miditok')
        from miditok import REMI, TSD, MIDILike, TokenizerConfig
        
        tokenizer_map = {
            'REMI': REMI,
            'TSD': TSD,
            'MIDILike': MIDILike
        }
        
        if tokenizer_class not in tokenizer_map:
            raise ValueError(f"Unknown tokenizer class: {tokenizer_class}. Must be one of {list(tokenizer_map.keys())}")
            
        config = TokenizerConfig(use_tempos=True, **params)
        self._tokenizer = tokenizer_map[tokenizer_class](config)

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
        """Add timing information to tokens.
        
        This method needs to be implemented by specific tokenizer classes to add
        appropriate timing information based on their token format.
        
        Args:
            raw_tokens: List of raw token values from the tokenizer
            
        Returns:
            List of Token objects with timing information added where appropriate
        """
        raise NotImplementedError("_add_timing must be implemented by subclasses")

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
        super().__init__('REMI', **params)

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
        super().__init__('TSD', **params)
    
    def _add_timing(self, raw_tokens: List[Any]) -> List[Token]:
        """Add TSD-specific timing to tokens."""
        tokens = []
        current_time = 0.0  # Start at zero
        current_tempo = 120.0  # Default tempo in BPM
        vocab = {v: k for k, v in self._tokenizer.vocab.items()}
        
        for tok in raw_tokens:
            token_str = vocab.get(tok, str(tok))
            
            try:
                if token_str.startswith('TimeShift_'):
                    # Format is (num_beats, num_samples, resolution) -- source: https://miditok.readthedocs.io/en/latest/configuration.html
                    # e.g., 2.3.8 means 2 + 3/8 beats
                    parts = token_str.split('_')[1].split('.')
                    if len(parts) == 3:
                        num_beats = int(parts[0])
                        num_samples = int(parts[1])
                        resolution = int(parts[2])
                        shift = num_beats + (num_samples / resolution)
                        time_delta = shift * (60.0 / current_tempo)
                        current_time += time_delta
                elif token_str.startswith('Tempo_'):
                    current_tempo = float(token_str.split('_')[1])
            except Exception as e:
                pass
            
            if token_str.startswith('Pitch_'):
                tokens.append(Token(value=tok, start_time=current_time))
            else:
                tokens.append(Token(value=tok))
                
        return tokens

class MIDILikeTokenizer(SymusicTokenizer):
    """MIDI-Like tokenizer using miditok's implementation."""
    
    def __init__(self, **params):
        super().__init__('MIDILike', **params)
    
    def _add_timing(self, raw_tokens: List[Any]) -> List[Token]:
        """Add MIDI-Like specific timing to tokens."""
        tokens = []
        current_time = 0.0  # Start at zero
        current_tempo = 120.0  # Default tempo in BPM
        vocab = {v: k for k, v in self._tokenizer.vocab.items()}
        
        for tok in raw_tokens:
            token_str = vocab.get(tok, str(tok))
            
            try:
                if token_str.startswith('TimeShift_'):
                    parts = token_str.split('_')[1].split('.')
                    if len(parts) == 3:
                        num_beats = int(parts[0])
                        num_samples = int(parts[1])
                        resolution = int(parts[2])
                        shift = num_beats + (num_samples / resolution)
                        time_delta = shift * (60.0 / current_tempo)
                        current_time += time_delta
                elif token_str.startswith('Tempo_'):
                    current_tempo = float(token_str.split('_')[1])
            except Exception as e:
                pass
            
            if token_str.startswith('NoteOn_'):
                tokens.append(Token(value=tok, start_time=current_time))
            else:
                tokens.append(Token(value=tok))
                
        return tokens