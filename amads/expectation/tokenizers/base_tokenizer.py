"""
Base classes for all tokenization approaches in the AMADS library.
All specific tokenizer implementations should inherit from these base classes.
"""

from typing import List
from amads.core.basics import Score, Note
from amads.pitch.ismonophonic import ismonophonic
from abc import ABC, abstractmethod

class Token:
    """
    Base token class that defines core functionality all tokens must share.
    All specific token types should inherit from this class and maintain
    compatibility with its basic interface.
    
    Attributes:
        value: The value this token represents
        start_time: Starting time of the token's span (None for timeless tokens)
        end_time: Ending time of the token's span (None for timeless tokens)
        name: String representation of the token's type/name (e.g., 'Pitch_60'). Optional and not always applciable.
    """
    def __init__(self, value, start_time: float = None, end_time: float = None, name: str = None):
        self.value = value
        self.start_time = start_time
        self.end_time = end_time
        self.name = name
        
        # Simple validation that end isn't before start
        if start_time is not None and end_time is not None:
            assert start_time <= end_time, "End time cannot be before start time"

    def __eq__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        if self.start_time is None:
            return str(self.value)
        return f"{self.value} ({self.start_time}->{self.end_time})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

class Tokenizer(ABC):
    """
    Abstract base class for all tokenizers.
    Defines the basic interface that all tokenizer implementations must follow.
    """
    @abstractmethod    
    def tokenize(self, x) -> List[Token]:
        """Convert input data into a sequence of tokens.

        Parameters
        ----------
        x : Any
            Input data to be tokenized (type depends on specific tokenizer)

        Returns
        -------
        List[Token]
            A sequence of tokens representing the input
        """
        pass

