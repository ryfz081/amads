"""
Tokenization schemes for MIDI data in the AMADS library.
Currently implements:
- MIDI-Like tokenization (based on MidiTok's implementation)
"""

from enum import Enum
from typing import List, Dict, Tuple
from amads.core.basics import Score, Note
from amads.expectation.tokenizer import Token, Tokenizer

class MIDITokenizer(Tokenizer):
    """Base class for all MIDI tokenization schemes.
    
    Attributes:
        EventType (Optional[Type[Enum]]): If the tokenizer uses discrete events,
            they should be defined in an EventType enum within the class.
    """
    
    EventType = None  # Default to None in base class
    
    def tokenize(self, score: Score) -> List[Token]:
        """Convert a Score object into a sequence of Tokens."""
        raise NotImplementedError
    
    def decode(self, tokens: List[Token]) -> List[Tuple[Enum, Dict]]:
        """Convert tokens back to MIDI events.
        
        Returns:
            List of tuples containing (EventType, parameters), where EventType
            is an Enum subclass specific to the tokenization scheme.
        """
        raise NotImplementedError


class TSDTokenizer(MIDITokenizer):
    """Implements Time-Shift-Duration (TSD) tokenization.
    Converts MIDI data into NOTE_ON (with velocity), TIME_SHIFT, and DURATION events.
    
    Parameters:
        ticks_per_beat: The number of ticks per quarter note (default: 24)
        max_time_shift: Maximum time shift value in ticks (default: 96, 4 beats)
        max_duration: Maximum duration value in ticks (default: 96, 4 beats)
    """
    class EventType(Enum):
        NOTE = 0
        TIME_SHIFT = 1
    
    def __init__(self, ticks_per_beat: int = 24, max_time_shift: int = 96, max_duration: int = 96):
        self.ticks_per_beat = ticks_per_beat
        self.max_time_shift = max_time_shift
        self.max_duration = max_duration

    def _get_note_value(self, pitch: int, velocity: int, duration: int) -> int:
        """Calculate NOTE token value combining pitch, velocity, and duration."""
        duration = min(duration, self.max_duration)
        return pitch + (velocity * 128) + (duration * 128 * 128)

    def _get_time_shift_value(self, time: int) -> int:
        """Calculate TIME_SHIFT token value."""
        base_value = 128 * 128 * (self.max_duration + 1)  # After all possible NOTE values
        return base_value + min(time, self.max_time_shift)

    def _score_to_events(self, score: Score) -> List[Tuple[float, Tuple[Enum, Dict]]]:
        """Convert Score to list of (time, event) pairs."""
        flat_score = score.flatten(collapse=False)
        notes = list(flat_score.find_all(Note))
        events = []
        current_time = 0

        # Sort notes by start time
        notes.sort(key=lambda n: n.start)

        for note in notes:
            start_tick = round(note.start * self.ticks_per_beat)
            duration_ticks = round(note.duration * self.ticks_per_beat)
            velocity = getattr(note, 'velocity', 64)

            # Add time shift if needed
            if start_tick > current_time:
                time_delta = start_tick - current_time
                while time_delta > 0:
                    current_shift = min(time_delta, self.max_time_shift)
                    events.append((
                        current_time,
                        (self.EventType.TIME_SHIFT, {'time': current_shift})
                    ))
                    current_time += current_shift
                    time_delta -= current_shift

            # Add note event
            events.append((
                start_tick,
                (self.EventType.NOTE, {
                    'pitch': note.keynum,
                    'velocity': velocity,
                    'duration': duration_ticks
                })
            ))
            current_time = start_tick

        return events

    def _events_to_tokens(self, events: List[Tuple[float, Tuple[Enum, Dict]]]) -> List[Token]:
        """Convert events to token sequence.
        
        Args:
            events: List of (time, (event_type, params)) tuples, sorted by time.
                
        Returns:
            List of Tokens representing the event sequence.
            
        Raises:
            ValueError: If events contain invalid parameters or time values.
        """
        if not events:
            return []
        
        tokens = []
        last_time = events[0][0]
        
        if not isinstance(last_time, (int, float)) or last_time < 0:
            raise ValueError(f"Invalid initial timestamp: {last_time}")

        for time, (event_type, params) in events:
            # Validate current timestamp
            if not isinstance(time, (int, float)):
                raise ValueError(f"Invalid timestamp type: {type(time)}")
            
            if time < last_time:
                raise ValueError(
                    f"Events must be sorted in ascending order. "
                    f"Found time {time} after {last_time}"
                )

            # Process each event type
            try:
                if event_type == self.EventType.NOTE:
                    value = self._get_note_value(
                        params['pitch'],
                        params['velocity'],
                        params['duration']
                    )
                elif event_type == self.EventType.TIME_SHIFT:
                    value = self._get_time_shift_value(params['time'])
                else:
                    raise ValueError(f"Unknown event type: {event_type}")
                
                tokens.append(Token(value))
            except KeyError as e:
                raise ValueError(f"Missing required parameter in {event_type} event: {e}")
            except Exception as e:
                raise ValueError(f"Error processing {event_type} event: {e}")
                
            last_time = time

        return tokens

    def decode(self, tokens: List[Token]) -> List[Tuple[Enum, Dict]]:
        """Convert tokens back to MIDI events."""
        base_time_shift = 128 * 128 * (self.max_duration + 1)
        events = []
        
        for token in tokens:
            value = token.value
            if not isinstance(value, int):
                raise ValueError(f"Token value must be an integer, got {type(value)}")
                
            if value < 0:
                raise ValueError(f"Token value must be non-negative, got {value}")
                
            if value < base_time_shift:  # NOTE
                duration = value // (128 * 128)
                velocity = (value % (128 * 128)) // 128
                pitch = value % 128
                
                # Validate ranges
                if not (0 <= pitch <= 127 and 0 <= velocity <= 127):
                    raise ValueError(f"Invalid note parameters: pitch={pitch}, velocity={velocity}")
                    
                events.append((
                    self.EventType.NOTE,
                    {
                        'pitch': pitch,
                        'velocity': velocity,
                        'duration': duration
                    }
                ))
            else:  # TIME_SHIFT
                time = value - base_time_shift
                if time > self.max_time_shift:
                    raise ValueError(f"Time shift {time} exceeds maximum {self.max_time_shift}")
                    
                events.append((
                    self.EventType.TIME_SHIFT,
                    {'time': time}
                ))
        return events
