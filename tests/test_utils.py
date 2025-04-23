import pytest

from amads.core import Score
from amads.core.utils import (
    dir2coll,
    hz2key_num,
    key_num2hz,
    keyname,
    preprocess_melody,
)
from amads.music import example


@pytest.fixture
def example_files():
    """Fixture providing example music files"""
    midi_file = example.fullpath("midi/sarabande.mid")
    xml_file = example.fullpath("musicxml/ex1.xml")
    return [str(midi_file), str(xml_file)]


def test_dir2coll(example_files):
    """Test the dir2coll function"""
    print("-----------Testing dir2coll function-----------")
    scores = dir2coll(example_files)
    assert isinstance(scores, dict)
    assert len(scores) == 2
    print("Scores extracted from files: ", scores.keys())


def test_hz2key_num():
    """Test converting frequencies to MIDI key numbers."""
    print("-----------Testing hz2key_num function-----------")
    assert hz2key_num(440.0).key_num == 69  # A4 (440 Hz) is MIDI key 69
    assert hz2key_num(880.0).key_num == 81  # A5 (880 Hz) is MIDI key 81


def test_key_num2hz():
    """Test converting MIDI key numbers to frequencies."""
    print("-----------Testing key_num2hz function-----------")
    assert abs(key_num2hz(69) - 440.0) < 0.01  # MIDI key 69 is 440 Hz
    assert abs(key_num2hz(81) - 880.0) < 0.01  # MIDI key 81 is 880 Hz


def test_keyname():
    """Test converting key numbers to key names."""
    print("------- Testing keyname function--------------")

    # Test for 'nameoctave' (default) detail option
    assert keyname(60) == "C4"  # MIDI key 60 should be 'C4'
    assert keyname(61) == "C#4"  # MIDI key 61 should be 'C#4'
    assert keyname(69) == "A4"  # MIDI key 69 should be 'A4'

    # Test for 'nameonly' detail option
    assert keyname(60, detail="nameonly") == "C"  # Just the note name
    assert keyname(61, detail="nameonly") == "C#"  # Just the note name
    assert keyname(69, detail="nameonly") == "A"  # Just the note name

    # Test list input with 'nameoctave'
    assert keyname([60, 61, 69]) == ["C4", "C#4", "A4"]  # List of MIDI keys

    # Test list input with 'nameonly'
    assert keyname([60, 61, 69], detail="nameonly") == [
        "C",
        "C#",
        "A",
    ]  # List of note names

    # Test edge cases
    assert keyname(0) == "C-1"  # Lowest MIDI key
    assert keyname(127) == "G9"  # Highest MIDI key

    # Test invalid detail option
    with pytest.raises(ValueError):
        keyname(60, detail="invalid_option")


def test_preprocess_melody():
    """Test the preprocess_melody decorator."""
    print("------- Testing preprocess_melody decorator --------------")

    # Create a test score with known values
    score = Score.from_melody(
        pitches=[60, 62, 64], durations=[1.0, 0.5, 2.0]  # C4, D4, E4
    )

    # Test decorator with only pitches
    @preprocess_melody
    def test_pitches_only(pitches):
        return pitches

    # Test decorator with all parameters
    @preprocess_melody
    def test_all_params(pitches, onset_times, durations):
        return pitches, onset_times, durations

    # Test extracting just pitches
    assert test_pitches_only(score=score) == [60, 62, 64]

    # Test extracting all parameters
    pitches, onsets, durs = test_all_params(score=score)
    assert pitches == [60, 62, 64]
    assert onsets == [0.0, 1.0, 1.5]
    assert durs == [1.0, 0.5, 2.0]

    # Test with empty score
    empty_score = Score()

    @preprocess_melody
    def test_empty(pitches, durations):
        return pitches, durations

    empty_pitches, empty_durs = test_empty(score=empty_score)
    assert empty_pitches == []
    assert empty_durs == []
