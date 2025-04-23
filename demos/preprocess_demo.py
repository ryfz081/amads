from amads.core import Score
from amads.core.utils import preprocess_melody


@preprocess_melody
def get_melody_stats(
    pitches: list[int], onset_times: list[float], durations: list[float]
) -> dict:
    """Calculate basic statistics about a melody."""
    return {
        "num_notes": len(pitches),
        "pitch_range": max(pitches) - min(pitches),
        "total_duration": sum(durations),
        "average_duration": sum(durations) / len(durations),
        "first_onset": onset_times[0],
        "last_onset": onset_times[-1],
    }


@preprocess_melody
def find_leaps(pitches: list[int], onset_times: list[float]) -> list[tuple[int, int]]:
    """Find all melodic leaps (intervals larger than a major 3rd)."""
    leaps = []
    for i in range(len(pitches)):
        interval = abs(pitches[i] - pitches[i - 1])
        if interval > 4:  # larger than major 3rd
            leaps.append((onset_times[i], interval))
    return leaps


def main():
    # Create a simple melody (C major scale with one leap)
    score = Score.from_melody(
        pitches=[60, 62, 64, 65, 67, 69, 71, 72, 60],
        durations=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0],
    )

    # Get basic statistics about the melody
    # We use the score object directly, as the preprocess_melody decorator will extract the
    # pitches, onset_times, and durations automatically
    stats = get_melody_stats(score=score)
    print("Melody Statistics:")
    print(f"Number of notes: {stats['num_notes']}")
    print(f"Pitch range: {stats['pitch_range']} semitones")
    print(f"Total duration: {stats['total_duration']} beats")
    print(f"Average note duration: {stats['average_duration']:.2f} beats")
    print(f"First note at: {stats['first_onset']} beats")
    print(f"Last note at: {stats['last_onset']} beats")

    # Find melodic leaps
    # As before, we can just pass the score object directly into the function
    leaps = find_leaps(score=score)
    print("\nMelodic Leaps:")
    for onset, interval in leaps:
        print(f"Leap of {interval} semitones at {onset} beats")


if __name__ == "__main__":
    main()
