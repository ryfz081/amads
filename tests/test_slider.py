from musmart.core.basics import Note, Part, Score


def test_slider():
    score = Score(
        content=Part([
            Note(pitch=pitch)
            for pitch in [60, 60, 67, 67, 69, 69, 67]
        ])
    )

    score