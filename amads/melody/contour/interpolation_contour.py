"""Calculates the Interpolation Contour of a melody, along with related features, as
implemented in the FANTASTIC toolbox of Müllensiefen (2009) [1].
"""

__author__ = "David Whyatt"

import numpy as np


class InterpolationContour:
    """Class for calculating and analyzing interpolation contours of melodies, according to
    Müllensiefen (2009) [1]. This representation was first formalised by Steinbeck (1982)
    [2], and informed a varient of the present implementation in Müllensiefen & Frieler
    (2004) [3].
    """

    def __init__(self, times: list[float], pitches: list[int]):
        """Initialize with time and pitch values.

        Parameters
        ----------
        times : list[float]
            Array of onset times in seconds
        pitches : list[int]
            Array of pitch values

        Raises
        ------
        ValueError
            If times and pitches are not the same length

        Examples
        --------
        >>> happy_birthday_pitches = [
        ...     60, 60, 62, 60, 65, 64, 60, 60, 62, 60, 67, 65,
        ...     60, 60, 72, 69, 67, 65, 64, 70, 69, 65, 67, 65
        ... ]
        >>> happy_birthday_times = [
        ...     0, 0.75, 1, 2, 3, 4, 6, 6.75, 7, 8, 9, 10,
        ...     12, 12.75, 13, 14, 15, 16, 17, 18, 18.75, 19, 20, 21
        ... ]
        >>> ic = InterpolationContour(happy_birthday_times, happy_birthday_pitches)
        >>> ic.direction_changes
        0.6
        >>> ic.class_label
        'ccbc'
        >>> ic.mean_gradient
        2.512...
        >>> ic.gradient_std
        5.496...
        >>> ic.global_direction
        1

        References
        ----------
        [1] Müllensiefen, D. (2009). Fantastic: Feature ANalysis Technology Accessing
        STatistics (In a Corpus): Technical Report v1.5
        [2] W. Steinbeck, Struktur und Ähnlichkeit: Methoden automatisierter
            Melodieanalyse. Bärenreiter, 1982.
        [3] Müllensiefen, D. & Frieler, K. (2004). Cognitive Adequacy in the Measurement
        of Melodic Similarity: Algorithmic vs. Human Judgments
        """
        if len(times) != len(pitches):
            raise ValueError("Times and pitches must have the same length")
        self.times = times
        self.pitches = pitches
        self.contour = self.calculate_interpolation_contour()

    @staticmethod
    def _find_contour_extrema(pitches: list[int]) -> list[int]:
        """Determine contour extremum notes (local minima and maxima, including endpoints).

        Excludes changing notes (notae cambiatae).

        Parameters
        ----------
        pitches : list[int]
            List of MIDI pitch values

        Returns
        -------
        list[int]
            Array of indices corresponding to extrema points in the melody
        """
        extrema_indices = []
        n = len(pitches)

        # Always include first and last notes
        extrema_indices.append(0)

        # Find local extrema
        for i in range(1, n - 1):
            # Check previous and next notes
            prev_pitch = pitches[i - 1]
            curr_pitch = pitches[i]
            next_pitch = pitches[i + 1]

            # Skip changing notes (where adjacent pitches are equal)
            if prev_pitch == next_pitch:
                continue

            # Check if it's a peak or valley
            is_peak = prev_pitch < curr_pitch and next_pitch < curr_pitch
            is_valley = prev_pitch > curr_pitch and next_pitch > curr_pitch

            if is_peak or is_valley:
                # Additional check for equal adjacent notes
                is_extremum = True
                if i > 1 and prev_pitch == curr_pitch:
                    is_extremum = (
                        (pitches[i - 2] - curr_pitch) * (next_pitch - curr_pitch)
                    ) < 0
                if i < n - 2 and next_pitch == curr_pitch:
                    is_extremum = (
                        (prev_pitch - curr_pitch) * (pitches[i + 2] - curr_pitch)
                    ) < 0
                if is_extremum:
                    extrema_indices.append(i)

        extrema_indices.append(n - 1)  # Add last note
        return extrema_indices

    def calculate_interpolation_contour(self) -> list[float]:
        """Calculate the interpolation contour representation of a melody [1].

        Returns
        -------
        list[float]
            Array containing the interpolation contour representation
        """
        # Find candidate points
        candidate_points_pitch = [self.pitches[0]]  # Start with first pitch
        candidate_points_time = [self.times[0]]  # Start with first time

        if len(self.pitches) in [3, 4]:
            # Special case for very short melodies
            for i in range(1, len(self.pitches) - 1):
                if (
                    self.pitches[i] > self.pitches[i - 1]
                    and self.pitches[i] > self.pitches[i + 1]
                ) or (
                    self.pitches[i] < self.pitches[i - 1]
                    and self.pitches[i] < self.pitches[i + 1]
                ):
                    candidate_points_pitch.append(self.pitches[i])
                    candidate_points_time.append(self.times[i])
        else:
            # For longer melodies
            for i in range(2, len(self.pitches) - 2):
                if (
                    (
                        self.pitches[i - 1] < self.pitches[i]
                        and self.pitches[i] > self.pitches[i + 1]
                    )
                    or (
                        self.pitches[i - 1] > self.pitches[i]
                        and self.pitches[i] < self.pitches[i + 1]
                    )
                    or (
                        self.pitches[i - 1] == self.pitches[i]
                        and self.pitches[i - 2] < self.pitches[i]
                        and self.pitches[i] > self.pitches[i + 1]
                    )
                    or (
                        self.pitches[i - 1] < self.pitches[i]
                        and self.pitches[i] == self.pitches[i + 1]
                        and self.pitches[i + 2] > self.pitches[i]
                    )
                    or (
                        self.pitches[i - 1] == self.pitches[i]
                        and self.pitches[i - 2] > self.pitches[i]
                        and self.pitches[i] < self.pitches[i + 1]
                    )
                    or (
                        self.pitches[i - 1] > self.pitches[i]
                        and self.pitches[i] == self.pitches[i + 1]
                        and self.pitches[i + 2] < self.pitches[i]
                    )
                ):
                    candidate_points_pitch.append(self.pitches[i])
                    candidate_points_time.append(self.times[i])

        # Initialize turning points with first note
        turning_points_pitch = [self.pitches[0]]
        turning_points_time = [self.times[0]]

        # Find turning points
        if len(candidate_points_pitch) > 2:
            for i in range(1, len(self.pitches) - 1):
                if self.times[i] in candidate_points_time:
                    if self.pitches[i - 1] != self.pitches[i + 1]:
                        turning_points_pitch.append(self.pitches[i])
                        turning_points_time.append(self.times[i])

        # Add last note
        turning_points_pitch.append(self.pitches[-1])
        turning_points_time.append(self.times[-1])

        # Calculate gradients
        gradients = np.diff(turning_points_pitch) / np.diff(turning_points_time)

        # Calculate durations
        durations = np.diff(turning_points_time)

        # Create weighted gradients vector
        samples_per_duration = np.round(durations * 10).astype(
            int
        )  # 10 samples per second
        interpolation_contour = np.repeat(gradients, samples_per_duration)

        return [float(x) for x in interpolation_contour]

    @property
    def global_direction(self) -> int:
        """Calculate the global direction of the interpolation contour by taking
        the sign of the sum of all contour values.

        Returns
        -------
        int
            1 if sum is positive, 0 if sum is zero, -1 if sum is negative
        """
        return int(np.sign(sum(self.contour)))

    @property
    def mean_gradient(self) -> float:
        """Calculate the absolute mean gradient of the interpolation contour.

        Returns
        -------
        float
            Mean of the absolute gradient values
        """
        return float(np.mean(np.abs(self.contour)))

    @property
    def gradient_std(self) -> float:
        """Calculate the standard deviation of the interpolation contour gradients.

        Returns
        -------
        float
            Standard deviation of the gradient values
        """
        return float(np.std(self.contour, ddof=1))

    @property
    def direction_changes(self) -> float:
        """Calculate the ratio of direction changes in the interpolation contour."""
        # Convert contour to numpy array for element-wise multiplication
        contour_array = np.array(self.contour)

        # Calculate products of consecutive gradients
        consecutive_products = contour_array[:-1] * contour_array[1:]

        # Get signs of products and count negative ones (direction changes)
        product_signs = np.sign(consecutive_products)
        direction_changes = np.sum(np.abs(product_signs[product_signs == -1]))

        # Count total gradient changes (where consecutive values are different)
        total_changes = np.sum(contour_array[:-1] != contour_array[1:])

        # Avoid division by zero
        if total_changes == 0:
            return 0.0

        return float(direction_changes / total_changes)

    @property
    def class_label(self) -> str:
        """Classify an interpolation contour into gradient categories.

        The contour is sampled at 4 equally spaced points and each gradient is
        normalized and classified into one of 5 categories:

        - 'a': Strong downward (-2) - normalized gradient <= -1.45
        - 'b': Downward (-1) - normalized gradient between -1.45 and -0.45
        - 'c': Flat (0) - normalized gradient between -0.45 and 0.45
        - 'd': Upward (1) - normalized gradient between 0.45 and 1.45
        - 'e': Strong upward (2) - normalized gradient >= 1.45

        Returns
        -------
        str
            String of length 4 containing letters a-e representing the gradient
            categories at 4 equally spaced points in the contour
        """
        # Sample the contour at 4 equally spaced points
        # Get 4 equally spaced indices
        n = len(self.contour)
        indices = np.linspace(0, n - 1, 4, dtype=int)

        # Sample the contour at those indices
        sampled_points = [self.contour[i] for i in indices]

        # Normalize the gradients to a norm where value of 1 corresponds to a semitone
        # change in pitch over 0.25 seconds. Given that base pitch and time units are
        # 1 second and 1 semitone respectively, just divide by 4
        norm_gradients = np.array(sampled_points) * 0.25

        classes = []
        for grad in norm_gradients:
            if grad <= -1.45:
                classes.append("a")  # strong down
            elif -1.45 < grad <= -0.45:
                classes.append("b")  # down
            elif -0.45 < grad < 0.45:
                classes.append("c")  # flat
            elif 0.45 <= grad < 1.45:
                classes.append("d")  # up
            else:
                classes.append("e")  # strong up

        return "".join(classes)
