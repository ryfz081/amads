from functools import cache
from types import SimpleNamespace

from amads.core.basics import Score
from amads.pitch.ismonophonic import ismonophonic
from amads.utils import check_python_package

base_packages = ["base", "utils"]
cran_packages = ["tibble", "R6", "remotes"]
github_packages = ["melsim"]

R = SimpleNamespace()


@cache
def load_melsim():
    check_python_package("pandas")
    check_python_package("rpy2")

    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    check_packages_installed()
    import_packages()


def requires_melsim(func):
    def wrapper(*args, **kwargs):
        load_melsim()
        return func(*args, **kwargs)

    return wrapper


def check_packages_installed():
    from rpy2.robjects.packages import importr, isinstalled

    utils = importr("utils")
    remotes = importr("remotes")
    for package in cran_packages:
        if not isinstalled(package):
            response = input(
                f"Package '{package}' is not installed. Would you like to install it? (y/n): "
            )
            if response.lower() != "y":
                raise ImportError(f"Package '{package}' is required but not installed")

            utils.install_packages(package)

    for package in github_packages:
        if not isinstalled(package):
            response = input(
                f"Package '{package}' is not installed. Would you like to install it? (y/n): "
            )
            if response.lower() != "y":
                raise ImportError(f"Package '{package}' is required but not installed")

            remotes.install_github(package)


def import_packages():
    from rpy2.robjects.packages import importr

    all_packages = base_packages + cran_packages + github_packages
    for package in all_packages:
        R[package] = importr(package)


@requires_melsim
def get_similarity(melody_1: Score, melody_2: Score, method: str):
    pass_melody_to_r(melody_1, "melody_1")
    pass_melody_to_r(melody_2, "melody_2")
    load_similarity_measure(method)
    return _get_similarity("melody_1", "melody_2", method)


loaded_melodies = {}


def pass_melody_to_r(melody: Score, name: str):
    """Convert a Score to a format compatible with melsim R package.

    Args:
        melody: Score object containing a monophonic melody

    Returns:
        A melsim Melody object
    """
    import rpy2.robjects as ro

    if name in loaded_melodies and loaded_melodies[name] is melody:
        return

    assert ismonophonic(melody)

    # Flatten the score to get notes in order
    notes = melody.flatten(collapse=True)

    # Extract onset, pitch, duration for each note
    onsets = [note.start for note in notes]
    pitches = [note.keynum for note in notes]
    durations = [note.duration for note in notes]

    # Create R tibble
    tibble = R.tibble(onset=R.c(*onsets), pitch=R.c(*pitches), duration=R.c(*durations))

    ro.r.assign(f"{name}", ro.r("melody_factory$new")(mel_data=tibble))
    loaded_melodies[name] = melody


@cache
def load_similarity_measure(method: str):
    import rpy2.robjects as ro

    ro.r.assign(
        f"{method}_sim",
        ro.r("sim_measure_factory$new")(
            name=method,
            full_name=method,
            transformation="pitch",
            parameters=ro.ListVector({}),
            sim_measure=method,
        ),
    )


def _get_similarity(melody_1: str, melody_2: str, method: str):
    """
    Use the melsim R package to get the similarity between two melodies.

    Args:
        melody_1: Name of the first melody. This should have already been passed to R (see pass_melody_to_r).
        melody_2: Name of the second melody. This should have already been passed to R (see pass_melody_to_r).
        method: Name of the similarity method. This should have already been loaded (see load_similarity_measure).

    Returns:
        The similarity between the two melodies
    """
    import rpy2.robjects as ro

    return ro.r(f"{melody_1}$similarity")(
        ro.r(f"{melody_2}"), ro.r(f"{method}_sim")
    ).rx2("sim")[0]
