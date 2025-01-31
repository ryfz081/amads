"""
This is a Python wrapper for the R package 'melsim' (https://github.com/sebsilas/melsim).
This wrapper seeks to allow the user to easily interface with the melsim package using
the Score objects in AMADS.

Melsim is a package for computing similarity between melodies. It is based on SIMILE, which was
written by Daniel Müllensiefen and Klaus Frieler in 2003/2004. Melsim is used to compare two
or more melodies pairwise across a range of similarity measures. Not all similarity measures
are implemented in melsim, but the ones that are can be used in AMADS.

All of the following similarity measures are implemented and functional in melsim:
Please be aware that the names of the similarity measures are case-sensitive.

Num:        Name:
1           Jaccard
2       Kulczynski2
3            Russel
4   simple matching
5             Faith
6          Tanimoto
7              Dice
8            Mozley
9            Ochiai
10          Simpson
11           cosine
12          angular
13      correlation
14        Tschuprow
15           Cramer
16            Gower
17        Euclidean
18        Manhattan
19         supremum
20         Canberra
21            Chord
22         Geodesic
23             Bray
24          Soergel
25           Podani
26        Whittaker
27         eJaccard
28            eDice
29   Bhjattacharyya
30       divergence
31        Hellinger
32    edit_sim_utf8
33         edit_sim
34      Levenshtein
35          sim_NCD
36            const
37          sim_dtw

The following similarity measures are not currently functional in melsim:
1    count_distinct (set-based)
2          tversky (set-based)
3   braun_blanquet (set-based)
4        minkowski (vector-based)
5           ukkon (distribution-based)
6      sum_common (distribution-based)
7       distr_sim (distribution-based)
8   stringdot_utf8 (sequence-based)
9             pmi (special)
10       sim_emd (special)
"""

from functools import cache
from itertools import combinations
from types import SimpleNamespace

import numpy as np

from amads.core.basics import Note, Score
from amads.pitch.ismonophonic import ismonophonic
from amads.utils import check_python_package

base_packages = ["base", "utils"]
cran_packages = ["tibble", "R6", "remotes"]
github_packages = ["melsim"]
github_repos = {
    "melsim": "sebsilas/melsim",
}

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


def check_packages_installed(install_missing: bool = False):
    from rpy2.robjects.packages import isinstalled

    for package in cran_packages + github_packages:
        if not isinstalled(package):
            if install_missing:
                install_r_package(package)
            else:
                raise ImportError(
                    f"Package '{package}' is required but not installed. "
                    "You can run install it by running the following command: "
                    "from amads.melody.similarity.melsim import install_dependencies; install_dependencies()"
                )


def install_r_package(package: str):
    from rpy2.robjects.packages import importr

    if package in cran_packages:
        print(f"Installing CRAN package '{package}'...")
        utils = importr("utils")
        utils.install_packages(package)

    elif package in github_packages:
        print(f"Installing GitHub package '{package}'...")
        repo = github_repos[package]
        devtools = importr("devtools")
        devtools.install_github(repo, upgrade="always")

    else:
        raise ValueError(f"Unknown package type for '{package}'")


def install_dependencies():
    check_packages_installed(install_missing=True)


def import_packages():
    from rpy2.robjects.packages import importr

    all_packages = base_packages + cran_packages + github_packages
    for package in all_packages:
        setattr(R, package, importr(package))


@requires_melsim
def get_similarity(melodies: list[Score], method: str) -> dict:
    """
    Calculate pairwise similarities between all melodies using the specified method.

    Args:
        melodies: List of Score objects containing monophonic melodies
        method: Name of the similarity method to use

    Returns:
        Dictionary containing pairwise similarities with normal float values
    """
    n = len(melodies)
    similarities = np.zeros((n, n))

    # Load all melodies into R
    for i, melody in enumerate(melodies):
        pass_melody_to_r(melody, f"melody_{i + 1}")

    # Load the similarity measure
    load_similarity_measure(method)
    # Calculate pairwise similarities
    for i, j in combinations(range(n), 2):
        sim = _get_similarity(f"melody_{i + 1}", f"melody_{j + 1}", method)
        similarities[i, j] = sim
        similarities[j, i] = sim

    # Convert to dictionary of melody pairs and similarity values
    similarity_dict = {}
    for i, j in combinations(range(n), 2):
        key = (f"melody_{i + 1}", f"melody_{j + 1}")
        similarity_dict[key] = float(similarities[i, j])

    return similarity_dict


loaded_melodies = {}


def pass_melody_to_r(melody: Score, name: str):
    """Convert a Score to a format compatible with melsim R package.

    Args:
        melody: Score object containing a monophonic melody

    Returns:
        A melsim Melody object
    """
    import rpy2.robjects as ro
    from rpy2.robjects import FloatVector

    if name in loaded_melodies and loaded_melodies[name] is melody:
        return

    assert ismonophonic(melody)

    # Flatten the score to get notes in order
    flattened_score = melody.flatten(collapse=True)
    notes = list(flattened_score.find_all(Note))

    # Extract onset, pitch, duration for each note
    onsets = FloatVector([note.start for note in notes])
    pitches = FloatVector([note.keynum for note in notes])
    durations = FloatVector([note.duration for note in notes])

    # Create R tibble using tibble::tibble()
    tibble = R.tibble.tibble(onset=onsets, pitch=pitches, duration=durations)

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
