import os
import sys
from pathlib import Path

import pytest

from .utils import in_amads_root_directory

# This dictionary specifies tests that are not run in the default tests_main CI job.
# The keys are the names of the CI jobs that they should be tested in,
# and the values are lists of paths to test.
ci_groups = {
    "tests_melsim": [
        "tests/test_melsim.py",
        "amads/melody/similarity/melsim.py",
        "examples/plot_melsim.py",
    ]
}

paths_in_ci_groups = [path for paths in ci_groups.values() for path in paths]
coverage_args = ["--cov=./", "--cov-report=xml"]


def run_main_tests():
    """
    Run the main tests, i.e. all tests except those in the ci_groups dictionary.
    Assumes that the working directory is the root of the repository.
    """
    paths_to_ignore = paths_in_ci_groups
    ignore_args = [f"--ignore={path}" for path in paths_to_ignore]
    pytest_args = coverage_args + ignore_args
    sys.exit(pytest.main(pytest_args))


def run_ci_group_tests(job_name):
    """
    Run the tests for a specific CI job.
    Assumes that the working directory is the root of the repository.
    """
    paths_to_test = ci_groups[job_name]
    pytest_args = coverage_args + paths_to_test
    sys.exit(pytest.main(pytest_args))


def should_run(test_path):
    """
    Determine if a test should be run based on the CI environment.

    Parameters
    ----------
    test_path : str
        The path to the test file. Should be relative to the root of the repository.

    Returns
    -------
    bool
        True if the test should be run, False otherwise.
    """
    # Confirm that we are in the AMADS root directory,
    # and that the test file is specified relative to the root.
    assert in_amads_root_directory()
    assert (Path.cwd() / Path(test_path)).exists()

    test_path = str(test_path)

    if in_venv_directory(test_path):
        print("Skipping .venv directory")
        return False

    if not in_test_directory(test_path):
        print("Skipping non-test directory")
        return False

    if os.environ.get("CI"):
        return runs_in_current_ci_job(test_path)

    return True


def in_venv_directory(test_path):
    return ".venv" in test_path


def in_test_directory(test_path):
    test_dirs = ["amads", "tests"]
    return any(test_path.startswith(dir + "/") for dir in test_dirs)


def runs_in_current_ci_job(test_path):
    job = os.environ.get("GITHUB_JOB")
    if not job:
        raise ValueError("GITHUB_JOB environment variable not set")

    return runs_in_ci_job(test_path, job)


def runs_in_ci_job(test_path, job):
    if job == "tests_main":
        # In the tests_main job, we run all tests that are not specified to be run
        # in specific other CI jobs.
        return test_path not in paths_in_ci_groups

    # In other jobs, we only run tests that are specified to be run in that job.
    return test_path in ci_groups[job]
