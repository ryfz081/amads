import os
import sys

import pytest

ci_groups = {
    "tests_melsim": [
        "tests/test_melsim.py",
        "amads/melody/similarity/melsim.py",
        "demos/melsim.py",
    ]
}

paths_in_ci_groups = [path for paths in ci_groups.values() for path in paths]


def run_main_tests():
    paths_to_ignore = paths_in_ci_groups
    pytest_args = [f"--ignore={path}" for path in paths_to_ignore]
    sys.exit(pytest.main(pytest_args))


def run_ci_group_tests(job_name):
    paths_to_test = ci_groups[job_name]
    sys.exit(pytest.main(paths_to_test))


def should_run(path):
    if not os.environ.get("CI"):
        return True

    job = os.environ.get("GITHUB_JOB")
    if not job:
        raise ValueError("GITHUB_JOB environment variable not set")

    if job == "tests_main":
        return path not in paths_in_ci_groups

    return path in ci_groups[job]
