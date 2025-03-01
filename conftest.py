from pathlib import Path

from amads.ci import in_amads_root_directory, should_run


def pytest_ignore_collect(path):
    """
    Tells pytest which files to ignore when collecting and running tests.
    Returns True if the file should be ignored.
    """
    assert in_amads_root_directory()

    relative_path = Path(path).resolve().relative_to(Path.cwd())

    if not should_run(relative_path):
        return True
