import os
import runpy

import pytest

from tests.utils_test import only_on_ci_job


def run_demo(script_name):
    script_path = os.path.join("demos", script_name)
    runpy.run_path(script_path, run_name="__main__")


@pytest.mark.parametrize(
    "script", [f for f in os.listdir("demos") if f.endswith(".py")]
)
def test_demos_run_without_errors(script):
    if script == "durdist2.py":
        pytest.skip(
            "Skipping durdist2 demo (see https://github.com/music-computing/amads/issues/43)"
        )

    if script == "melsim.py":
        only_on_ci_job("tests_melsim")(run_demo)(script)
    else:
        run_demo(script)
