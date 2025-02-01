import os
import runpy

import pytest


@pytest.mark.parametrize(
    "script", [f for f in os.listdir("demos") if f.endswith(".py")]
)
def test_demos_run_without_errors(script):
    if script == "durdist2.py":
        pytest.skip(
            "Skipping durdist2 demo (see https://github.com/music-computing/amads/issues/43)"
        )

    script_path = os.path.join("demos", script)
    runpy.run_path(script_path, run_name="__main__")
