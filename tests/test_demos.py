import os
import runpy

import pytest


@pytest.mark.parametrize(
    "script", [f for f in os.listdir("demos") if f.endswith(".py")]
)
def test_demos_run_without_errors(script):
    # Skip durdist2 demo in all environments
    if script == "durdist2.py":
        pytest.skip(
            "Skipping durdist2 demo (see https://github.com/music-computing/amads/issues/43)"
        )

    # Only apply CI-specific rules if running on CI
    if os.environ.get("CI"):
        is_melsim_job = os.environ.get("GITHUB_JOB") == "tests-melsim"

        # Only run melsim.py in the melsim job, skip all other demos
        if is_melsim_job:
            if script != "melsim.py":
                pytest.skip("Skipping this demo, test-melsim only runs the melsim demo")
        # Skip melsim.py in the main job, run all other demos
        else:
            if script == "melsim.py":
                pytest.skip("Skipping melsim demo (only runs in the melsim test job)")

    script_path = os.path.join("demos", script)
    runpy.run_path(script_path, run_name="__main__")
