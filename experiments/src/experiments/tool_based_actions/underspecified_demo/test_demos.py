"""
Smoke tests running each underspecified demo end to end on the simulated robot.

Every demo initializes ROS and builds its own world, so each one runs in its own
process. The tests live next to the demos instead of the shared test tree because each
run is a full simulation taking minutes.
"""

import subprocess
import sys

import pytest

DEMO_MODULES = (
    "experiments.tool_based_actions.underspecified_demo.demo_cutting",
    "experiments.tool_based_actions.underspecified_demo.demo_mixing",
    "experiments.tool_based_actions.underspecified_demo.demo_pouring",
    "experiments.tool_based_actions.underspecified_demo.demo_wiping",
)
"""
The demo modules under test, each runnable as ``python -m <module>``.
"""

DEMO_TIMEOUT = 1200.0
"""
Wall-clock limit in seconds for one demo run.
"""

@pytest.mark.parametrize("demo_module", DEMO_MODULES)
def test_underspecified_demo_runs_to_completion(demo_module: str) -> None:
    """
    Run one underspecified demo in an isolated process and expect a clean exit.
    """
    completed_process = subprocess.run(
        [sys.executable, "-m", demo_module], timeout=DEMO_TIMEOUT
    )
    assert completed_process.returncode == 0
