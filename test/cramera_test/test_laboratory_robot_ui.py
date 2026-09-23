"""
Robot launch controls follow server outcomes without browser pose edits.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% local robot execution controls
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_robot_controls() -> None:
    """
    Exercise asynchronous launch, status polling and mounted robot controls.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_robot.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
