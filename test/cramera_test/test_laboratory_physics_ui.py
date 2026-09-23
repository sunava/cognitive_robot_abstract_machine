"""
Physics controls send targets while displaying only simulated object poses.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% laboratory physics controls
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_physics_controls() -> None:
    """
    Exercise target serialization, authoritative poses and contact feedback.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_physics.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
