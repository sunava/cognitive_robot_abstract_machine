"""
Measured dosing is rendered on the stationary laboratory instrument.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% dosing view
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_pipetting_view() -> None:
    """
    Verify the live display, continuous stream and reservoir levels.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_pipetting_view.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
