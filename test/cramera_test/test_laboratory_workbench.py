"""
Manual laboratory interactions preserve rack occupancy and stopper attachment.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% laboratory interactions
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_workbench() -> None:
    """
    Exercise the laboratory controller through the viewer adapter contract.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_workbench.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
