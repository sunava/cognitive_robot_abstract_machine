"""
Refrigerator controls follow observed door, storage and temperature state.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% contact storage and instrument rendering
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_refrigerator_view() -> None:
    """
    Exercise storage paths, command guards, manuals and measured display updates.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_refrigerator.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
