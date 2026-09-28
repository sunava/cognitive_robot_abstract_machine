"""
Browser controls use authoritative cable physics and expose failures.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% native cable browser controls
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_cable_lab_controls() -> None:
    """
    Exercise configuration, commands, contact feedback and polling order.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_cable_lab.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
