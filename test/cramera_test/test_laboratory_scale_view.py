"""
The balance renders measured weight in its authored laboratory display.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% scale display and lifecycle
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_scale_view() -> None:
    """
    Exercise the display with real Three.js geometry and a recorded canvas.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_scale.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
