"""
Independent workshop parts and drawers retain safe manual state.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% workshop browser controls
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_workshop_controls() -> None:
    """
    Exercise animated workshop commands through the viewer adapter.
    """
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).parent / "js/test_workshop.js")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
