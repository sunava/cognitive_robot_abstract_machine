"""
Workshop controls render measured physics and send force-target commands.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% authoritative workshop physics in the browser
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_workshop_physics_controls() -> None:
    """
    Exercise command ordering and measured-pose rendering without a server.
    """
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).parent / "js/test_workshop_physics.js")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
