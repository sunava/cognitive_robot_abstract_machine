"""
Instrument selection follows authored components and respects occlusion.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% independent instrument selection
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_devices() -> None:
    """
    Check device identities, separate metadata and the nearest visible surface.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_devices.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
