"""
Measured laboratory bodies retain world poses inside articulated viewer models.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% measured model bodies
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_body_poses() -> None:
    """
    Verify authoritative body poses, namespaces and recorded interpolation.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_body_poses.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
