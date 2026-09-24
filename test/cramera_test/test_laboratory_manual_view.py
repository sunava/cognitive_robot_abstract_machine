"""
Instrument annotations stay attached to articulated geometry.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% interactive instrument guide
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_manual_view() -> None:
    """
    Exercise projection, annotation layout, accessibility and cleanup.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_manual_view.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
