"""
Contained and spilled liquid retains the authored laboratory appearance.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% liquid geometry and viewer integration
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_liquid_view() -> None:
    """
    Exercise clipped liquid surfaces and their actual Three.js scene adapter.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_liquid.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
