"""
Camera and object adapters used by the manual laboratory controls.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

# %% viewer integration


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_viewer() -> None:
    """
    Run the actual viewer adapters against scene objects and camera transforms.
    """
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).parent / "js/test_laboratory_viewer.js")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
