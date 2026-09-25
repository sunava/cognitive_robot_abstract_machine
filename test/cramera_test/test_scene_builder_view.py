"""
Scene Builder transforms preserve independently placed models and subtrees.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% browser viewport contracts
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_scene_builder_view() -> None:
    """
    Exercise scene graph edits without requiring a graphics driver.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js/test_scene_builder_view.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
