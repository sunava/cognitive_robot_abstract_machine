"""
Point-cloud browser validation and rendering honor the upload limits.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% bounded browser point clouds
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_scene_builder_point_cloud() -> None:
    """
    Exercise supported scan formats and bounded reference rendering.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js/test_scene_builder_point_cloud.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
