"""
The renamed laboratory remains reachable from existing browser controls.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% laboratory resource name and stable runtime pages
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_base_scene_navigation() -> None:
    """
    Exercise the new scene name at the renderer boundary and return links.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            "--test-name-pattern=AICOR medical lab",
            str(Path(__file__).parent / "js/test_scene_renderer_unavailable.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
