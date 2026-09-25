"""
Laboratory asset credits preserve permission and prior-license notices.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% literal-text laboratory disclosure
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_asset_rights_controls() -> None:
    """
    Exercise the mounted disclosure, inert metadata and scene cleanup.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js/test_laboratory_asset_rights.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
