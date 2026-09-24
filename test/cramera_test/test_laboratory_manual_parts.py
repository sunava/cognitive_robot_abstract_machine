"""
Device callouts follow authored parts and observed operating knowledge.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% annotated instrument knowledge
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_manual_parts() -> None:
    """
    Check mesh anchors, manual revisions and unknown observations.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_manual_parts.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
