"""
The laboratory contact overlay can be hidden without stopping physics.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% scene layers
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_contact_layer() -> None:
    """
    Verify the actual scene adapter and checkbox retain live hidden contacts.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_contact_layer.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
