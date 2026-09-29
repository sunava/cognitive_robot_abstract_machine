"""
Expanded laboratory controls retain observed state and controller ownership.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

# %% browser-independent instrument UI


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_medical_lab_controls() -> None:
    """
    Verify scene cameras, mechanism commands and authoritative observations.
    """
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).parent / "js/test_medical_lab.js")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
