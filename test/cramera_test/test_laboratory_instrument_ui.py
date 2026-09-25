"""
Independent robot handle controls retain authoritative program ownership.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% standalone robot controls
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_instrument_controls() -> None:
    """
    Verify device commands, busy interlocks and standalone progress labels.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_instrument.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
