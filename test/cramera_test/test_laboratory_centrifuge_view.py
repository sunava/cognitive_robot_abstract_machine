"""
The centrifuge's authored screen follows observed instrument state.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% physical instrument display
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_centrifuge_view() -> None:
    """
    Exercise measured countdown rendering, geometry and renderer lifecycle.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_centrifuge.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
