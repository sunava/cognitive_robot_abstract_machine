"""
Check standalone scene document editing without a live browser.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

# %% scene document


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_scene_document_edits() -> None:
    """
    Preserve independent object poses and validate portable scene documents.
    """
    result = subprocess.run(
        ["node", str(Path(__file__).parent / "js" / "test_scene_builder_state.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_scene_draft_api() -> None:
    """
    Use the active mount and distinguish rejected saves from expired sessions.
    """
    result = subprocess.run(
        ["node", str(Path(__file__).parent / "js" / "test_scene_builder_api.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_scene_editor_without_webgl() -> None:
    """
    Edit and persist robot and furniture poses when rendering is unavailable.
    """
    result = subprocess.run(
        ["node", str(Path(__file__).parent / "js" / "test_scene_builder_page.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
