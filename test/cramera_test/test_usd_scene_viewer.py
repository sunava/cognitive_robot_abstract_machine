"""
The viewer renders imported scene bundles with the rendering, camera and lighting
fields they carry, and leaves bundles without those fields as they were.
"""

from __future__ import annotations

import shutil
import subprocess
from enum import StrEnum
from pathlib import Path

import pytest

# %% node test files of the viewer modules


class ViewerModuleTest(StrEnum):
    """
    Node test file of one viewer module under ``test/cramera_test/js``.
    """

    SCENE_RENDERING = "test_scene_rendering.js"
    """
    The ``rendering`` block of scene.json and its defaults.
    """

    SCENE_CAMERA = "test_scene_camera.js"
    """
    The lens and far plane of the ``camera`` block.
    """

    SHADOW_FIT = "test_shadow_fit.js"
    """
    The key light's shadow camera fitted to the shadow bounds.
    """

    LAB_ENVIRONMENT = "test_lab_environment.js"
    """
    Image-based lighting and the key light from a lighting description.
    """

    AMBIENT_OCCLUSION_OUTPUT = "test_ambient_occlusion_output.js"
    """
    The colour encoding of frames drawn through the ambient-occlusion composer.
    """

    SCENE_LIGHTING_PANEL = "test_scene_lighting_panel.js"
    """
    The scene panel applying a bundle's floor, shadow, lighting and camera settings.
    """


# %% running them


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("module_test", list(ViewerModuleTest))
def test_viewer_module(module_test: ViewerModuleTest) -> None:
    """
    Run one viewer module's node tests, failing with their output.
    """
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).parent / "js" / module_test)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
