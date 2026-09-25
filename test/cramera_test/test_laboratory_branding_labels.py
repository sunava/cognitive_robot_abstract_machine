"""
Public laboratory labels share one name without changing scene identities.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from cramera import laboratory_demo
from cramera.laboratory_branding import LaboratoryBranding
from cramera.laboratory_demo import (
    LaboratoryDemo,
    LaboratoryRecording,
    LaboratoryResult,
)
from cramera.laboratory_share import LaboratoryAssets, ShareOptions
from cramera.laboratory_world import LaboratoryAsset


# %% authored and published names
@pytest.fixture(autouse=True)
def recording_directory(tmp_path: Path) -> Path:
    """
    Provide the output directory normally created by the recording bundle writer.
    """
    destination = tmp_path / LaboratoryRecording.SCENE
    destination.mkdir()
    return destination


@pytest.mark.parametrize(
    "manifest",
    [
        "cramera/tools/laboratory/specification.json",
        "cramera/src/cramera/resources/aicor_medical_lab/scene.json",
    ],
)
def test_authored_laboratory_display_name_preserves_scene_identity(
    manifest: str,
) -> None:
    """
    Authored and packaged manifests expose the common display name.
    """
    root = Path(__file__).resolve().parents[2]
    scene = json.loads((root / manifest).read_text())
    assert scene["environmentName"] == LaboratoryBranding.NAME
    assert scene["name"] == LaboratoryAsset.SCENE_NAME


def test_shared_scene_picker_uses_laboratory_display_name() -> None:
    """
    The protected gateway shows the brand while retaining its allowed scene.
    """
    index = json.loads(LaboratoryAssets.scene_index())
    assert index["scenes"][0]["environment"] == LaboratoryBranding.NAME
    assert index["scenes"][0]["name"] == ShareOptions.SCENE_NAME
    assert index["default"] == ShareOptions.SCENE_NAME


def test_recorded_laboratory_uses_display_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A saved replay receives the same display name without rerunning a robot.
    """
    monkeypatch.setattr(
        laboratory_demo.paths, "local_scenes_directory", lambda: tmp_path
    )
    monkeypatch.setattr(laboratory_demo, "write_scene_index", Mock())
    scene = {"name": LaboratoryRecording.SCENE}
    monkeypatch.setattr(
        laboratory_demo, "write_recording_bundle", Mock(return_value=scene)
    )
    visualization = Mock()
    visualization.bridge.recording.stop.return_value = []
    visualization.bridge.recording.frames_per_second.return_value = 30
    result = LaboratoryResult(
        success=True,
        attached_frames=1,
        maximum_lift=0.16,
        position_error=0,
        orientation_error=0,
        minimum_collision_avoidance_goals=1,
        minimum_collision_distance=None,
        released=True,
        other_objects_unchanged=True,
        action_statuses=[],
    )
    destination = LaboratoryDemo().save_recording(visualization, result)
    recorded = json.loads((destination / LaboratoryRecording.SCENE_FILE).read_text())
    assert recorded["environmentName"] == LaboratoryBranding.NAME
    assert recorded["name"] == LaboratoryRecording.SCENE
