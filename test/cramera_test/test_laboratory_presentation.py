"""
The native laboratory run inherits its authored browser presentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cramera.laboratory_demo import LaboratoryDemo, LaboratoryRecording
from cramera.laboratory_world import LaboratoryAsset, LaboratoryWorld
from cramera.live.recording import Recording

from .test_laboratory_acceptance import TransferEvidence, transfer_evidence
from .test_laboratory_bundle import laboratory_directory
from .test_live_bundle import attached_bridge, use_scratch_scenes_directory


# %% source presentation
@pytest.fixture
def presented_demo(laboratory_directory: Path) -> LaboratoryDemo:
    """
    Use edited source settings to detect hardcoded presentation overrides.
    """
    manifest = laboratory_directory / LaboratoryAsset.SCENE
    scene = json.loads(manifest.read_text())
    scene["rendering"]["exposure"] /= 2
    scene["camera"]["position"][0] += 0.5
    manifest.write_text(json.dumps(scene))
    return LaboratoryDemo(
        laboratory=LaboratoryWorld(bundle_directory=laboratory_directory)
    )


def test_native_visualization_uses_the_source_presentation(
    presented_demo: LaboratoryDemo,
) -> None:
    """
    The robot's viewer carries the same camera and rendering as its source lab.
    """
    source = json.loads(
        (presented_demo.laboratory.bundle_directory / LaboratoryAsset.SCENE).read_text()
    )
    visualization = presented_demo.create_visualization(attached_bridge().world)
    projected = {"models": [], "objects": []}
    visualization.bridge.presentation.apply_to_scene(projected)
    assert projected["rendering"] == source["rendering"]
    assert projected["camera"] == source["camera"]
    assert "laboratory" not in projected


def test_saved_laboratory_recording_retains_source_presentation(
    presented_demo: LaboratoryDemo,
    transfer_evidence: TransferEvidence,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Publishing the measured replay must preserve the live view's source settings.
    """
    use_scratch_scenes_directory(monkeypatch, tmp_path)
    source = json.loads(
        (presented_demo.laboratory.bundle_directory / LaboratoryAsset.SCENE).read_text()
    )
    bridge = attached_bridge()
    visualization = presented_demo.create_visualization(bridge.world)
    bridge.presentation = visualization.bridge.presentation
    visualization.bridge = bridge
    bridge.snapshot()
    bridge.recording = Recording()
    bridge.recording.start()
    bridge.recording.append(bridge.state)
    destination = presented_demo.save_recording(
        visualization, transfer_evidence.validate()
    )
    saved = json.loads((destination / LaboratoryRecording.SCENE_FILE).read_text())
    assert saved["rendering"] == source["rendering"]
    assert saved["camera"] == source["camera"]
