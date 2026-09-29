"""
The expanded room preserves calibrated work and leaves usable robot aisles.
"""

from __future__ import annotations

import importlib
import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest


# %% Authored layout
@pytest.fixture
def large_layout(monkeypatch: pytest.MonkeyPatch):
    """
    Load room authoring independently of Blender.
    """
    directory = Path(__file__).resolve().parents[2] / "cramera/tools/laboratory"
    monkeypatch.syspath_prepend(str(directory / "instruments"))
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("large_layout")


def test_rear_workbench_leaves_pr2_aisle(large_layout) -> None:
    """
    The new bank does not squeeze circulation behind the calibrated preparation bench.
    """
    layout = large_layout.LargeLaboratoryLayout()
    assert layout.rear_bench_front - layout.original_bench_back >= 1.4


def test_every_instrument_has_one_named_station(large_layout) -> None:
    """
    All separate instrument resources appear once and belong to a camera station.
    """
    layout = large_layout.LargeLaboratoryLayout()
    instruments = layout.placements()
    catalogue = importlib.import_module("catalog").build_devices()
    assert {item.identifier for item in instruments} == {
        item.identifier for item in catalogue
    }
    assert len(instruments) == len(catalogue)
    assert {item.station for item in instruments} <= {
        station.identifier for station in layout.stations()
    }


def test_room_surface_matches_existing_work_height(large_layout) -> None:
    """
    All new benchtops meet the established ninety-centimetre preparation surface.
    """
    layout = large_layout.LargeLaboratoryLayout()
    model = layout.room_definition()
    model.validate()
    worktops = [
        shape
        for part in model.parts
        for shape in part.shapes
        if shape.name.startswith("Worktop")
    ]
    assert len(worktops) == 3
    for worktop in worktops:
        assert worktop.center[2] + worktop.dimensions[2] / 2 == pytest.approx(
            layout.work_height
        )


# %% Composed exported world
@pytest.fixture
def expanded_bundle():
    """
    Inspect generated resources when the optional authored model library is installed.
    """
    resources = Path(__file__).resolve().parents[2] / "cramera/src/cramera/resources"
    directory = resources / "aicor_medical_lab_large"
    if not (directory / "scene.json").is_file():
        pytest.skip("Expanded laboratory assets have not been generated")
    return (
        directory,
        json.loads((directory / "scene.json").read_text()),
        ElementTree.parse(directory / "environment.urdf").getroot(),
    )


def test_preparation_joint_frames_are_unchanged(expanded_bundle) -> None:
    """
    Existing calibrated grasp and manipulation coordinates survive composition exactly.
    """
    directory, _, root = expanded_bundle
    source = ElementTree.parse(
        directory.parent / "aicor_medical_lab/environment.urdf"
    ).getroot()
    exported = {
        joint.get("name"): ElementTree.tostring(joint)
        for joint in root.findall("joint")
    }
    for joint in source.findall("joint"):
        assert exported[joint.get("name")] == ElementTree.tostring(joint)


def test_original_free_objects_keep_their_poses(expanded_bundle) -> None:
    """
    Existing sample tubes and stopper remain usable by calibrated robot plans.
    """
    directory, scene, _ = expanded_bundle
    source = json.loads((directory.parent / "aicor_medical_lab/scene.json").read_text())
    assert scene["objects"] == source["objects"]
    for key in (
        "rackSlots",
        "tubes",
        "centrifuge",
        "pipetting",
        "refrigerator",
        "scale",
    ):
        assert scene["laboratory"][key] == source["laboratory"][key]


def test_composed_world_is_one_connected_environment(expanded_bundle) -> None:
    """
    All devices and semantic targets connect to the preserved laboratory root.
    """
    _, scene, root = expanded_bundle
    assert len([model for model in scene["models"] if not model["robot"]]) == 1
    names = [link.get("name") for link in root.findall("link")]
    assert len(names) == len(set(names))
    children = {}
    child_names = []
    for joint in root.findall("joint"):
        parent, child = joint.find("parent").get("link"), joint.find("child").get(
            "link"
        )
        assert parent in names and child in names
        children.setdefault(parent, []).append(child)
        child_names.append(child)
    assert len(child_names) == len(set(child_names))
    assert set(names) - set(child_names) == {"laboratory_root"}
    pending, visited = ["laboratory_root"], set()
    while pending:
        current = pending.pop()
        assert current not in visited
        visited.add(current)
        pending.extend(children.get(current, []))
    assert visited == set(names)


def test_assembled_meshes_are_local_and_present(expanded_bundle) -> None:
    """
    The new world can move together with its copied resource folders.
    """
    directory, _, root = expanded_bundle
    for mesh in root.findall(".//mesh"):
        filename = Path(mesh.get("filename"))
        assert not filename.is_absolute()
        assert ".." not in filename.parts
        assert (directory / filename).is_file()
    for collision in root.findall(".//collision"):
        assert collision.get("name")


def test_mounted_probe_follows_its_adjustable_support(expanded_bundle) -> None:
    """
    Moving the electrode support carries the independently authored probe.
    """
    _, _, root = expanded_bundle
    attachment = root.find("joint[@name='ph_probe__placement_joint']")
    assert attachment.find("parent").get("link") == "ph_meter__ph_meter_probe_mount"


def test_runtime_limits_match_the_loadable_urdf(expanded_bundle) -> None:
    """
    Viewer controls and simulated actuators share exact exported joint limits.
    """
    _, scene, root = expanded_bundle
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    for instrument in scene["medicalLab"]["instruments"]:
        for metadata in instrument["joints"]:
            joint = joints[metadata["name"]]
            limit = joint.find("limit")
            assert metadata["effort"] == float(limit.get("effort"))
            assert metadata["velocity"] == float(limit.get("velocity"))
            if joint.get("type") != "continuous":
                assert metadata["lower"] == float(limit.get("lower"))
                assert metadata["upper"] == float(limit.get("upper"))
                assert metadata["lower"] <= metadata["inspection"] <= metadata["upper"]
