"""
Load independent laboratory bodies into a robot planning world.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest
import trimesh

from cramera.laboratory_world import (
    LaboratoryAsset,
    LaboratoryBody,
    LaboratorySlot,
    LaboratoryWorld,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world_description.connections import Connection6DoF

from .test_laboratory_bundle import laboratory_directory


# %% authored description fixture
@pytest.fixture
def laboratory_bundle(laboratory_directory: Path) -> Path:
    """
    Supply portable visual meshes alongside the real generated collisions.
    """
    for description in laboratory_directory.glob("*.urdf"):
        for visual in ElementTree.parse(description).findall(".//visual/geometry/mesh"):
            mesh_path = laboratory_directory / visual.attrib["filename"]
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            trimesh.creation.box().export(mesh_path)
    return laboratory_directory


# %% metadata and body placement
def test_slot_and_grasp_geometry_comes_from_bundle(laboratory_bundle: Path) -> None:
    """
    Planning targets agree with the same metadata used to create the rack.
    """
    laboratory = LaboratoryWorld(bundle_directory=laboratory_bundle)
    semantics = json.loads((laboratory_bundle / LaboratoryAsset.SEMANTICS).read_text())
    poses = {slot["id"]: slot["pose"] for slot in semantics["rack"]["slots"]}
    assert laboratory.source_body_name == LaboratoryBody.CLEAR_TUBE
    np.testing.assert_allclose(
        laboratory.source_pose.to_np()[:3, 3], poses[LaboratorySlot.A1][:3]
    )
    np.testing.assert_allclose(
        laboratory.target_pose.to_np()[:3, 3], poses[LaboratorySlot.A3][:3]
    )
    assert laboratory.grasp_height == semantics["tube"]["graspHeight"]
    assert laboratory.tube_height == semantics["tube"]["height"]
    assert laboratory.tube_radius == semantics["tube"]["outerRadius"]


def test_default_bundle_uses_local_scene_resolution(
    laboratory_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Default construction honors CRAMERA's scene resolver.
    """
    monkeypatch.setattr(
        "cramera.laboratory_world.paths.resolve_scene_directory",
        lambda name: laboratory_bundle if name == LaboratoryAsset.SCENE_NAME else None,
    )
    assert LaboratoryWorld().bundle_directory == laboratory_bundle


def test_world_contains_robot_and_independently_movable_tubes(
    laboratory_bundle: Path,
) -> None:
    """
    Moving one free tube preserves the neighbors and all authored collision parts.
    """
    laboratory = LaboratoryWorld(bundle_directory=laboratory_bundle)
    world = laboratory.build()
    [robot] = world.get_semantic_annotations_by_type(PR2)
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), laboratory.base_pose.to_np(), atol=1e-12
    )
    scene = json.loads((laboratory_bundle / LaboratoryAsset.SCENE).read_text())
    before = {}
    for item in scene["objects"]:
        body = world.get_body_by_name(item["key"])
        before[item["key"]] = body.global_pose.to_np().copy()
        assert isinstance(body.parent_connection, Connection6DoF)
        assert body.parent_connection.parent is world.root
        np.testing.assert_allclose(body.global_pose.to_np()[:3, 3], item["spawn"][:3])
        description = ElementTree.parse(laboratory_bundle / item["urdf"])
        assert len(body.collision) == len(description.findall(".//collision"))
        assert Path(body.visual[0].filename) == laboratory_bundle / item["mesh"]

    source = world.get_body_by_name(laboratory.source_body_name)
    source.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=2, reference_frame=world.root, child_frame=source
    )
    for item in scene["objects"]:
        if item["key"] == laboratory.source_body_name:
            continue
        np.testing.assert_allclose(
            world.get_body_by_name(item["key"]).global_pose.to_np(), before[item["key"]]
        )


def test_support_contacts_do_not_exempt_robot_or_worktop(
    laboratory_bundle: Path,
) -> None:
    """
    Only tube-to-rack contacts are permitted by the scene loader.
    """
    laboratory = LaboratoryWorld(bundle_directory=laboratory_bundle)
    world = laboratory.build()
    rack = world.get_body_by_name(LaboratoryBody.RACK)
    scene_rules = [
        rule
        for rule in world.collision_manager.ignore_collision_rules
        if isinstance(rule, AllowCollisionBetweenGroups) and rack in rule.body_group_b
    ]
    assert len(scene_rules) == 1
    [rule] = scene_rules
    assert rule.body_group_b == [rack]
    assert {body.name.name for body in rule.body_group_a} == {
        tube["key"]
        for tube in json.loads((laboratory_bundle / LaboratoryAsset.SCENE).read_text())[
            "laboratory"
        ]["tubes"]
    }


# %% authored cabinet structure
def test_cabinet_collision_blocks_pr2_from_entering_positive_side(
    laboratory_bundle: Path,
) -> None:
    """
    The visible closed cabinet front blocks a base entering from the bench front.
    """
    laboratory = LaboratoryWorld(bundle_directory=laboratory_bundle)
    world = laboratory.build()
    [robot] = world.get_semantic_annotations_by_type(PR2)
    cabinet = world.get_body_by_name(LaboratoryBody.CABINET)
    base = world.get_body_by_name("base_link")
    detector = world.collision_manager.collision_detector
    assert detector.check_collision_between_bodies(base, cabinet) is None

    robot.set_root_pose(Pose.from_xyz_rpy(x=0.52, y=-0.40, reference_frame=world.root))
    contact = detector.check_collision_between_bodies(base, cabinet)
    assert contact is not None
    assert contact.distance < 0
    assert len(cabinet.visual) == 0


def test_cabinet_panels_leave_sliding_drawer_bay_open(laboratory_bundle: Path) -> None:
    """
    Individual panel collisions do not fill the functional drawer cavity.
    """
    laboratory = LaboratoryWorld(bundle_directory=laboratory_bundle)
    world = laboratory.build()
    cabinet = world.get_body_by_name(LaboratoryBody.CABINET)
    drawer = world.get_body_by_name("laboratory_drawer")
    cavity_center = drawer.global_pose.to_np()[:3, 3]
    for shape in cabinet.collision:
        center = shape.origin.to_np()[:3, 3]
        half_extents = shape.scale.to_np() / 2
        assert not np.all(np.abs(cavity_center - center) < half_extents)

    assert (
        world.collision_manager.collision_detector.check_collision_between_bodies(
            drawer, cabinet
        )
        is None
    )
