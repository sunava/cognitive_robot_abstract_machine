"""
The authored laboratory is reusable independently of its PR2 demonstration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cramera.laboratory_world import (
    LaboratoryBody,
    LaboratoryEnvironment,
    LaboratorySlot,
    MissingLaboratoryBundle,
)
from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    PrismaticConnection,
)

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_world import laboratory_bundle


# %% independent worlds
def test_environment_loads_without_robot(laboratory_bundle: Path) -> None:
    """
    A laboratory alone keeps each manipulable object free and the drawer articulated.
    """
    laboratory = LaboratoryEnvironment(bundle_directory=laboratory_bundle)
    world = laboratory.create_world()
    assert world.get_semantic_annotations_by_type(AbstractRobot) == []
    for item in laboratory.description.objects:
        body = world.get_body_by_name(item.body_name)
        assert isinstance(body.parent_connection, Connection6DoF)
        np.testing.assert_allclose(body.global_pose.to_np(), item.pose.to_np())
    drawer = world.get_body_by_name(LaboratoryBody.DRAWER)
    assert isinstance(drawer.parent_connection, PrismaticConnection)
    initial_pose = drawer.global_pose.to_np().copy()
    drawer.parent_connection.position = 0.1
    np.testing.assert_allclose(
        drawer.global_pose.to_np()[:3, 3], initial_pose[:3, 3] + [0, -0.1, 0]
    )


@pytest.mark.parametrize("robot_type", [PR2, HSRB])
def test_environment_accepts_different_robot_specifications(
    laboratory_bundle: Path, robot_type: type[AbstractRobot]
) -> None:
    """
    The caller chooses the robot and its initial placement using the ordinary API.
    """
    robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, y=-1.0)
    laboratory = LaboratoryEnvironment(bundle_directory=laboratory_bundle)
    world = laboratory.create_world(
        robots=[RobotSpecification(robot_type, odom_T_robot_start=robot_pose)]
    )
    [robot] = world.get_semantic_annotations_by_type(robot_type)
    np.testing.assert_allclose(robot.root.global_pose.to_np(), robot_pose.to_np())
    assert (
        world.get_body_by_name(LaboratoryBody.CLEAR_TUBE).parent_connection.parent
        is world.root
    )


def test_environment_populates_existing_world(
    laboratory_bundle: Path, pr2_world_copy: World
) -> None:
    """
    Adding the laboratory preserves the caller's existing robot instance and pose.
    """
    [robot] = pr2_world_copy.get_semantic_annotations_by_type(PR2)
    initial_pose = robot.root.global_pose.to_np().copy()
    LaboratoryEnvironment(bundle_directory=laboratory_bundle).populate(pr2_world_copy)
    assert pr2_world_copy.get_semantic_annotations_by_type(PR2) == [robot]
    np.testing.assert_allclose(robot.root.global_pose.to_np(), initial_pose)
    assert pr2_world_copy.get_body_by_name(LaboratoryBody.RACK)._world is pr2_world_copy


def test_repeated_creation_does_not_share_movable_objects(
    laboratory_bundle: Path,
) -> None:
    """
    Moving a tube in one experiment does not change a separately created world.
    """
    laboratory = LaboratoryEnvironment(bundle_directory=laboratory_bundle)
    first, second = laboratory.create_world(), laboratory.create_world()
    first_tube = first.get_body_by_name(LaboratoryBody.CLEAR_TUBE)
    second_tube = second.get_body_by_name(LaboratoryBody.CLEAR_TUBE)
    initial_pose = second_tube.global_pose.to_np().copy()
    first_tube.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=2, reference_frame=first.root, child_frame=first_tube
    )
    assert first_tube.id != second_tube.id
    np.testing.assert_allclose(second_tube.global_pose.to_np(), initial_pose)


# %% placement and collision policy
def test_slot_pose_is_bound_to_requested_world(laboratory_bundle: Path) -> None:
    """
    Placement targets carry the world frame required by CRAM plans.
    """
    laboratory = LaboratoryEnvironment(bundle_directory=laboratory_bundle)
    world = laboratory.create_world()
    pose = laboratory.slot_pose(LaboratorySlot.B3, world=world)
    assert pose.reference_frame is world.root
    np.testing.assert_allclose(
        pose.to_np(), laboratory.description.slots[LaboratorySlot.B3].to_np()
    )
    assert laboratory.description.slots[LaboratorySlot.B3].reference_frame is None


@pytest.mark.parametrize("allow_support_contacts", [False, True])
def test_support_collision_exemption_is_explicit(
    laboratory_bundle: Path, allow_support_contacts: bool
) -> None:
    """
    General loading only disables tube-to-rack checks when the caller requests it.
    """
    laboratory = LaboratoryEnvironment(
        bundle_directory=laboratory_bundle,
        allow_support_contacts=allow_support_contacts,
    )
    world = laboratory.create_world()
    rack = world.get_body_by_name(LaboratoryBody.RACK)
    rules = [
        rule
        for rule in world.collision_manager.ignore_collision_rules
        if isinstance(rule, AllowCollisionBetweenGroups) and rack in rule.body_group_b
    ]
    assert len(rules) == int(allow_support_contacts)


# %% portable resources
def test_default_environment_is_independent_of_local_scene_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A clean installation can load the included lab without the user's scene cache.
    """
    monkeypatch.setattr(
        "cramera.laboratory_world.paths.resolve_scene_directory", lambda name: None
    )
    laboratory = LaboratoryEnvironment()
    assert laboratory.bundle_directory == LaboratoryEnvironment.resolve_directory()
    world = laboratory.create_world()
    assert (
        world.get_body_by_name(LaboratoryBody.CLEAR_TUBE).name.name
        == LaboratoryBody.CLEAR_TUBE
    )


def test_explicit_missing_bundle_does_not_fall_back(tmp_path: Path) -> None:
    """
    An incorrect explicit path reports the requested location instead of loading another
    lab.
    """
    with pytest.raises(MissingLaboratoryBundle) as error:
        LaboratoryEnvironment(bundle_directory=tmp_path)
    assert error.value.directory == tmp_path
