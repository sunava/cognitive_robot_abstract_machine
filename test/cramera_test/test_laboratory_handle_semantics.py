"""
Authored laboratory handles remain named semantic frames on their moving parts.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest

from cramera.laboratory_handles import LaboratoryHandle
from cramera.laboratory_plan_world import (
    LaboratoryJointMeasurement,
    LaboratoryPlanWorld,
)
from cramera.laboratory_world import LaboratoryEnvironment
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
)

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_world import laboratory_bundle


# %% semantic handle bodies
def test_environment_exposes_every_authored_handle_semantically(
    laboratory_bundle: Path,
) -> None:
    """
    Every dedicated handle frame is independently selectable by its Handle annotation.
    """
    world = LaboratoryEnvironment(bundle_directory=laboratory_bundle).create_world()
    handles = world.get_semantic_annotations_by_type(Handle)
    assert {str(handle.root.name) for handle in handles} == set(LaboratoryHandle)
    for handle in handles:
        assert isinstance(handle.root.parent_connection, FixedConnection)
        assert len(handle.root.visual) == 0
        assert len(handle.root.collision) == 0


@pytest.mark.parametrize(
    "name",
    [
        LaboratoryHandle.DRAWER,
        LaboratoryHandle.CENTRIFUGE_LID,
        LaboratoryHandle.REFRIGERATOR_DOOR,
    ],
)
def test_measured_articulation_moves_handle_frame_with_its_parent(
    laboratory_bundle: Path, name: LaboratoryHandle
) -> None:
    """
    Detached native-plan measurements update each handle's world transform through its
    joint.
    """
    world = LaboratoryEnvironment(bundle_directory=laboratory_bundle).create_world()
    handle = world.get_body_by_name(name)
    parent = handle.parent_connection.parent
    joint = parent.parent_connection
    assert isinstance(joint, ActiveConnection1DOF)
    initial_world_T_handle = handle.global_pose.to_np().copy()
    parent_T_handle = world.compute_forward_kinematics(parent, handle).to_np().copy()
    measured_position = (
        joint.dof.limits.lower.position + joint.dof.limits.upper.position
    ) / 2

    LaboratoryPlanWorld(
        simulation_time=1,
        joints=(LaboratoryJointMeasurement(str(joint.name), measured_position),),
        poses=(),
    ).apply(world)

    assert joint.position == measured_position
    assert not np.allclose(handle.global_pose.to_np(), initial_world_T_handle)
    np.testing.assert_allclose(
        world.compute_forward_kinematics(parent, handle).to_np(),
        parent_T_handle,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        handle.global_pose.to_np(),
        parent.global_pose.to_np() @ parent_T_handle,
        atol=1e-12,
    )


def test_legacy_laboratory_bundle_without_handle_frames_still_loads(
    laboratory_bundle: Path,
) -> None:
    """
    Older authored assets remain usable without inventing handle transforms.
    """
    description = laboratory_bundle / "environment.urdf"
    tree = ElementTree.parse(description)
    root = tree.getroot()
    for element in list(root):
        if element.tag == "link" and element.attrib["name"] in LaboratoryHandle:
            root.remove(element)
        elif (
            element.tag == "joint"
            and element.find("child").attrib["link"] in LaboratoryHandle
        ):
            root.remove(element)
    tree.write(description)
    world = LaboratoryEnvironment(bundle_directory=laboratory_bundle).create_world()
    assert world.get_semantic_annotations_by_type(Handle) == []
