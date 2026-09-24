"""
Force-driven travel preserves the measured robot and object state.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest

from cramera.laboratory_physics_robot import (
    LaboratoryRobotPhysics,
    RobotBaseJoint,
    RobotPhysicsParameters,
)
from cramera.laboratory_robot_base import RobotBaseDescription

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics


# %% physical base commands
@pytest.fixture
def mobile_robot(robot_physics: LaboratoryRobotPhysics) -> LaboratoryRobotPhysics:
    """
    Enable planar actuation without changing the stationary fixture.
    """
    return LaboratoryRobotPhysics(
        bundle_directory=robot_physics.bundle_directory,
        robot_directory=robot_physics.robot_directory,
        robot_description=robot_physics.robot_description,
        robot_parameters=RobotPhysicsParameters(mobile_base=True),
    )


def test_planar_command_does_not_assign_robot_or_sample_pose(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    The mobile base reaches its target through bounded actuator forces.
    """
    robot_physics = mobile_robot
    before = robot_physics.data.qpos.copy()
    robot_physics.set_joint_targets({RobotBaseJoint.HORIZONTAL: -0.15})
    np.testing.assert_array_equal(robot_physics.data.qpos, before)
    robot_physics.step(2500)
    assert robot_physics.robot_state()[RobotBaseJoint.HORIZONTAL] == pytest.approx(
        -0.15, abs=0.003
    )
    actuator = robot_physics.model.actuator(RobotBaseJoint.HORIZONTAL).id
    assert robot_physics.model.actuator_forcelimited[actuator]
    assert abs(robot_physics.data.actuator_force[actuator]) <= (
        robot_physics.robot_parameters.base_force
    )


def test_base_frames_remain_at_recorded_origin_without_motion(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Existing stationary recipes keep the original initial base placement.
    """
    robot_physics = mobile_robot
    robot_physics.step(1000)
    for joint in RobotBaseJoint:
        assert robot_physics.robot_state()[joint] == pytest.approx(0.0, abs=0.00001)


# %% renderer kinematics
def test_visual_base_joints_match_world_translation_axes(
    mobile_robot: LaboratoryRobotPhysics, tmp_path: Path
) -> None:
    """
    An augmented recording exposes the same planar joints as the simulator.
    """
    robot_physics = mobile_robot
    source = robot_physics.robot_directory / "pr2.urdf"
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes())
    description = RobotBaseDescription(target, robot_physics.base_rotation)
    description.write()
    description.write()
    tree = ElementTree.parse(target)
    for name in RobotBaseJoint:
        joints = tree.findall(f"joint[@name='{name}']")
        assert len(joints) == 1
        joint = joints[0]
        expected = robot_physics.model.jnt_axis[robot_physics.model.joint(name).id]
        np.testing.assert_allclose(
            np.fromstring(joint.find("axis").attrib["xyz"], sep=" "), expected
        )


def test_visual_base_effort_limits_match_physical_actuators(
    mobile_robot: LaboratoryRobotPhysics, tmp_path: Path
) -> None:
    """
    Generated joint effort metadata matches each physical servo's force limit.
    """
    source = mobile_robot.robot_directory / "pr2.urdf"
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes())
    RobotBaseDescription(target, mobile_robot.base_rotation).write()
    tree = ElementTree.parse(target)
    for name in RobotBaseJoint:
        limit = tree.find(f"joint[@name='{name}']/limit")
        actuator = mobile_robot.model.actuator(name).id
        assert (
            float(limit.attrib["effort"])
            == mobile_robot.model.actuator_forcerange[actuator, 1]
        )
