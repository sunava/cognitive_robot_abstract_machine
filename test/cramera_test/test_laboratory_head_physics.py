"""
The PR2's head follows bounded physical actuator commands.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest

from cramera.laboratory_physics_robot import (
    LaboratoryRobotPhysics,
    RobotHeadJoint,
)

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics


# %% measured head articulation
@pytest.mark.parametrize("name", RobotHeadJoint)
def test_head_uses_description_position_and_effort_limits(
    robot_physics: LaboratoryRobotPhysics, name: RobotHeadJoint
) -> None:
    """
    Each independently controlled head coordinate preserves the URDF limits.
    """
    assert name in robot_physics.robot_joints
    joint = robot_physics.robot_joints[name]
    source = ElementTree.parse(robot_physics.robot_description)
    limit = source.find(f"joint[@name='{name.removeprefix('pr2/')}']/limit")
    assert joint.lower == float(limit.attrib["lower"])
    assert joint.upper == float(limit.attrib["upper"])
    assert joint.actuator_index is not None
    assert robot_physics.model.actuator_forcelimited[joint.actuator_index]
    maximum = float(limit.attrib["effort"])
    np.testing.assert_array_equal(
        robot_physics.model.actuator_forcerange[joint.actuator_index],
        (-maximum, maximum),
    )


def test_head_moves_through_servos_and_publishes_measured_positions(
    robot_physics: LaboratoryRobotPhysics,
) -> None:
    """
    Changing the gaze command leaves poses intact until physics advances.
    """
    before = robot_physics.data.qpos.copy()
    targets = {RobotHeadJoint.PAN: 0.35, RobotHeadJoint.TILT: 0.7}
    robot_physics.set_joint_targets(targets)
    np.testing.assert_array_equal(robot_physics.data.qpos, before)
    for name in RobotHeadJoint:
        joint = robot_physics.robot_joints[name]
        assert robot_physics.robot_state()[name] == before[joint.position_index]
        assert robot_physics.robot_state()[name] != targets[name]

    robot_physics.step(2500)

    for name, target in targets.items():
        joint = robot_physics.robot_joints[name]
        actual = robot_physics.data.qpos[joint.position_index]
        assert actual == pytest.approx(target, abs=0.002)
        assert robot_physics.snapshot()["frames"][name] == actual
        assert abs(robot_physics.data.actuator_force[joint.actuator_index]) <= (
            robot_physics.model.actuator_forcerange[joint.actuator_index, 1]
        )


# %% command rejection
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), True])
def test_invalid_head_command_preserves_the_whole_command(
    robot_physics: LaboratoryRobotPhysics, invalid: float | bool
) -> None:
    """
    A non-finite or non-numeric head angle cannot partially apply a command.
    """
    before = robot_physics.data.ctrl.copy()
    targets = robot_physics.joint_targets.copy()
    with pytest.raises(ValueError):
        robot_physics.set_joint_targets(
            {RobotHeadJoint.PAN: 0.35, RobotHeadJoint.TILT: invalid}
        )
    np.testing.assert_array_equal(robot_physics.data.ctrl, before)
    assert robot_physics.joint_targets == targets


@pytest.mark.parametrize("name", RobotHeadJoint)
def test_out_of_range_head_command_is_rejected(
    robot_physics: LaboratoryRobotPhysics, name: RobotHeadJoint
) -> None:
    """
    A head command beyond its physical hard stops leaves all targets unchanged.
    """
    before = robot_physics.data.ctrl.copy()
    joint = robot_physics.robot_joints[name]
    with pytest.raises(ValueError):
        robot_physics.set_joint_targets({name: joint.upper + 0.1})
    np.testing.assert_array_equal(robot_physics.data.ctrl, before)
    with pytest.raises(ValueError):
        robot_physics.set_joint_targets({name: joint.lower - 0.1})
    np.testing.assert_array_equal(robot_physics.data.ctrl, before)
