"""
Object-specific finger targets remain bounded and preserve enum defaults.
"""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum

import pytest

from coraplex.alternative_motion_mappings.stretch_motion_mapping import (
    StretchMoveGripperMotion,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.motions import gripper
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.pr2 import PR2


# %% explicit targets and default compatibility
def test_explicit_gripper_target_controls_the_joint_goal(immutable_model_world) -> None:
    """
    A partial closure reaches its requested finger positions and tolerance.
    """
    _, robot, context = immutable_model_world
    goal = JointState.from_mapping(
        {
            connection: 0.15
            for connection in robot.left_arm.end_effector.get_joint_state_by_type(
                GripperState.CLOSE
            ).connections
        }
    )
    motion = MoveGripperMotion(
        GripperState.CLOSE,
        Arms.LEFT,
        goal_state=goal,
        joint_position_threshold=0.0001,
    )
    execute_single(motion, context)

    task = motion.motion_chart
    assert isinstance(task, JointPositionList)
    assert list(task.goal_state.items()) == list(goal.items())
    assert task.threshold == motion.joint_position_threshold


@pytest.mark.parametrize("state", [GripperState.OPEN, GripperState.CLOSE])
def test_gripper_without_explicit_target_keeps_named_state(
    immutable_model_world, state: GripperState
) -> None:
    """
    Existing callers keep the robot's named targets and convergence threshold.
    """
    _, robot, context = immutable_model_world
    motion = MoveGripperMotion(state, Arms.LEFT)
    execute_single(motion, context)
    expected = robot.left_arm.end_effector.get_joint_state_by_type(state)

    task = motion.motion_chart
    assert isinstance(task, JointPositionList)
    assert list(task.goal_state.items()) == list(expected.items())
    assert task.threshold == JointPositionList(goal_state=expected).threshold


def test_explicit_gripper_target_keeps_finger_speed_limit(
    immutable_model_world,
) -> None:
    """
    A partial close retains the actual velocity constraint on its target joints.
    """
    _, robot, context = immutable_model_world
    goal = JointState.from_mapping(
        {
            connection: 0.15
            for connection in robot.left_arm.end_effector.get_joint_state_by_type(
                GripperState.CLOSE
            ).connections
        }
    )
    motion = MoveGripperMotion(
        GripperState.CLOSE, Arms.LEFT, goal_state=goal, finger_velocity=0.1
    )
    execute_single(motion, context)

    chart = motion.motion_chart
    assert isinstance(chart, Parallel)
    limit = next(node for node in chart.nodes if isinstance(node, JointVelocityLimit))
    target = next(node for node in chart.nodes if isinstance(node, JointPositionList))
    assert limit.connections == goal.connections
    assert limit.max_velocity == motion.finger_velocity
    assert target.goal_state.target_values == goal.target_values


def test_gripper_target_rebinds_to_copied_context(immutable_model_world) -> None:
    """
    Mirrored planning resolves joint identities in the selected execution world.
    """
    world, robot, _ = immutable_model_world
    goal = JointState.from_mapping(
        {
            connection: 0.15
            for connection in robot.left_arm.end_effector.get_joint_state_by_type(
                GripperState.CLOSE
            ).connections
        }
    )
    copied_world = deepcopy(world)
    copied_robot = copied_world.get_semantic_annotations_by_type(PR2)[0]
    motion = MoveGripperMotion(GripperState.CLOSE, Arms.LEFT, goal_state=goal)
    execute_single(motion, Context(copied_world, copied_robot))

    resolved = motion.motion_chart.goal_state
    expected = {
        connection.child.id: connection
        for connection in copied_robot.left_arm.end_effector.active_connections
    }
    assert resolved.target_values == goal.target_values
    assert all(
        connection is expected[connection.child.id]
        for connection in resolved.connections
    )
    assert all(
        connection is not original
        for connection, original in zip(resolved.connections, goal.connections)
    )


# %% rejected targets
class MalformedGripperTarget(StrEnum):
    """
    Invalid layouts of requested gripper joints and target values.
    """

    EMPTY = "empty"
    """No joints or positions requested."""
    MISSING_VALUE = "missing_value"
    """A selected joint has no matching position."""
    DUPLICATE_JOINT = "duplicate_joint"
    """One joint is assigned two positions."""


@pytest.mark.parametrize("position", [float("nan"), float("inf"), -float("inf")])
def test_gripper_rejects_nonfinite_targets(
    immutable_model_world, position: float
) -> None:
    """
    Invalid numeric positions cannot enter the controller's equality constraints.
    """
    _, robot, context = immutable_model_world
    goal = JointState.from_mapping(
        {
            connection: position
            for connection in robot.left_arm.end_effector.active_connections
        }
    )
    motion = MoveGripperMotion(GripperState.CLOSE, Arms.LEFT, goal_state=goal)
    execute_single(motion, context)

    with pytest.raises(gripper.InvalidGripperGoal):
        motion.motion_chart


@pytest.mark.parametrize("above_upper_limit", [False, True])
def test_gripper_rejects_out_of_range_targets(
    immutable_model_world, above_upper_limit: bool
) -> None:
    """
    Explicit targets are rejected instead of silently clamped to full closure.
    """
    _, robot, context = immutable_model_world
    connection = robot.left_arm.end_effector.active_connections[0]
    limit = (
        connection.dof.limits.upper.position
        if above_upper_limit
        else connection.dof.limits.lower.position
    )
    goal = JointState.from_mapping(
        {connection: limit + (0.1 if above_upper_limit else -0.1)}
    )
    motion = MoveGripperMotion(GripperState.CLOSE, Arms.LEFT, goal_state=goal)
    execute_single(motion, context)

    with pytest.raises(gripper.InvalidGripperGoal):
        motion.motion_chart


def test_gripper_rejects_joints_of_the_other_hand(immutable_model_world) -> None:
    """
    An explicit goal cannot move an end effector other than the selected hand.
    """
    _, robot, context = immutable_model_world
    goal = robot.right_arm.end_effector.get_joint_state_by_type(GripperState.CLOSE)
    motion = MoveGripperMotion(GripperState.CLOSE, Arms.LEFT, goal_state=goal)
    execute_single(motion, context)

    with pytest.raises(gripper.InvalidGripperGoal):
        motion.motion_chart


@pytest.mark.parametrize("shape", list(MalformedGripperTarget))
def test_gripper_rejects_incomplete_or_ambiguous_targets(
    immutable_model_world, shape: MalformedGripperTarget
) -> None:
    """
    Every requested joint needs one unambiguous target value.
    """
    _, robot, context = immutable_model_world
    connection = robot.left_arm.end_effector.active_connections[0]
    goals = {
        MalformedGripperTarget.EMPTY: JointState(),
        MalformedGripperTarget.MISSING_VALUE: JointState(connections=[connection]),
        MalformedGripperTarget.DUPLICATE_JOINT: JointState(
            connections=[connection, connection], target_values=[0.1, 0.2]
        ),
    }
    motion = MoveGripperMotion(GripperState.CLOSE, Arms.LEFT, goal_state=goals[shape])
    execute_single(motion, context)

    with pytest.raises(gripper.InvalidGripperGoal):
        motion.motion_chart


@pytest.mark.parametrize("threshold", [0.0, -0.1, float("nan"), float("inf")])
def test_gripper_rejects_invalid_position_thresholds(
    immutable_model_world, threshold: float
) -> None:
    """
    A convergence threshold must have a finite positive value.
    """
    _, _, context = immutable_model_world
    motion = MoveGripperMotion(
        GripperState.CLOSE, Arms.LEFT, joint_position_threshold=threshold
    )
    execute_single(motion, context)

    with pytest.raises(gripper.InvalidGripperGoal):
        motion.motion_chart


# %% executed partial closure
def test_alternative_gripper_mapping_preserves_explicit_target(
    immutable_stretch_apartment_world,
) -> None:
    """A robot-specific mapping must honor a partial closure and its speed limit."""
    _, robot, context = immutable_stretch_apartment_world
    goal = JointState.from_mapping(
        {
            connection: 0.03
            for connection in robot.get_arms()[0].end_effector.active_connections
        }
    )
    motion = StretchMoveGripperMotion(
        GripperState.CLOSE,
        Arms.LEFT,
        goal_state=goal,
        joint_position_threshold=0.0001,
        finger_velocity=0.01,
    )
    execute_single(motion, context)

    chart = motion.motion_chart
    assert isinstance(chart, Parallel)
    target = next(node for node in chart.nodes if isinstance(node, JointPositionList))
    assert target.goal_state.target_values == goal.target_values
    assert target.threshold == motion.joint_position_threshold
    limit = next(node for node in chart.nodes if isinstance(node, JointVelocityLimit))
    assert limit.max_velocity == motion.finger_velocity


def test_gripper_executes_partial_closure(pr2_world_copy) -> None:
    """
    Native execution stops at the requested joint state above fully closed.
    """
    world = pr2_world_copy
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    end_effector = robot.left_arm.end_effector
    end_effector.get_joint_state_by_type(GripperState.OPEN).apply_to(world)
    goal = JointState.from_mapping(
        {
            connection: 0.15
            for connection in end_effector.get_joint_state_by_type(
                GripperState.CLOSE
            ).connections
        }
    )
    motion = MoveGripperMotion(
        GripperState.CLOSE,
        Arms.LEFT,
        goal_state=goal,
        joint_position_threshold=0.0001,
        finger_velocity=0.1,
    )
    node = execute_single(motion, Context(world, robot))

    with simulated_robot:
        node.plan.perform()

    for connection, target in goal.items():
        assert connection.position == pytest.approx(
            target, abs=motion.joint_position_threshold
        )
