"""
Executable swirling actions and constrained container tracking.
"""

import numpy as np
import pytest

from coraplex.datastructures.enums import Arms
from coraplex.datastructures.swirling import SwirlProfile, SwirlTrajectory
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.actions.composite.swirling import SwirlingAction
from coraplex.robot_plans.motions.swirling import (
    ContainerNotHeld,
    SwirlContainerMotion,
    SwirlPoseTask,
)
from coraplex.view_manager import ViewManager
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.tasks.feature_functions import AngleGoal
from krrood.symbolic_math.symbolic_math import FloatVariable

from .test_tool_based_actions import _attach_box_to_gripper, tool_action_world

# %% Action expansion


@pytest.fixture
def held_swirl_action(tool_action_world):
    """
    A glass held with a measured, nonzero offset from the gripper.
    """
    world, robot, context, container, _ = tool_action_world
    held = _attach_box_to_gripper(
        world, robot, "held_swirl_container", (0.04, 0.04, 0.2), -0.08
    )
    action = SwirlingAction(container=held, arm=Arms.RIGHT)
    sequential([action], context)
    return action


def _motion(action):
    action.expand()
    motions = [
        node.designator
        for node in action.plan.all_nodes
        if isinstance(node, MotionNode)
        and isinstance(node.designator, SwirlContainerMotion)
    ]
    assert len(motions) == 1
    return motions[0]


def test_swirl_action_expands_to_continuous_motion(held_swirl_action):
    motion = _motion(held_swirl_action)
    assert motion.container == held_swirl_action.container
    assert motion.arm == held_swirl_action.arm
    assert motion.profile == held_swirl_action.profile


def test_swirl_action_rejects_container_outside_gripper(tool_action_world):
    world, robot, context, container, _ = tool_action_world
    action = SwirlingAction(container=container, arm=Arms.RIGHT)
    sequential([action], context)
    with pytest.raises(ContainerNotHeld):
        action.expand()


def test_swirl_motion_has_real_giskard_tilt_inequality(held_swirl_action):
    motion = _motion(held_swirl_action)
    chart = motion._motion_chart
    tilt = next(node for node in chart.nodes if isinstance(node, AngleGoal))
    assert tilt.lower_angle == 0.0
    assert tilt.upper_angle == motion.maximum_tilt
    context = MotionStatechartContext(world=held_swirl_action.world)
    artifacts = tilt.build(context)
    assert len(artifacts.constraints.inequality_constraints) == 1


def test_swirl_tracking_preserves_measured_grasp_and_completes_after_duration(
    held_swirl_action,
):
    motion = _motion(held_swirl_action)
    task = next(
        node for node in motion._motion_chart.nodes if isinstance(node, SwirlPoseTask)
    )
    context = MotionStatechartContext(world=held_swirl_action.world)
    context.control_cycle_variable = FloatVariable("swirl_control_cycle")
    context.float_variable_data.register_expression(context.control_cycle_variable)
    context.float_variable_data.set_value(context.control_cycle_variable, 0.0)
    artifacts = task.build(context)
    task.on_start(context)
    assert len(artifacts.constraints.equality_constraints) == 6
    duration = task.profile.duration
    elapsed = duration * 0.5
    context.float_variable_data.set_value(
        context.control_cycle_variable,
        elapsed / context.qp_controller_config.control_dt,
    )
    task.on_tick(context)
    world = held_swirl_action.world
    tool_frame = ViewManager.get_end_effector_view(
        motion.arm, held_swirl_action.robot
    ).tool_frame
    container_T_tool = world.compute_forward_kinematics_np(motion.container, tool_frame)
    expected = (
        SwirlTrajectory(
            world.compute_forward_kinematics_np(world.root, motion.container),
            task.container_P_pivot.to_np()[:3],
            task.profile,
        ).pose_at(elapsed)
        @ container_T_tool
    )
    actual = task.root_T_goal.evaluate()
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert not bool(artifacts.observation.evaluate())


def test_swirl_tracking_pause_does_not_advance_phase(held_swirl_action):
    motion = _motion(held_swirl_action)
    task = next(
        node for node in motion._motion_chart.nodes if isinstance(node, SwirlPoseTask)
    )
    context = MotionStatechartContext(world=held_swirl_action.world)
    context.control_cycle_variable = FloatVariable("swirl_pause_cycle")
    context.float_variable_data.register_expression(context.control_cycle_variable)
    context.float_variable_data.set_value(context.control_cycle_variable, 0.0)
    task.build(context)
    task.on_start(context)
    context.float_variable_data.set_value(context.control_cycle_variable, 4.0)
    task.on_tick(context)
    before = task.root_T_goal.evaluate()
    task.on_pause(context)
    context.float_variable_data.set_value(context.control_cycle_variable, 1004.0)
    task.on_unpause(context)
    task.on_tick(context)
    np.testing.assert_allclose(task.root_T_goal.evaluate(), before, atol=1e-12)


# %% Tracking progress


def test_swirl_clock_waits_when_gripper_lags_behind(held_swirl_action):
    """
    An obstructed gripper cannot complete a swirl by waiting out the timer.
    """
    motion = _motion(held_swirl_action)
    task = next(
        node for node in motion._motion_chart.nodes if isinstance(node, SwirlPoseTask)
    )
    context = MotionStatechartContext(world=held_swirl_action.world)
    context.control_cycle_variable = FloatVariable("swirl_obstructed_cycle")
    context.float_variable_data.register_expression(context.control_cycle_variable)
    context.float_variable_data.set_value(context.control_cycle_variable, 0.0)
    artifacts = task.build(context)
    task.on_start(context)
    midpoint_cycle = (
        0.5 * task.profile.duration / context.qp_controller_config.control_dt
    )
    context.float_variable_data.set_value(
        context.control_cycle_variable, midpoint_cycle
    )
    task.on_tick(context)
    elapsed_before = context.float_variable_data.get_value(task.elapsed)
    context.float_variable_data.set_value(
        context.control_cycle_variable, midpoint_cycle + 1.0
    )
    task.on_tick(context)
    assert context.float_variable_data.get_value(task.elapsed) == elapsed_before
    assert not bool(artifacts.observation.evaluate())
