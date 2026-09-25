"""
Native manipulation follows the moving tool with the PR2 optical camera.
"""

from __future__ import annotations

import numpy as np

from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from cramera.workshop_gaze import (
    FollowingPointing,
    ManipulationWithGaze,
    WorkshopMoveToolCenterPoint,
)
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.spatial_types import Pose


# %% alternative native motion mapping
def test_workshop_mapping_retains_tcp_goal_and_tracks_active_tool(
    pr2_world_copy,
) -> None:
    """
    The native mapping adds head gaze without replacing arm tuning or collision rules.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    rules = world.collision_manager.rules.copy()
    context = Context(
        world=world,
        robot=robot,
        alternative_motion_mappings=[WorkshopMoveToolCenterPoint],
    )
    target = Pose.from_xyz_rpy(x=0.5, y=0.2, z=1.1, reference_frame=world.root)
    motion = MoveToolCenterPointMotion(target, Arms.LEFT)
    execute_single(motion, context=context)
    with simulated_robot_advanced:
        assert (
            AlternativeMotion.check_for_alternative(
                context.alternative_motion_mappings, robot, MoveToolCenterPointMotion
            )
            is WorkshopMoveToolCenterPoint
        )
        chart = motion.motion_chart
    assert isinstance(chart, ManipulationWithGaze)
    cartesian, gaze = chart.nodes
    assert isinstance(cartesian, CartesianPose)
    assert cartesian.goal_pose is target
    assert isinstance(gaze, FollowingPointing)
    assert gaze.goal_point.reference_frame is robot.left_arm.end_effector.tool_frame
    assert gaze.root_link is robot.get_torso().root
    assert gaze.tip_link is robot.get_default_camera().root
    assert world.collision_manager.rules == rules


def test_pointing_rebinds_gripper_position_after_arm_movement(pr2_world_copy) -> None:
    """
    Changing a measured arm joint updates the pointing target on the next tick.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    context = Context(world=world, robot=robot)
    motion = WorkshopMoveToolCenterPoint(Pose(reference_frame=world.root), Arms.LEFT)
    execute_single(motion, context=context)
    gaze = motion._motion_chart.nodes[1]
    statechart_context = MotionStatechartContext(world=world)
    gaze.build(statechart_context)
    initial = gaze.root_T_goal_reference_frame.evaluate().copy()
    joint = world.get_connection_by_name(PR2Joint.LEFT_SHOULDER_PAN)
    joint.position += 0.2
    gaze.on_tick(statechart_context)
    expected = world.compute_forward_kinematics_np(
        gaze.root_link, gaze.goal_reference_frame
    )
    np.testing.assert_allclose(gaze.root_T_goal_reference_frame.evaluate(), expected)
    assert not np.allclose(initial, expected)


def test_gaze_does_not_complete_a_tool_motion_before_its_goal(pr2_world_copy) -> None:
    """
    The composite observation stays bound to the native arm task.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    motion = WorkshopMoveToolCenterPoint(Pose(reference_frame=world.root), Arms.LEFT)
    execute_single(motion, context=Context(world=world, robot=robot))
    chart = motion._motion_chart
    assert (
        chart.build_artifacts(MotionStatechartContext(world=world)).observation
        is chart.nodes[0].observation_variable
    )


def test_native_pointing_controls_only_the_two_head_joints(pr2_world_copy) -> None:
    """
    The compiled pointing QP follows the tool without borrowing arm or base motion.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    motion = WorkshopMoveToolCenterPoint(Pose(reference_frame=world.root), Arms.LEFT)
    execute_single(motion, context=Context(world=world, robot=robot))
    gaze = motion._motion_chart.nodes[1]
    chart = MotionStatechart()
    chart.add_node(gaze)
    executor = Executor(MotionStatechartContext(world=world))
    executor.compile(chart)
    initial_error = gaze.error_signal.expression.evaluate().item()
    for _ in range(50):
        executor.tick()
    assert set(executor.qp_controller.active_dofs) == {
        world.get_connection_by_name(name).raw_dof
        for name in (PR2Joint.HEAD_PAN, PR2Joint.HEAD_TILT)
    }
    assert gaze.error_signal.expression.evaluate().item() < initial_error
