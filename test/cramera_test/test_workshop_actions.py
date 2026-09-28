"""
Measured blade engagement and native ideal thread coupling.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import MotionNode
from cramera.workshop_actions import (
    EngageThreadNode,
    MoveHeldBodyMotion,
    ReleaseScrewSupportAction,
    ScrewAssembly,
    ScrewEngagementFailed,
    TurnScrewMotion,
)
from cramera.workshop_world import WorkshopWorld
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.open_close import Close
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
    ScrewConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body

from .test_workshop_bundle import workshop_directory
from .test_workshop_world import workshop_bundle


# %% aligned workshop fixture
@pytest.fixture
def aligned_assembly(workshop_bundle: Path):
    """
    Seat a blade in a screw held one millimetre into the authored pilot mouth.
    """
    world = WorkshopWorld(bundle_directory=workshop_bundle).create_world()
    screw = world.get_body_by_name("screw_1")
    driver = world.get_body_by_name("driver")
    workpiece = world.get_body_by_name("workshop_assembly")
    map_T_screw = (
        world.compute_forward_kinematics(world.root, workpiece)
        @ HomogeneousTransformationMatrix.from_xyz_rpy(z=0.139)
    ).to_np()
    screw.parent_connection.origin = HomogeneousTransformationMatrix(
        map_T_screw, reference_frame=world.root
    )
    map_T_driver = (
        HomogeneousTransformationMatrix(map_T_screw)
        @ HomogeneousTransformationMatrix.from_xyz_rpy(z=0.122)
    ).to_np()
    driver.parent_connection.origin = HomogeneousTransformationMatrix(
        map_T_driver, reference_frame=world.root
    )
    return world, ScrewAssembly(screw=screw, driver=driver, workpiece=workpiece)


# %% native ideal thread
def test_thread_installation_preserves_measured_pose_and_collision_rules(
    aligned_assembly,
):
    """
    Engagement changes one connection without teleporting parts or allowing collisions.
    """
    world, assembly = aligned_assembly
    before = assembly.screw.global_pose.to_np().copy()
    rules = world.collision_manager.rules.copy()
    connection = assembly.install(world)
    assert isinstance(connection, ScrewConnection)
    assert connection.parent is assembly.workpiece
    np.testing.assert_allclose(assembly.screw.global_pose.to_np(), before)
    assert world.collision_manager.rules == rules


def test_quarter_turn_advances_screw_by_quarter_pitch(aligned_assembly):
    """
    The world connection itself couples rotation to axial insertion.
    """
    world, assembly = aligned_assembly
    initial = world.compute_forward_kinematics_np(assembly.workpiece, assembly.screw)
    connection = assembly.install(world)
    connection.position = -math.pi / 2
    actual = world.compute_forward_kinematics_np(assembly.workpiece, assembly.screw)
    assert actual[2, 3] == pytest.approx(initial[2, 3] - assembly.pitch / 4)
    np.testing.assert_allclose(actual[:3, 0], [0, -1, 0], atol=1e-12)


def test_final_depth_accounts_for_initial_insertion(aligned_assembly):
    """
    Closing the native joint ends at the configured depth from the pilot mouth.
    """
    world, assembly = aligned_assembly
    connection = assembly.install(world)
    connection.position = connection.dof.limits.lower.position
    actual = world.compute_forward_kinematics_np(assembly.workpiece, assembly.screw)
    assert actual[2, 3] == pytest.approx(assembly.mouth_height - assembly.depth)


def test_misaligned_blade_fails_before_changing_attachment(aligned_assembly):
    """
    An off-centre blade cannot silently engage the ideal thread.
    """
    world, assembly = aligned_assembly
    previous = assembly.screw.parent_connection
    map_T_driver = assembly.driver.global_pose.to_np()
    map_T_driver[0, 3] += 0.003
    assembly.driver.parent_connection.origin = HomogeneousTransformationMatrix(
        map_T_driver, reference_frame=world.root
    )
    with pytest.raises(ScrewEngagementFailed):
        assembly.install(world)
    assert assembly.screw.parent_connection is previous


def test_unaligned_screw_cannot_install_pilot_constraint(aligned_assembly):
    """
    A screw outside the pilot axis remains held instead of snapping into place.
    """
    world, assembly = aligned_assembly
    previous = assembly.screw.parent_connection
    for body in (assembly.screw, assembly.driver):
        map_T_body = body.global_pose.to_np()
        map_T_body[0, 3] += 0.01
        body.parent_connection.origin = HomogeneousTransformationMatrix(
            map_T_body, reference_frame=world.root
        )
    with pytest.raises(ScrewEngagementFailed):
        assembly.install(world)
    assert assembly.screw.parent_connection is previous


def test_symbolic_thread_notification_does_not_change_world(aligned_assembly):
    """
    Expanding the deferred operation leaves the held screw's connection intact.
    """
    world, assembly = aligned_assembly
    previous = assembly.screw.parent_connection
    before = world.state._data.copy()
    EngageThreadNode(assembly=assembly).notify()
    assert assembly.screw.parent_connection is previous
    np.testing.assert_array_equal(world.state._data, before)


@pytest.fixture
def actuated_driver(aligned_assembly):
    """
    Attach the driver to an actuated axial slide and rotary wrist.
    """
    world, assembly = aligned_assembly
    previous = assembly.driver.parent_connection
    map_T_driver = world.compute_forward_kinematics_np(world.root, assembly.driver)
    wrist = Body(name=PrefixedName("test_driver_wrist"))
    with world.modify_world():
        world.add_body(wrist)
        world.add_connection(
            PrismaticConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=wrist,
                axis=Vector3.Z(),
                parent_T_connection_expression=HomogeneousTransformationMatrix(
                    map_T_driver, reference_frame=world.root
                ),
                dof_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=-0.2, velocity=-0.1),
                    upper=DerivativeMap(position=0.2, velocity=0.1),
                ),
            )
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=wrist,
                child=assembly.driver,
                axis=Vector3.Z(),
                dof_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(velocity=-2.0),
                    upper=DerivativeMap(velocity=2.0),
                ),
            )
        )
        world.remove_connection(previous)
        world.delete_orphaned_dofs()
    return world, assembly


@pytest.mark.parametrize("turns", [0.25, 2.0])
def test_native_controller_turns_driver_and_advances_ideal_thread(
    actuated_driver, turns
):
    """
    The native Close controller maintains blade seating across complete revolutions.
    """
    world, assembly = actuated_driver
    connection = assembly.install(world)
    initial_seating = world.compute_forward_kinematics_np(
        assembly.screw, assembly.driver
    )
    chart = MotionStatechart()
    chart.add_node(
        Close(
            tip_link=assembly.driver,
            environment_link=assembly.screw,
            goal_joint_state=-math.tau * turns,
        )
    )
    executor = Executor(MotionStatechartContext(world=world))
    executor.compile(chart)
    for _ in range(
        math.ceil(math.tau * turns / assembly.maximum_angular_velocity / 0.02) + 180
    ):
        executor.tick()
    assert connection.position == pytest.approx(-math.tau * turns, abs=0.005)
    actual = world.compute_forward_kinematics_np(assembly.screw, assembly.driver)
    np.testing.assert_allclose(actual, initial_seating, atol=0.005)


def test_alignment_keeps_supporting_tool_fixed_in_world(pr2_world_copy):
    """
    A torso or base change cannot move the other hand during blade alignment.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    motion = MoveHeldBodyMotion(arm=Arms.RIGHT, support_arm=Arms.LEFT)
    execute_single(motion, context=Context(world=world, robot=robot))
    chart = motion.motion_chart
    statechart = MotionStatechart()
    statechart.add_node(chart)
    chart.expand(MotionStatechartContext(world=world))
    hold = chart.nodes[0].nodes[1]
    assert hold.tip_link is robot.left_arm.end_effector.tool_frame
    assert hold.root_link is world.root
    assert hold.goal_pose.reference_frame is hold.tip_link


def test_support_retreat_keeps_driver_hand_fixed(pr2_world_copy):
    """
    Withdrawing the left hand must not lift the seated blade through torso motion.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    action = ReleaseScrewSupportAction(screw=robot.root)
    execute_single(action, context=Context(world=world, robot=robot))
    plan = action._action_plan.plan
    motions = [node.designator for node in plan.nodes if isinstance(node, MotionNode)]
    [retreat] = [motion for motion in motions if isinstance(motion, MoveHeldBodyMotion)]
    assert retreat.support_arm is Arms.RIGHT


def test_turn_motion_can_be_parsed_before_thread_engagement(pr2_world_copy):
    """
    Sequential plan parsing must defer thread checks until installation has executed.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    motion = TurnScrewMotion(screw=robot.root, driver=robot.root)
    execute_single(motion, context=Context(world=world, robot=robot))
    before = robot.root.parent_connection
    chart = motion.motion_chart
    assert chart.motion is motion
    assert robot.root.parent_connection is before


def test_full_thread_drive_uses_bounded_native_rotation_stages(aligned_assembly):
    """
    Long screws receive multiple native stages instead of exhausting one motion budget.
    """
    world, assembly = aligned_assembly
    connection = assembly.install(world)
    connection.position = -math.pi / 2
    motion = TurnScrewMotion(screw=assembly.screw, driver=assembly.driver)
    targets = motion.joint_targets()
    assert targets[-1] == connection.dof.limits.lower.position
    angles = np.array([connection.position, *targets])
    assert np.all(np.diff(angles) < 0)
    assert np.max(np.abs(np.diff(angles))) <= motion.maximum_turns_per_stage * math.tau


def test_thread_turning_keeps_mobile_base_at_its_station(pr2_world_copy):
    """
    Thread control holds the base in addition to the stationary supporting hand.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    motion = TurnScrewMotion(screw=robot.root, driver=robot.root)
    execute_single(motion, context=Context(world=world, robot=robot))
    chart = motion.resolved_chart(-math.pi / 2)
    [hold] = [
        node
        for node in chart.nodes
        if isinstance(node, CartesianPose) and node.tip_link is robot.root
    ]
    assert hold.root_link is world.root
    assert hold.goal_pose.reference_frame is robot.root


# %% supported driver pickup
def test_thread_turning_retains_the_native_seated_coupling_priority(pr2_world_copy):
    """
    Keep axial blade seating constrained while the helical joint advances.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    motion = TurnScrewMotion(screw=robot.root, driver=robot.root)
    execute_single(motion, context=Context(world=world, robot=robot))
    [drive] = [
        node
        for node in motion.resolved_chart(-math.pi / 2).nodes
        if isinstance(node, Close)
    ]
    assert drive.weight is DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE


def test_driver_pickup_opens_before_approaching_and_retains_lift_orientation(
    pr2_world_copy,
):
    """
    A cradle pickup clears the guides without tilting the held blade.
    """
    from coraplex.datastructures.enums import (
        ApproachDirection,
        VerticalAlignment,
        MovementType,
    )
    from coraplex.robot_plans.motions.gripper import (
        MoveGripperMotion,
        MoveToolCenterPointMotion,
    )
    from cramera.workshop_assembly_demo import DriverGrasp, PickUpDriverAction
    from semantic_digital_twin.datastructures.definitions import GripperState

    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    grasp = DriverGrasp(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        robot.right_arm.end_effector,
    )
    action = PickUpDriverAction(robot.root, Arms.RIGHT, grasp)
    execute_single(action, context=Context(world=world, robot=robot))
    motions = [
        node.designator
        for node in action._action_plan.plan.nodes
        if isinstance(node, MotionNode)
    ]
    assert isinstance(motions[0], MoveGripperMotion)
    assert motions[0].motion is GripperState.OPEN
    [approach, reach, lift] = [
        motion for motion in motions if isinstance(motion, MoveToolCenterPointMotion)
    ]
    assert reach.position_threshold == action.grasp_position_tolerance
    assert lift.movement_type is MovementType.CARTESIAN
    np.testing.assert_allclose(
        lift.target.to_np()[:3, :3], reach.target.to_np()[:3, :3]
    )
    assert lift.target.z > reach.target.z


def test_driver_lift_constrains_the_extraction_to_a_straight_line(pr2_world_copy):
    """
    The shaft clears narrow cradle guides before any lateral transport.
    """
    from cramera.workshop_assembly_demo import DriverLiftMotion
    from giskardpy.motion_statechart.tasks.cartesian_tasks import (
        CartesianPositionStraight,
        CartesianOrientation,
    )
    from semantic_digital_twin.spatial_types import Pose

    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    tool = robot.right_arm.end_effector.tool_frame
    motion = DriverLiftMotion(
        Pose.from_xyz_rpy(z=0.1, reference_frame=tool), Arms.RIGHT
    )
    execute_single(motion, context=Context(world=world, robot=robot))
    goal = motion.motion_chart.nodes[0]
    assert isinstance(goal.nodes[0], CartesianPositionStraight)
    assert isinstance(goal.nodes[1], CartesianOrientation)
    assert goal.nodes[0].tip_link is tool
    assert goal.nodes[0].root_link is world.root
