"""
Reject incomplete or inaccurate laboratory execution evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pytest

from coraplex.datastructures.dataclasses import Context, MotionToleranceConfig
from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    TaskStatus,
    VerticalAlignment,
)
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.motions.placement import MovePlacementMotion, PlacementStage
from cramera.laboratory_demo import (
    LaboratoryObservation,
    LaboratoryPlaceAction,
    LaboratoryResult,
    TubeGrasp,
)
from cramera.laboratory_world import LaboratoryBody
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.api import BodySpecification, Connection6DoFSpecification
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale


# %% execution evidence
class UnselectedObject(StrEnum):
    """
    Scene objects whose poses must survive the transfer unchanged.
    """

    AMBER = "tube_amber"
    TEAL = "tube_teal"
    STOPPER = "stopper"


@dataclass
class TransferEvidence:
    """
    A native action tree and observations ready for outcome validation.
    """

    observation: LaboratoryObservation
    """
    Measurements supplied independently of action completion statuses.
    """

    plan: Plan
    """
    Actual pickup and placement nodes with recorded completion statuses.
    """

    target: Pose
    """
    Requested final object pose in the world frame.
    """

    def validate(self) -> LaboratoryResult:
        """
        Evaluate the current action and geometric evidence.
        """
        return self.observation.validate(self.plan, self.target)


@pytest.fixture
def transfer_evidence(pr2_world_copy: World) -> TransferEvidence:
    """
    Provide successful evidence without external laboratory assets or execution.
    """
    world = pr2_world_copy
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    bodies = [
        BodySpecification.box(
            name,
            Scale(0.018, 0.018, 0.15),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=3.0 + index, z=0.906
            ),
            connection_specification=Connection6DoFSpecification(),
        ).spawn(world)
        for index, name in enumerate([LaboratoryBody.CLEAR_TUBE, *UnselectedObject])
    ]
    context = Context(
        world=world,
        robot=robot,
        motion_tolerances=MotionToleranceConfig(
            default_tcp_position_threshold=0.0005,
            tool_orientation_threshold=0.003,
        ),
    )
    observation = LaboratoryObservation(
        context=context,
        body=bodies[0],
        position_tolerance=context.motion_tolerances.default_tcp_position_threshold,
        required_lift=0.16,
    )
    target = Pose.from_xyz_rpy(x=3.12, z=0.906, reference_frame=world.root)
    grasp = TubeGrasp(
        ApproachDirection.RIGHT,
        VerticalAlignment.NoAlignment,
        robot.left_arm.end_effector,
    )
    plan = sequential(
        [
            PickUpAction(bodies[0], Arms.LEFT, grasp),
            PlaceAction(bodies[0], target, Arms.LEFT),
        ],
        context=context,
    ).plan
    for action in (PickUpAction, PlaceAction):
        plan.get_nodes_by_designator_type(action)[0].status = TaskStatus.SUCCEEDED
    observation.attached_frames = 1
    observation.maximum_lift = observation.required_lift
    observation.avoidance_counts = [1]
    observation.minimum_collision_distance = observation.position_tolerance
    bodies[0].parent_connection.origin = target.to_homogeneous_matrix()
    world.collision_manager.update_collision_matrix()
    return TransferEvidence(observation, plan, target)


# %% independent acceptance conditions
def test_complete_transfer_evidence_is_accepted(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    The positive control contains both action completion and measured evidence.
    """
    assert transfer_evidence.validate().success is True


def test_action_success_does_not_accept_the_wrong_slot(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    Completed actions cannot conceal millimetres of insertion error.
    """
    observation = transfer_evidence.observation
    misplaced = transfer_evidence.target.to_np().copy()
    misplaced[0, 3] += observation.position_tolerance * 3
    observation.body.parent_connection.origin = HomogeneousTransformationMatrix(
        misplaced, reference_frame=observation.context.world.root
    )
    result = transfer_evidence.validate()
    assert result.position_error == pytest.approx(observation.position_tolerance * 3)
    assert result.success is False


def test_action_success_does_not_accept_a_tilted_tube(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    A correct tube-bottom position does not excuse an incorrect insertion angle.
    """
    observation = transfer_evidence.observation
    angular_error = observation.context.motion_tolerances.tool_orientation_threshold * 3
    tilted = (
        transfer_evidence.target.to_homogeneous_matrix()
        @ HomogeneousTransformationMatrix.from_xyz_rpy(roll=angular_error)
    )
    observation.body.parent_connection.origin = tilted
    result = transfer_evidence.validate()
    assert result.orientation_error == pytest.approx(angular_error)
    assert result.success is False


def test_final_pose_does_not_replace_grasp_attachment(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    Reaching the destination alone is not evidence of a robot grasp.
    """
    transfer_evidence.observation.attached_frames = 0
    assert transfer_evidence.validate().success is False


def test_transfer_requires_clearance_above_neighbouring_tubes(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    A successful action tree must also contain sufficient measured extraction.
    """
    observation = transfer_evidence.observation
    observation.maximum_lift = (
        observation.required_lift - observation.position_tolerance * 2
    )
    assert transfer_evidence.validate().success is False


def test_moving_an_unselected_tube_rejects_the_transfer(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    The selected tube cannot reach its target by displacing a neighbour.
    """
    observation = transfer_evidence.observation
    neighbour = observation.context.world.get_body_by_name(UnselectedObject.AMBER)
    displaced = neighbour.global_transform.to_np().copy()
    displaced[0, 3] += observation.position_tolerance
    neighbour.parent_connection.origin = HomogeneousTransformationMatrix(
        displaced, reference_frame=observation.context.world.root
    )
    result = transfer_evidence.validate()
    assert result.other_objects_unchanged is False
    assert result.success is False


def test_collision_avoidance_must_remain_active(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    One observed tick without collision avoidance rejects the execution.
    """
    transfer_evidence.observation.avoidance_counts.append(0)
    assert transfer_evidence.validate().success is False


def test_geometric_success_does_not_conceal_action_failure(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    Placement must complete successfully even when the object is near its goal.
    """
    transfer_evidence.plan.get_nodes_by_designator_type(PlaceAction)[
        0
    ].status = TaskStatus.FAILED
    assert transfer_evidence.validate().success is False


def test_transfer_requires_releasing_the_tube(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    An object at its destination must no longer follow the gripper.
    """
    observation = transfer_evidence.observation
    world = observation.context.world
    with world.modify_world():
        world.move_branch_with_fixed_connection(
            observation.body, observation.context.robot.left_arm.end_effector.tool_frame
        )
    result = transfer_evidence.validate()
    assert result.released is False
    assert result.success is False


def test_observed_penetration_rejects_otherwise_complete_evidence(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    Enabled collision avoidance is insufficient when checked geometry penetrates.
    """
    observation = transfer_evidence.observation
    observation.minimum_collision_distance = -observation.position_tolerance
    assert transfer_evidence.validate().success is False


# %% evidence accumulated during motion
def test_unattached_motion_does_not_count_as_robot_extraction(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    A displaced world object cannot supply the grasp's lift measurement.
    """
    observation = transfer_evidence.observation
    observation.maximum_lift = 0.0
    raised = observation.body.global_transform.to_np().copy()
    raised[2, 3] += observation.required_lift
    observation.body.parent_connection.origin = HomogeneousTransformationMatrix(
        raised, reference_frame=observation.context.world.root
    )
    statechart = MotionStatechart()
    statechart.add_node(ExternalCollisionAvoidance(robot=observation.context.robot))
    observation.on_motion_tick(statechart)
    assert observation.maximum_lift == 0.0


def test_restoring_a_neighbour_does_not_erase_its_motion(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    Unselected objects must remain stationary throughout the execution.
    """
    observation = transfer_evidence.observation
    neighbour = observation.context.world.get_body_by_name(UnselectedObject.AMBER)
    original = neighbour.global_transform
    displaced = original.to_np().copy()
    displaced[0, 3] += observation.position_tolerance
    neighbour.parent_connection.origin = HomogeneousTransformationMatrix(
        displaced, reference_frame=observation.context.world.root
    )
    statechart = MotionStatechart()
    statechart.add_node(ExternalCollisionAvoidance(robot=observation.context.robot))
    observation.on_motion_tick(statechart)
    neighbour.parent_connection.origin = original
    observation.on_motion_tick(statechart)
    result = transfer_evidence.validate()
    assert result.other_objects_unchanged is False
    assert result.success is False


# %% insertion precision
def test_coarse_approach_preserves_precise_release(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    Free-space arrival may settle early while rack insertion keeps its tolerance.
    """
    observation = transfer_evidence.observation
    action = LaboratoryPlaceAction(
        observation.body, transfer_evidence.target, Arms.LEFT
    )
    plan = sequential([action], context=observation.context).plan
    bound_action = plan.get_nodes_by_designator_type(LaboratoryPlaceAction)[
        0
    ].designator
    expanded = bound_action._action_plan
    motions = {
        node.designator.stage: node.designator
        for node in expanded.descendants
        if isinstance(node, MotionNode)
        and isinstance(node.designator, MovePlacementMotion)
    }
    assert (
        motions[PlacementStage.APPROACH].position_threshold
        == action.approach_position_threshold
    )
    assert motions[PlacementStage.RELEASE].position_threshold is None
    assert (
        motions[PlacementStage.RELEASE].resolved_position_threshold()
        == observation.position_tolerance
    )
    assert motions[PlacementStage.RETRACT].position_threshold is None
