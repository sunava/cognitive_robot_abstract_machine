"""
Build the current CRAM transfer without executing robot motion.
"""

import pytest
import numpy as np

from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.motions.gripper import (
    MoveToolCenterPointMotion,
    MoveGripperMotion,
)
from cramera_bridge.laboratory.grasping import (
    PadContactMotion,
    PadContactNode,
    VesselReachAction,
)
from cramera_bridge.laboratory.transfer import VesselTransferDemo
from cramera_bridge.exceptions import TaskFailed
from semantic_digital_twin.spatial_types import Pose
from giskardpy.motion_statechart.data_types import LifeCycleValues


@pytest.fixture(scope="module")
def transfer_plan(laboratory_bundle):
    """
    :return: The transfer and its expanded native plan at the authored start state.
    """
    demo = VesselTransferDemo.at_the_bench(laboratory_bundle)
    context = demo.build_context(demo.build_world())
    plan = demo.build_plan(context)
    plan.root.notify()
    return demo, plan


def test_the_transfer_picks_and_places_the_same_vessel(transfer_plan):
    demo, plan = transfer_plan
    [pickup] = plan.get_nodes_by_designator_type(PickUpAction)
    [placing] = plan.get_nodes_by_designator_type(PlaceAction)
    assert pickup.designator.grasp.graspable.root is demo.observation.body
    assert placing.designator.object_designator.root is demo.observation.body
    assert pickup.designator.arm is demo.observation.context.robot.left_arm
    assert not demo.observation.context.robot.drive.has_hardware_interface


def test_contact_measurement_is_an_execution_boundary_after_the_approach(transfer_plan):
    demo, plan = transfer_plan
    contacts = [
        node for node in plan.root.descendants if isinstance(node, PadContactNode)
    ]
    assert len(contacts) == 1
    assert contacts[0].contains_execution_boundary
    [closing] = plan.get_nodes_by_designator_type(PadContactMotion)
    assert closing.designator.object_designator is demo.observation.body

    def leaves(node):
        if not node.children:
            yield node
        for child in node.children:
            yield from leaves(child)

    ordered = [
        node
        for node in leaves(plan.root)
        if isinstance(node, (MotionNode, PadContactNode, ReAttachNode))
    ]
    position = ordered.index(contacts[0])
    first, opening, final = ordered[position - 3 : position]
    boundary, close, attach = ordered[position : position + 3]
    assert isinstance(first.designator, MoveToolCenterPointMotion)
    assert isinstance(opening.designator, MoveGripperMotion)
    assert isinstance(final.designator, MoveToolCenterPointMotion)
    assert boundary is contacts[0]
    assert close is closing
    assert isinstance(attach, ReAttachNode)
    assert attach.body is demo.observation.body
    assert (
        attach.new_parent
        is demo.observation.context.robot.left_arm.end_effector.tool_frame
    )


def test_the_grasp_candidate_preserves_the_calibrated_tool_pose(transfer_plan):
    demo, plan = transfer_plan
    [reach] = plan.get_nodes_by_designator_type(VesselReachAction)
    motion_nodes = [
        node
        for node in reach.descendants
        if hasattr(node, "designator")
        and isinstance(node.designator, MoveToolCenterPointMotion)
    ]
    approach, grasp = [node.designator.target for node in motion_nodes]
    expected = Pose.from_xyz_quaternion(
        0,
        0,
        demo.laboratory.vessels.grasp_height,
        *demo.tool_orientation.to_np(),
        reference_frame=demo.observation.body,
    )
    np.testing.assert_allclose(grasp.to_np(), expected.to_np(), atol=1e-12)
    expected_approach = expected.to_np().copy()
    expected_approach[:3, 3] -= (
        demo.laboratory.vessels.outer_radius + demo.approach_margin
    ) * expected_approach[:3, 0]
    np.testing.assert_allclose(approach.to_np(), expected_approach, atol=1e-12)
    [pickup] = plan.get_nodes_by_designator_type(PickUpAction)
    poses = pickup.designator.grasp_pose_sequence(
        pickup.designator.grasp.grasp_pose,
        pickup.designator.arm.end_effector,
        pickup.designator.grasp,
    )
    np.testing.assert_allclose(
        poses.retreat.to_np()[:3, 3] - poses.grasp.to_np()[:3, 3],
        [0, 0, demo.lift],
        atol=1e-12,
    )


def test_an_unexecuted_plan_is_not_accepted(transfer_plan):
    demo, _ = transfer_plan
    with pytest.raises(TaskFailed):
        demo.require_success(demo.observation.context)


@pytest.mark.parametrize(
    "status", [LifeCycleValues.NOT_STARTED, LifeCycleValues.FAILED]
)
def test_one_unfinished_motion_refuses_action_success(
    transfer_plan, monkeypatch, status
):
    demo, plan = transfer_plan
    motions = [node for node in plan.nodes if isinstance(node, MotionNode)]
    assert motions
    for motion in motions:
        monkeypatch.setattr(motion, "status", LifeCycleValues.SUCCEEDED)
    demo.require_success(demo.observation.context)
    monkeypatch.setattr(motions[-1], "status", status)
    with pytest.raises(TaskFailed):
        demo.require_success(demo.observation.context)


def test_actions_without_motion_descendants_refuse_success(transfer_plan):
    demo, _ = transfer_plan
    unexpanded = VesselTransferDemo.at_the_bench(demo.laboratory.directory)
    unexpanded.build_plan(demo.observation.context)
    with pytest.raises(TaskFailed):
        unexpanded.require_success(demo.observation.context)
