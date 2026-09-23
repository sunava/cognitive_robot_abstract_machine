"""
Acceptance of the PR2's native laboratory transfer.
"""

from __future__ import annotations

import numpy as np
import pytest

from cramera import paths
from coraplex.datastructures.enums import ApproachDirection, VerticalAlignment
from coraplex.execution_environment import simulated_robot_advanced
from cramera.laboratory_demo import LaboratoryDemo, TubeGrasp
from cramera.laboratory_world import LaboratoryWorld
from semantic_digital_twin.api import BodySpecification, Connection6DoFSpecification
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import Scale


# %% grasp geometry
def test_tube_grasp_lifts_from_the_authored_grip_height(pr2_world_copy) -> None:
    """
    The tool approaches the glass wall above the rack, then extracts vertically.
    """
    world = pr2_world_copy
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    body = BodySpecification.box(
        "grasp_test_tube",
        Scale(0.018, 0.018, 0.15),
        connection_specification=Connection6DoFSpecification(),
    ).spawn(world)
    grasp = TubeGrasp(
        ApproachDirection.RIGHT,
        VerticalAlignment.NoAlignment,
        robot.left_arm.end_effector,
        grasp_height=0.105,
        manipulation_offset=0.18,
    )
    before, contact, lifted = grasp.grasp_pose_sequence(body)
    assert float(contact.z) == pytest.approx(grasp.grasp_height)
    assert float(lifted.z - contact.z) == pytest.approx(grasp.manipulation_offset)
    np.testing.assert_allclose(lifted.to_np()[:2, 3], contact.to_np()[:2, 3])
    assert float(before.z) == pytest.approx(grasp.grasp_height)


# %% complete native manipulation
def test_pr2_transfers_the_glass_to_a3_with_collision_avoidance() -> None:
    """
    A native grasp carries the glass above its neighbours and releases in A3.
    """
    if paths.resolve_scene_directory("precision_lab") is None:
        pytest.skip("Build the precision_lab asset bundle to run its native acceptance")
    laboratory = LaboratoryWorld()
    demonstration = LaboratoryDemo(laboratory=laboratory)
    world = demonstration.build_simulated_world()
    context = demonstration.build_context(world)
    plan = demonstration.build_plan(context)
    observation = demonstration.observe(plan, context)
    with simulated_robot_advanced:
        plan.perform()
    result = observation.validate(plan, laboratory.target_pose)
    assert result.success is True
    assert result.attached_frames > 0
    assert (
        result.maximum_lift
        >= demonstration.grasp_lift - demonstration.position_tolerance
    )
    assert result.position_error <= demonstration.position_tolerance * 2
    assert result.minimum_collision_avoidance_goals > 0
    assert result.released is True
    assert result.other_objects_unchanged is True
