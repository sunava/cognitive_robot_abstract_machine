"""
Resolve partial PR2 closure from the reached object's collision geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from cramera.pr2_gripper import Pr2GraspGeometryError, Pr2GripperContact
from semantic_digital_twin.api import BodySpecification, Connection6DoFSpecification
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale


# %% reached grasp fixture
@pytest.fixture
def reached_grasp(pr2_world_copy) -> Pr2GripperContact:
    """
    Put an eighteen-millimetre collision body between the existing PR2 pads.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    end_effector = robot.left_arm.end_effector
    body = BodySpecification.box(
        "fragile_grasp_target",
        Scale(0.04, 0.018, 0.018),
        connection_specification=Connection6DoFSpecification(),
    ).spawn(
        world,
        parent_T_self=end_effector.tool_frame.global_pose.to_homogeneous_matrix(),
    )
    end_effector.get_joint_state_by_type(GripperState.OPEN).apply_to(world)
    return Pr2GripperContact(world, end_effector, body)


@dataclass(eq=False)
class StatePublicationCounter(StateChangeCallback):
    """
    Count any changes published to the live grasp world.
    """

    publications: int = field(default=0, init=False)
    """
    State notifications received after registration.
    """

    def on_state_change(self, **kwargs) -> None:
        """
        Record notifications without changing the robot state.
        """
        self.publications += 1


# %% geometry and isolation
def test_goal_stops_before_both_fingers_penetrate(reached_grasp) -> None:
    """
    The native joint goal leaves each pad at the configured object clearance.
    """
    goal = reached_grasp.goal_state()
    assert len(goal.connections) == 1
    assert goal.target_values[0] > goal.connections[0].raw_dof.limits.lower.position
    goal.apply_to(reached_grasp.world)
    distances = reached_grasp.distances()
    assert distances.minimum == pytest.approx(reached_grasp.clearance, abs=1e-7)
    assert distances.maximum == pytest.approx(reached_grasp.clearance, abs=1e-7)
    assert reached_grasp.aperture() >= reached_grasp.body.collision[0].scale.y


def test_search_preserves_live_world_and_publishes_no_probes(reached_grasp) -> None:
    """
    Only a private world copy sees trial closures during calibration.
    """
    before = reached_grasp.world.state._data.copy()
    counter = StatePublicationCounter(_world=reached_grasp.world)
    goal = reached_grasp.goal_state()
    np.testing.assert_array_equal(reached_grasp.world.state._data, before)
    assert counter.publications == 0
    assert goal.connections[0] is reached_grasp.connection


def test_object_outside_gripper_cannot_be_grasped(reached_grasp) -> None:
    """
    A distant object supplies no valid closing contact bracket.
    """
    reached_grasp.body.parent_connection.origin = (
        reached_grasp.body.global_pose.to_homogeneous_matrix()
        @ HomogeneousTransformationMatrix.from_xyz_rpy(y=0.3)
    )
    with pytest.raises(Pr2GraspGeometryError):
        reached_grasp.goal_state()


def test_off_center_object_cannot_attach_to_only_one_pad(reached_grasp) -> None:
    """
    The selected opening must put both fingers near the object.
    """
    reached_grasp.body.parent_connection.origin = (
        reached_grasp.body.global_pose.to_homogeneous_matrix()
        @ HomogeneousTransformationMatrix.from_xyz_rpy(y=0.006)
    )
    with pytest.raises(Pr2GraspGeometryError):
        reached_grasp.goal_state()


def test_object_wider_than_open_gripper_is_rejected(reached_grasp) -> None:
    """
    An object cannot be accepted when fully open fingers already penetrate it.
    """
    wide_body = BodySpecification.box(
        "oversize_grasp_target",
        Scale(0.04, 0.2, 0.018),
        connection_specification=Connection6DoFSpecification(),
    ).spawn(
        reached_grasp.world,
        parent_T_self=(
            reached_grasp.end_effector.tool_frame.global_pose.to_homogeneous_matrix()
        ),
    )
    reached_grasp.body = wide_body
    with pytest.raises(Pr2GraspGeometryError):
        reached_grasp.goal_state()
