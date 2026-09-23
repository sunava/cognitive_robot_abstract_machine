"""
Require geometric evidence that the fingers hold the glass without penetration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from cramera.pr2_gripper import FingerContactDistances, Pr2GripperContact
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart

from .test_laboratory_acceptance import TransferEvidence, transfer_evidence


# %% geometric observations
@dataclass
class FingerGeometryEvidence:
    """
    Supply independent fingertip measurements to execution acceptance.
    """

    transfer: TransferEvidence
    """Otherwise complete native manipulation evidence."""
    contact: Pr2GripperContact
    """
    Geometry query whose returned distances are controlled by each test.
    """

    distances: FingerContactDistances
    """Current signed pad clearances, in metres."""
    aperture: float = 0.0185
    """
    Current separation of the pad mesh surfaces, in metres.
    """

    def sample(self) -> None:
        """
        Observe one geometric state while the tube is attached.
        """
        self.transfer.observation._observe_gripper_contact()


@pytest.fixture
def finger_geometry(
    transfer_evidence: TransferEvidence, monkeypatch: pytest.MonkeyPatch
) -> FingerGeometryEvidence:
    """
    Add explicit contact checking without executing robot motions.
    """
    observation = transfer_evidence.observation
    contact = Pr2GripperContact(
        observation.context.world,
        observation.context.robot.left_arm.end_effector,
        observation.body,
    )
    evidence = FingerGeometryEvidence(
        transfer_evidence,
        contact,
        FingerContactDistances(contact.clearance, contact.clearance),
    )
    monkeypatch.setattr(contact, "distances", lambda: evidence.distances)
    monkeypatch.setattr(contact, "aperture", lambda: evidence.aperture)
    observation.gripper_contact = contact
    return evidence


# %% grasp acceptance
def test_configured_contact_requires_an_attached_measurement(
    finger_geometry: FingerGeometryEvidence,
) -> None:
    """
    Action completion cannot replace missing fingertip measurements.
    """
    assert finger_geometry.transfer.validate().success is False


def test_legacy_observation_without_contact_preserves_acceptance(
    transfer_evidence: TransferEvidence,
) -> None:
    """
    Callers without an explicit contact query keep their existing contract.
    """
    result = transfer_evidence.validate()
    assert result.success is True
    assert result.gripper_geometry_verified is False
    assert result.minimum_finger_clearance is None


def test_safe_close_fingers_supply_geometric_grasp_evidence(
    finger_geometry: FingerGeometryEvidence,
) -> None:
    """
    Both close fingertips and a nonpenetrating carry satisfy the extra checks.
    """
    finger_geometry.sample()
    result = finger_geometry.transfer.validate()
    assert result.success is True
    assert result.gripper_geometry_verified is True
    assert result.minimum_finger_clearance == finger_geometry.distances.minimum
    assert result.first_grasp_maximum_clearance == finger_geometry.distances.maximum
    assert result.first_grasp_aperture == finger_geometry.aperture


def test_penetrating_fingers_reject_otherwise_complete_transfer(
    finger_geometry: FingerGeometryEvidence,
) -> None:
    """
    Allowed grasp collisions cannot hide fingers inside the glass.
    """
    finger_geometry.distances = FingerContactDistances(
        -finger_geometry.contact.clearance, finger_geometry.contact.clearance
    )
    finger_geometry.sample()
    result = finger_geometry.transfer.validate()
    assert result.minimum_finger_clearance == finger_geometry.distances.minimum
    assert result.gripper_geometry_verified is False
    assert result.success is False


def test_penetration_during_carry_survives_later_safe_measurements(
    finger_geometry: FingerGeometryEvidence,
) -> None:
    """
    A correct initial grasp does not erase a later finger penetration.
    """
    finger_geometry.sample()
    initial_distances = finger_geometry.distances
    penetration = -finger_geometry.contact.clearance
    finger_geometry.distances = FingerContactDistances(
        penetration, finger_geometry.contact.clearance
    )
    finger_geometry.sample()
    finger_geometry.distances = initial_distances
    finger_geometry.sample()
    result = finger_geometry.transfer.validate()
    assert result.minimum_finger_clearance == penetration
    assert result.success is False


def test_one_distant_finger_cannot_count_as_a_grasp(
    finger_geometry: FingerGeometryEvidence,
) -> None:
    """
    Both pads must initially surround the tube near their stopping clearance.
    """
    finger_geometry.distances = FingerContactDistances(
        finger_geometry.contact.clearance,
        finger_geometry.contact.clearance
        + finger_geometry.contact.maximum_contact_difference * 2,
    )
    finger_geometry.sample()
    assert finger_geometry.transfer.validate().success is False


def test_opening_before_detach_does_not_reject_a_safe_grasp(
    finger_geometry: FingerGeometryEvidence,
) -> None:
    """
    The release motion may open the fingers while attachment is still present.
    """
    finger_geometry.sample()
    first_clearance = finger_geometry.distances.maximum
    first_aperture = finger_geometry.aperture
    finger_geometry.distances = FingerContactDistances(0.02, 0.02)
    finger_geometry.aperture += 0.04
    finger_geometry.sample()
    result = finger_geometry.transfer.validate()
    assert result.first_grasp_maximum_clearance == first_clearance
    assert result.first_grasp_aperture == first_aperture
    assert result.success is True


@pytest.mark.parametrize("invalid_distance", [math.nan, math.inf, -math.inf])
def test_nonfinite_carry_measurement_rejects_grasp(
    finger_geometry: FingerGeometryEvidence, invalid_distance: float
) -> None:
    """
    Missing or invalid geometry cannot silently preserve a previous safe sample.
    """
    finger_geometry.sample()
    finger_geometry.distances = FingerContactDistances(
        invalid_distance, finger_geometry.contact.clearance
    )
    finger_geometry.sample()
    assert finger_geometry.transfer.validate().success is False


# %% motion callback integration
def test_unattached_tick_does_not_measure_grasp_contact(
    finger_geometry: FingerGeometryEvidence,
) -> None:
    """
    An approaching gripper is not counted as a held glass.
    """
    observation = finger_geometry.transfer.observation
    statechart = MotionStatechart()
    statechart.add_node(ExternalCollisionAvoidance(robot=observation.context.robot))
    observation.on_motion_tick(statechart)
    assert observation.first_grasp_aperture is None


def test_attached_tick_records_finger_geometry(
    finger_geometry: FingerGeometryEvidence,
) -> None:
    """
    Executed attached motion ticks feed the explicit fingertip check.
    """
    observation = finger_geometry.transfer.observation
    world = observation.context.world
    with world.modify_world():
        world.move_branch_with_fixed_connection(
            observation.body, observation.context.robot.left_arm.end_effector.tool_frame
        )
    statechart = MotionStatechart()
    statechart.add_node(ExternalCollisionAvoidance(robot=observation.context.robot))
    observation.on_motion_tick(statechart)
    assert observation.first_grasp_aperture == finger_geometry.aperture
    assert observation.minimum_finger_clearance == finger_geometry.distances.minimum
