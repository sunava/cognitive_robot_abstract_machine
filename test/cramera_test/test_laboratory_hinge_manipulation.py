"""
Robot contact opens and closes physical instrument hinges.
"""

from __future__ import annotations

import numpy as np
import pytest
import mujoco

from cramera.laboratory_hinge_manipulation import (
    HingeInstrument,
    LaboratoryHingeManipulation,
)
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_program import JointReference
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_mixing_program import LaboratoryMixingProgram

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% physical hand-operated articulation
@pytest.mark.parametrize("instrument", list(HingeInstrument))
def test_robot_opens_and_closes_hinge_through_handle_contacts(
    mobile_robot: LaboratoryRobotPhysics, instrument: HingeInstrument
) -> None:
    """
    Each hinge completes both directions with its motor disabled during travel.
    """
    physics = mobile_robot
    for opening in (False, True, False):
        before = physics.data.qpos.copy()
        manipulation = LaboratoryHingeManipulation(physics, instrument, opening)
        manipulation.start()
        np.testing.assert_array_equal(physics.data.qpos, before)
        for _ in range(int(180 / physics.timestep)):
            manipulation.advance()
            physics.step()
            manipulation.observe()
            assert_clear_environment_contacts(physics, manipulation)
            if manipulation.state != ProgramState.RUNNING:
                break
        assert manipulation.state == ProgramState.SUCCEEDED, manipulation.snapshot()
        if manipulation.moved:
            assert manipulation.grasp_verified
            assert manipulation.maximum_handle_force > 0
            assert manipulation.maximum_motor_force == 0
        assert manipulation.angle == pytest.approx(
            manipulation.target_angle, abs=manipulation.parameters.angle_tolerance
        )
        assert not np.any(physics.data.xfrc_applied)


def test_cancellation_restores_original_actuator_limits(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Interrupting a handle action restores glass-safe finger torque bounds.
    """
    physics = mobile_robot
    finger = physics.model.actuator(JointReference.GRIPPER_JOINT).id
    before = physics.model.actuator_forcerange.copy()
    action = LaboratoryHingeManipulation(physics, HingeInstrument.CENTRIFUGE, False)
    action.start()
    assert not np.array_equal(physics.model.actuator_forcerange[finger], before[finger])
    action.cancel()
    np.testing.assert_array_equal(physics.model.actuator_forcerange, before)


@pytest.mark.parametrize(
    ("instrument", "opening"),
    [
        (HingeInstrument.CENTRIFUGE, False),
        (HingeInstrument.CENTRIFUGE, True),
        (HingeInstrument.REFRIGERATOR, True),
    ],
)
def test_robot_clears_surrounding_surfaces_during_handle_manipulation(
    mobile_robot: LaboratoryRobotPhysics, instrument: HingeInstrument, opening: bool
) -> None:
    """
    Approach, hinge movement and withdrawal never push other robot surfaces through
    panels.
    """
    physics = mobile_robot
    if instrument == HingeInstrument.CENTRIFUGE and opening:
        physics.centrifuge_command("close")
        physics.step(4000)
    action = LaboratoryHingeManipulation(physics, instrument, opening)
    action.start()
    for _ in range(int(180 / physics.timestep)):
        action.advance()
        physics.step()
        action.observe()
        assert_clear_environment_contacts(physics, action)
        if action.state != ProgramState.RUNNING:
            break
    assert action.state == ProgramState.SUCCEEDED, action.snapshot()


def test_open_lid_releases_without_dragging_the_cover_backwards(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    The hand releases an opened cover without driving it back towards the chamber.
    """
    physics = mobile_robot
    physics.centrifuge_command("close")
    physics.step(4000)
    action = LaboratoryHingeManipulation(physics, HingeInstrument.CENTRIFUGE, True)
    action.start()
    released_from = None
    for _ in range(int(180 / physics.timestep)):
        action.advance()
        physics.step()
        action.observe()
        if action.phase.value == "release" and released_from is None:
            released_from = action.angle
        if released_from is not None:
            assert action.angle >= released_from - action.parameters.angle_tolerance
        if action.state != ProgramState.RUNNING:
            break
    assert action.state == ProgramState.SUCCEEDED, action.snapshot()


def assert_clear_environment_contacts(
    physics: LaboratoryRobotPhysics, action: LaboratoryHingeManipulation
) -> None:
    """
    Require all non-grasp robot surfaces to remain outside environment solids.
    """
    for contact in physics.data.contact:
        geometries = (contact.geom1, contact.geom2)
        bodies = [
            physics.model.body(physics.model.geom_bodyid[geometry]).name
            for geometry in geometries
        ]
        if not any(body.startswith("pr2/") for body in bodies) or all(
            body.startswith("pr2/") for body in bodies
        ):
            continue
        if any(
            physics.model.geom_type[geometry] == mujoco.mjtGeom.mjGEOM_PLANE
            for geometry in geometries
        ):
            continue
        intended = action.handle in geometries and any(
            body in LaboratoryMixingProgram.FINGERTIPS for body in bodies
        )
        if not intended:
            assert contact.dist >= -0.0002, (action.phase, bodies, contact.dist)
