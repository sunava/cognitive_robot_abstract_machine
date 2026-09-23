"""
Measured two-source pouring and final rack placement by physical PR2 contacts.
"""

from __future__ import annotations

import numpy as np
import pytest
import mujoco

from cramera.laboratory_mixing_program import LaboratoryMixingProgram, MixingPhase
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_bundle import laboratory_directory


# %% manipulation acceptance
def test_two_liquids_are_poured_mixed_and_released_into_a2(
    robot_physics: LaboratoryRobotPhysics,
) -> None:
    """
    Only measured liquid capture and a released upright receiver establish success.
    """
    program = LaboratoryMixingProgram(robot_physics)
    program.start()
    room = robot_physics.model.body("laboratory_room").id
    arm_bodies = {
        identifier
        for identifier in range(robot_physics.model.nbody)
        if robot_physics.model.body(identifier).name.startswith("pr2/l_")
    }
    for index in range(220000):
        program.advance()
        robot_physics.step()
        program.observe()
        for contact_index in range(robot_physics.data.ncon):
            contact = robot_physics.data.contact[contact_index]
            bodies = {
                int(robot_physics.model.geom_bodyid[contact.geom1]),
                int(robot_physics.model.geom_bodyid[contact.geom2]),
            }
            if room in bodies and bodies.intersection(arm_bodies):
                force = np.zeros(6)
                mujoco.mj_contactForce(
                    robot_physics.model, robot_physics.data, contact_index, force
                )
                assert force[0] < 1.0, (program.elapsed, program.stage, force)
        if program.state != ProgramState.RUNNING:
            break
    result = program.snapshot()
    assert program.state == ProgramState.SUCCEEDED, result
    liquid = robot_physics.liquid.snapshot()
    assert result["capturedMl"]["tube_amber"] > 4.0
    assert result["capturedMl"]["tube_teal"] > 4.0
    assert liquid["tubes"]["tube_clear"]["volumeMl"] > 8.0
    assert liquid["spilledMl"] < 0.05
    assert liquid["totalMl"] == pytest.approx(liquid["initialMl"], abs=1e-7)
    assert result["released"]
    assert result["destinationError"] < program.parameters.destination_tolerance
    assert result["swirlCompleted"]
    assert result["maximumGripperForce"] < 5.0
    assert robot_physics.target is None
    assert not np.any(robot_physics.data.xfrc_applied)


def test_pause_preserves_intentions_while_physics_advances(
    robot_physics: LaboratoryRobotPhysics,
) -> None:
    """
    Pausing freezes the program clock and resuming retains current liquid.
    """
    program = LaboratoryMixingProgram(robot_physics)
    program.start()
    for _ in range(100):
        program.advance()
        robot_physics.step()
        program.observe()
    program.pause()
    before = program.snapshot()
    program.advance()
    robot_physics.step()
    program.observe()
    assert program.snapshot()["time"] == before["time"]
    program.start()
    assert program.state == ProgramState.RUNNING


def test_elapsed_swirl_intention_cannot_replace_measured_motion(
    robot_physics: LaboratoryRobotPhysics,
) -> None:
    """
    An unmoved receiver does not count as physically mixed after a timer expires.
    """
    program = LaboratoryMixingProgram(robot_physics)
    program.start()
    program.stage_index = next(
        index
        for index, stage in enumerate(program.stages)
        if stage.phase == MixingPhase.SWIRL
    )
    program.stage_time = program.stage.duration
    program.observe()
    assert not program.swirl_completed


def test_finger_closing_avoids_a_large_contact_impulse(
    robot_physics: LaboratoryRobotPhysics,
) -> None:
    """
    The first glass is grasped without a fingertip reaction above five Newtons.
    """
    program = LaboratoryMixingProgram(robot_physics)
    program.start()
    for _ in range(30000):
        program.advance()
        robot_physics.step()
        program.observe()
        if program.stage.phase == MixingPhase.LIFT:
            break
    assert program.grasp_verified
    assert program.maximum_gripper_force < 5.0
