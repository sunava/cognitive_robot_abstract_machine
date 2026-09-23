"""
Rack support and quiet finger opening establish a completed placement.
"""

from __future__ import annotations

from dataclasses import replace
from math import ceil

import mujoco
import numpy as np
import pytest

from cramera.laboratory_mixing_program import LaboratoryMixingProgram, MixingPhase
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics


# %% physical rack seating
@pytest.fixture
def seated_program(robot_physics: LaboratoryRobotPhysics) -> LaboratoryMixingProgram:
    """
    Observe the originally spawned teal tube after it settles in its own slot.
    """
    program = LaboratoryMixingProgram(robot_physics)
    robot_physics.step(700)
    program.stage_index = next(
        index
        for index, stage in enumerate(program.stages[:-1])
        if stage.phase == MixingPhase.PLACE
        and stage.object_key == LaboratoryBody.TEAL_TUBE
        and program.stages[index + 1].phase == MixingPhase.RELEASE
    )
    program.grasp_verified = True
    program.state = ProgramState.RUNNING
    program._enter_stage()
    return program


def observe_for(program: LaboratoryMixingProgram, duration: float) -> None:
    """
    Repeat measured contact observations without moving the resting test state.
    """
    for _ in range(ceil(duration / program.physics.timestep)):
        program.observe()


def test_supported_seating_releases_before_the_nominal_descent_finishes(
    seated_program: LaboratoryMixingProgram,
) -> None:
    """
    An upright stationary tube on its rack base no longer needs opposing fingers.
    """
    program = seated_program
    observe_for(program, program.parameters.lost_contact_timeout / 2)
    assert program.state == ProgramState.RUNNING
    assert program.stage.phase == MixingPhase.RELEASE
    assert not program.stage.closed


def test_seating_requires_persistent_support_before_release(
    seated_program: LaboratoryMixingProgram,
) -> None:
    """
    One frame of valid support cannot complete a placement.
    """
    program = seated_program
    program.observe()
    assert program.stage.phase == MixingPhase.PLACE


def test_final_descent_requests_a_bounded_rack_contact(
    seated_program: LaboratoryMixingProgram,
) -> None:
    """
    The hand transfers weight by descending slightly through the nominal support plane.
    """
    program = seated_program
    target = program._stage_position()
    nominal = np.asarray(program.stage.position)
    np.testing.assert_array_equal(target[:2], nominal[:2])
    assert nominal[2] - program.parameters.position_tolerance < target[2] < nominal[2]


def test_transport_waypoint_keeps_its_authored_clearance(
    seated_program: LaboratoryMixingProgram,
) -> None:
    """
    The contact-seeking descent is restricted to the final rack insertion.
    """
    program = seated_program
    program.stage_index -= 1
    program._enter_stage()
    np.testing.assert_array_equal(program._stage_position(), program.stage.position)


def test_teal_return_and_placement_restore_the_original_front_grasp_heading(
    seated_program: LaboratoryMixingProgram,
) -> None:
    """
    A crowded rack approach retains the same hand heading while replacing its tube.
    """
    stages = [
        stage
        for stage in seated_program.stages
        if stage.object_key == LaboratoryBody.TEAL_TUBE
    ]
    approach = next(stage for stage in stages if stage.phase == MixingPhase.APPROACH)
    final_return = [stage for stage in stages if stage.phase == MixingPhase.RETURN][-1]
    placement = [stage for stage in stages if stage.phase == MixingPhase.PLACE]
    for stage in [final_return, *placement]:
        np.testing.assert_allclose(stage.rotation, approach.rotation, atol=1e-12)


@pytest.mark.parametrize(
    "condition", ["missing_base", "translating", "rotating", "wrong_slot", "tilted"]
)
def test_invalid_seating_preserves_the_lost_grasp_failure(
    seated_program: LaboratoryMixingProgram,
    condition: str,
) -> None:
    """
    Side contact, motion, a wrong position or excessive tilt cannot substitute for
    seating.
    """
    program = seated_program
    physics = program.physics
    body = physics.objects[LaboratoryBody.TEAL_TUBE].body_id
    joint = int(physics.model.body_jntadr[body])
    position = int(physics.model.jnt_qposadr[joint])
    velocity = int(physics.model.jnt_dofadr[joint])
    if condition == "missing_base":
        rack = physics.model.body(LaboratoryBody.RACK).id
        base = min(
            (
                index
                for index in range(physics.model.ngeom)
                if physics.model.geom_bodyid[index] == rack
            ),
            key=lambda index: physics.data.geom_xpos[index, 2],
        )
        physics.model.geom_contype[base] = 0
        physics.model.geom_conaffinity[base] = 0
    elif condition == "translating":
        physics.data.qvel[velocity] = 0.05
    elif condition == "rotating":
        physics.data.qvel[velocity + 3] = 0.3
    elif condition == "wrong_slot":
        physics.data.qpos[position] += program.parameters.position_tolerance * 2
    elif condition == "tilted":
        angle = program.parameters.upright_tolerance * 2
        physics.data.qpos[position + 3 : position + 7] = [
            np.cos(angle / 2),
            np.sin(angle / 2),
            0,
            0,
        ]
    mujoco.mj_forward(physics.model, physics.data)
    observe_for(program, program.parameters.lost_contact_timeout + physics.timestep)
    assert program.state == ProgramState.FAILED
    assert program.stage.phase == MixingPhase.PLACE


def test_intermediate_placement_waypoint_cannot_release_from_support(
    seated_program: LaboratoryMixingProgram,
) -> None:
    """
    Only the final descent directly followed by finger opening accepts rack support.
    """
    program = seated_program
    program.stages[program.stage_index + 1] = replace(
        program.stages[program.stage_index + 1],
        phase=MixingPhase.PLACE,
    )
    observe_for(
        program, program.parameters.lost_contact_timeout + program.physics.timestep
    )
    assert program.state == ProgramState.FAILED
    assert program.stage.phase == MixingPhase.PLACE


def test_elapsed_placement_time_cannot_replace_measured_support(
    seated_program: LaboratoryMixingProgram,
) -> None:
    """
    An aligned glass without rack contact cannot be released by the stage timer.
    """
    program = seated_program
    physics = program.physics
    body = physics.objects[LaboratoryBody.TEAL_TUBE].body_id
    joint = int(physics.model.body_jntadr[body])
    position = int(physics.model.jnt_qposadr[joint])
    physics.data.qpos[position : position + 3] = program.stage.position
    physics.data.qpos[position + 3 : position + 7] = [1, 0, 0, 0]
    physics.model.geom_contype[:] = 0
    physics.model.geom_conaffinity[:] = 0
    mujoco.mj_forward(physics.model, physics.data)
    program.stage_time = program.stage.duration
    program.observe()
    assert program.state == ProgramState.RUNNING
    assert program.stage.phase == MixingPhase.PLACE


def test_interrupted_support_restarts_the_seating_confirmation(
    seated_program: LaboratoryMixingProgram,
) -> None:
    """
    Two short resting intervals separated by motion do not add up to a seated state.
    """
    program = seated_program
    physics = program.physics
    body = physics.objects[LaboratoryBody.TEAL_TUBE].body_id
    joint = int(physics.model.body_jntadr[body])
    velocity = int(physics.model.jnt_dofadr[joint])
    observe_for(program, program.parameters.seating_confirmation * 0.6)
    physics.data.qvel[velocity] = program.parameters.seating_linear_speed * 2
    mujoco.mj_forward(physics.model, physics.data)
    program.observe()
    physics.data.qvel[velocity] = 0
    mujoco.mj_forward(physics.model, physics.data)
    observe_for(program, program.parameters.seating_confirmation * 0.6)
    assert program.stage.phase == MixingPhase.PLACE
    observe_for(program, program.parameters.seating_confirmation)
    assert program.stage.phase == MixingPhase.RELEASE


# %% stable hand pose while opening
def test_release_opens_fingers_without_realigning_a_calibrated_hand(
    seated_program: LaboratoryMixingProgram,
) -> None:
    """
    Opening cannot pull the hand toward a different nominal grasp calibration.
    """
    program = seated_program
    program.stage_index += 1
    program._enter_stage()
    start_position, start_rotation = program._virtual_pose()
    for _ in range(300):
        program.advance()
        program.physics.step()
        program.observe()
    position, rotation = program._virtual_pose()
    assert np.linalg.norm(position - start_position) < 0.003
    assert np.linalg.norm((rotation * start_rotation.inv()).as_rotvec()) < 0.02


def test_withdrawal_keeps_the_attained_height_and_orientation(
    seated_program: LaboratoryMixingProgram,
) -> None:
    """
    A released gripper retreats relative to its hand pose before the next approach.
    """
    program = seated_program
    program.stage_index += 2
    program._enter_stage()
    start_position, start_rotation = program._virtual_pose()
    expected_offset = np.asarray(program.stage.position) - np.asarray(
        program.stages[program.stage_index - 1].position
    )
    np.testing.assert_allclose(
        program._stage_position(), start_position + expected_offset
    )
    for _ in range(300):
        program.advance()
        program.physics.step()
        program.observe()
    position, rotation = program._virtual_pose()
    assert abs(position[2] - start_position[2]) < 0.003
    assert np.linalg.norm((rotation * start_rotation.inv()).as_rotvec()) < 0.02
