"""
Measured front grasp, two-source dosing and supported return to rack A1.
"""

from __future__ import annotations

import numpy as np
import pytest

from cramera.laboratory_liquid import LiquidField
from cramera.laboratory_mixing_program import LaboratoryMixingProgram
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import ProgramField, ProgramState
from cramera.laboratory_pipetting import PipettingAction, PipettingState
from cramera.laboratory_pipetting_program import (
    LaboratoryPipettingProgram,
    PipettingField,
    PipettingPhase,
)
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mixing_adapter import program_world
from .test_laboratory_physics_robot import robot_physics


# %% physical two-source recipe
def test_pipetting_contacts_deliver_both_sources_and_return_to_a1(program_world):
    """
    Actuated fingertips transport the receiver without modifying other contents.
    """
    physics = program_world.physics
    before = physics.liquid.snapshot()
    positions = physics.data.qpos.copy()
    program_world.run_pipetting()
    np.testing.assert_array_equal(physics.data.qpos, positions)
    assert physics.liquid.snapshot() == before
    program = program_world.program
    assert isinstance(program, LaboratoryPipettingProgram)
    for _ in range(140):
        program_world.step(500)
        if program.state != ProgramState.RUNNING:
            break
    result = program.snapshot()
    assert result[ProgramField.STATE] == ProgramState.SUCCEEDED, result
    assert result[ProgramField.GRASP_VERIFIED]
    assert result[ProgramField.RELEASED]
    assert (
        result[ProgramField.DESTINATION_ERROR]
        < program.parameters.destination_tolerance
    )
    assert result[ProgramField.MAXIMUM_GRIPPER_FORCE] < 5.0
    assert result[ProgramField.MAXIMUM_LIFT] >= program.parameters.minimum_lift
    assert result[PipettingField.DELIVERED] == pytest.approx(dict(program.doses))
    after = physics.liquid.snapshot()
    receiver = after[LiquidField.TUBES][LaboratoryBody.CLEAR_TUBE]
    expected_volume = before[LiquidField.TUBES][LaboratoryBody.CLEAR_TUBE][
        LiquidField.VOLUME
    ] + sum(program.doses.values())
    assert receiver[LiquidField.VOLUME] == pytest.approx(expected_volume, abs=1e-7)
    assert after[LiquidField.SPILLED] == pytest.approx(
        before[LiquidField.SPILLED], abs=1e-7
    )
    for key in (
        LaboratoryBody.AMBER_TUBE,
        LaboratoryBody.TEAL_TUBE,
        LaboratoryBody.RESERVE_TUBE,
    ):
        assert (
            after[LiquidField.TUBES][key][LiquidField.VOLUME]
            == before[LiquidField.TUBES][key][LiquidField.VOLUME]
        )
    assert physics.target is None
    assert not np.any(physics.data.xfrc_applied)


def test_pipetting_rejects_a_receiver_away_from_a1_without_reset(program_world):
    """
    A misplaced sample must be returned explicitly instead of being teleported.
    """
    physics = program_world.physics
    program_world.set_target(LaboratoryBody.CLEAR_TUBE, [0.16, -0.15, 1.1])
    program_world.step(1000)
    before = physics.data.qpos.copy()
    with pytest.raises(InvalidPhysicsTarget):
        program_world.run_pipetting()
    np.testing.assert_array_equal(physics.data.qpos, before)


def test_running_pipetting_cannot_be_replaced_by_mixing(program_world):
    """
    The mixing command must not inadvertently resume another controller.
    """
    program_world.run_pipetting()
    program = program_world.program
    with pytest.raises(InvalidPhysicsTarget):
        program_world.run_mixing()
    assert program_world.program is program


def test_paused_pipetting_resumes_without_refilling_or_restoring_poses(program_world):
    """
    A restart command keeps the same physical run and liquid ledger.
    """
    program_world.run_pipetting()
    program_world.step(100)
    program_world.pause_program()
    physics = program_world.physics
    before = physics.data.qpos.copy()
    contents = physics.liquid.snapshot()
    program = program_world.program
    program_world.run_pipetting()
    assert program_world.program is program
    assert program.state == ProgramState.RUNNING
    np.testing.assert_array_equal(physics.data.qpos, before)
    assert physics.liquid.snapshot() == contents


def test_partial_dose_pause_keeps_receiver_and_reservoir_accounting(program_world):
    """
    Stopping mid-dose preserves the undelivered remainder across robot resume.
    """
    program_world.run_pipetting()
    physics = program_world.physics
    for _ in range(600):
        program_world.step(100)
        if physics.pipetting.state == PipettingState.DISPENSING:
            break
    station = physics.pipetting
    assert station.state == PipettingState.DISPENSING, program_world.program.snapshot()
    assert 0 < station.delivered_ml < station.requested_ml
    program_world.pipetting_command(PipettingAction.STOP)
    assert program_world.program.state == ProgramState.PAUSED
    volume = physics.liquid.tubes[LaboratoryBody.CLEAR_TUBE].volume_ml
    delivered = station.delivered_ml
    program_world.step(250)
    assert station.delivered_ml == delivered
    assert physics.liquid.tubes[LaboratoryBody.CLEAR_TUBE].volume_ml == volume
    program_world.run_pipetting()
    assert station.state == PipettingState.DISPENSING
    assert station.delivered_ml == delivered
    for _ in range(50):
        program_world.step(100)
        if station.state == PipettingState.COMPLETE:
            break
    assert station.state == PipettingState.COMPLETE
    assert station.delivered_ml == pytest.approx(station.requested_ml, abs=1e-9)


def test_exhausted_reservoir_rejects_recipe_before_moving(program_world):
    """
    A robot recipe must not start when its declared liquid quantities are absent.
    """
    station = program_world.physics.pipetting
    for reservoir in station.reservoirs.values():
        reservoir.volume_ml = 0.0
    before = program_world.physics.data.qpos.copy()
    with pytest.raises(InvalidPhysicsTarget):
        program_world.run_pipetting()
    np.testing.assert_array_equal(program_world.physics.data.qpos, before)


def test_paused_pipetting_does_not_resume_when_mixing_is_requested(program_world):
    """
    An explicit controller change chooses the requested recipe type.
    """
    program_world.run_pipetting()
    program_world.pause_program()
    program_world.run_mixing()
    assert type(program_world.program) is LaboratoryMixingProgram


def test_active_station_prevents_starting_robot_recipes(program_world):
    """
    A receiver's active pump ownership excludes every robot recipe entry point.
    """
    program_world.physics.pipetting.state = PipettingState.DISPENSING
    for command in (
        program_world.run_program,
        program_world.run_mixing,
        program_world.run_pipetting,
    ):
        with pytest.raises(InvalidPhysicsTarget):
            command()
    assert program_world.program.state == ProgramState.IDLE


def test_replaced_pending_delivery_cannot_resume_robot_recipe(program_world):
    """
    A different operator request must not be mistaken for the robot's dose.
    """
    program_world.run_pipetting()
    program = program_world.program
    program_world.pause_program()
    program.stage_index = next(
        index
        for index, stage in enumerate(program.stages)
        if stage.phase == PipettingPhase.DISPENSING
    )
    program.dose_started = True
    station = program_world.physics.pipetting
    station.key = LaboratoryBody.RESERVE_TUBE
    station.source = program.stage.source
    station.requested_ml = program.doses[program.stage.source]
    station.delivered_ml = station.requested_ml
    with pytest.raises(InvalidPhysicsTarget):
        program_world.run_pipetting()
    assert program.state == ProgramState.PAUSED
