"""Robot program entry respects instrument ownership before changing physics."""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
import pytest

from cramera.laboratory_mixing_program import LaboratoryMixingProgram
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics

# %% robot program entry points


class RobotOperation(StrEnum):
    """Robot programs sharing the centrifuge's physical vessels."""

    TRANSFER = "transfer"
    """Recorded single-glass transfer."""
    MIX = "mix"
    """Two-source mixing recipe."""

    def run(self, adapter: LaboratoryProgramPhysics) -> None:
        """Call the public entry point for the selected robot program."""
        if self == self.TRANSFER:
            adapter.run_program()
        else:
            adapter.run_mixing()


class InstrumentOwnership(StrEnum):
    """Instrument conditions preventing robot control of the same workspace."""

    DRIVE = "drive"
    """Requested rotor motion, including the interval before acceleration."""
    COASTING = "coasting"
    """Measured rotor motion remaining after the drive command reaches zero."""
    FIXTURE = "fixture"
    """A glass retained by a holder while the rotor is stationary."""


@pytest.fixture(params=list(RobotOperation))
def operation(request: pytest.FixtureRequest) -> RobotOperation:
    """Exercise recorded transfer and mixing entry points independently."""
    return request.param


@pytest.fixture
def adapter(
    robot_physics: LaboratoryRobotPhysics, operation: RobotOperation
) -> LaboratoryProgramPhysics:
    """Prepare a real controller without advancing the physical simulation."""
    result = LaboratoryProgramPhysics(robot_physics, robot_physics.robot_directory)
    if operation == RobotOperation.MIX:
        result.program = LaboratoryMixingProgram(robot_physics)
    return result


# %% ownership rejection and safe resume


@pytest.mark.parametrize("state", [ProgramState.IDLE, ProgramState.PAUSED])
@pytest.mark.parametrize("ownership", list(InstrumentOwnership))
def test_owned_instrument_blocks_robot_start_and_resume_without_mutation(
    adapter: LaboratoryProgramPhysics,
    operation: RobotOperation,
    state: ProgramState,
    ownership: InstrumentOwnership,
) -> None:
    """A robot entry point cannot reset a cycle or resume against a secured glass."""
    physics = adapter.physics
    program = adapter.program
    program.state = state
    if ownership == InstrumentOwnership.DRIVE:
        physics.centrifuge.target_rpm = physics.centrifuge.parameters.default_rpm
    elif ownership == InstrumentOwnership.COASTING:
        velocity = physics.model.jnt_dofadr[physics.centrifuge.rotor_joint]
        physics.data.qvel[velocity] = (
            2 * physics.centrifuge.parameters.stopped_rpm * math.tau / 60
        )
    else:
        next(iter(physics.centrifuge.slots.values())).key = LaboratoryBody.CLEAR_TUBE
    instrument = physics.centrifuge.snapshot()
    positions = physics.data.qpos.copy()
    velocities = physics.data.qvel.copy()
    controls = physics.data.ctrl.copy()
    constraints = physics.data.eq_active.copy()
    with pytest.raises(InvalidPhysicsTarget):
        operation.run(adapter)
    assert adapter.program is program
    assert program.state == state
    assert physics.centrifuge.snapshot() == instrument
    np.testing.assert_array_equal(physics.data.qpos, positions)
    np.testing.assert_array_equal(physics.data.qvel, velocities)
    np.testing.assert_array_equal(physics.data.ctrl, controls)
    np.testing.assert_array_equal(physics.data.eq_active, constraints)


@pytest.mark.parametrize("state", [ProgramState.IDLE, ProgramState.PAUSED])
def test_stopped_unlatched_instrument_allows_robot_entry(
    adapter: LaboratoryProgramPhysics, operation: RobotOperation, state: ProgramState
) -> None:
    """A fully released instrument leaves existing robot start and resume usable."""
    adapter.program.state = state
    operation.run(adapter)
    assert adapter.program.state == ProgramState.RUNNING
