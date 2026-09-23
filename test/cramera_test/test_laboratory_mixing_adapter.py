"""
Program selection retains resumable state and exclusive manipulation control.
"""

import numpy as np
import pytest

from cramera.laboratory_mixing_program import MixingPhase
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import (
    LaboratoryPhysicsProgram,
    LaboratoryProgramPhysics,
    ProgramState,
)

from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_bundle import laboratory_directory


# %% shared physical scene
@pytest.fixture
def program_world(robot_physics):
    """
    Wrap the local articulated contact world with its selectable programs.
    """
    return LaboratoryProgramPhysics(robot_physics, robot_physics.robot_directory)


def test_fresh_mixing_restores_objects_and_removes_manual_force(program_world):
    """
    A new recipe starts from the original physical arrangement.
    """
    initial = program_world.physics.data.qpos.copy()
    position = program_world.snapshot()["objects"]["tube_clear"][:3]
    position[2] += 0.2
    program_world.set_target("tube_clear", position)
    program_world.step(100)
    assert not np.array_equal(program_world.physics.data.qpos, initial)
    program_world.run_mixing()
    np.testing.assert_array_equal(program_world.physics.data.qpos, initial)
    assert program_world.physics.target is None
    assert program_world.program.state == ProgramState.RUNNING


def test_paused_mixing_resumes_without_restoring_physical_state(program_world):
    """
    Resumption retains the current controller, body poses and liquid ledger.
    """
    program_world.run_mixing()
    program_world.step(100)
    program_world.pause_program()
    program = program_world.program
    positions = program_world.physics.data.qpos.copy()
    liquid = program_world.physics.liquid.snapshot()
    program_world.run_mixing()
    assert program_world.program is program
    assert program_world.program.state == ProgramState.RUNNING
    np.testing.assert_array_equal(program_world.physics.data.qpos, positions)
    assert program_world.physics.liquid.snapshot() == liquid


def test_running_mixing_cannot_be_replaced_by_transfer(program_world):
    """
    An active recipe keeps control until it is paused or completes.
    """
    program_world.run_mixing()
    program = program_world.program
    with pytest.raises(InvalidPhysicsTarget):
        program_world.run_program()
    assert program_world.program is program
    assert program.state == ProgramState.RUNNING


def test_running_transfer_cannot_be_replaced_by_mixing(program_world):
    """
    A transfer keeps control until it is paused or completes.
    """
    program_world.run_program()
    program = program_world.program
    with pytest.raises(InvalidPhysicsTarget):
        program_world.run_mixing()
    assert program_world.program is program
    assert program.state == ProgramState.RUNNING


def test_switching_from_paused_mixing_retains_current_tube_contents(program_world):
    """
    Choosing the original transfer preserves the mixed liquid state.
    """
    program_world.run_mixing()
    program_world.pause_program()
    program_world.fill_liquid("tube_clear", 7.0)
    contents = program_world.physics.liquid.snapshot()["tubes"]
    program_world.run_program()
    assert isinstance(program_world.program, LaboratoryPhysicsProgram)
    after = program_world.physics.liquid.snapshot()["tubes"]
    for key, tube in contents.items():
        assert after[key]["volumeMl"] == tube["volumeMl"]
        assert after[key]["color"] == tube["color"]


def test_paused_tilted_glass_remains_supported_by_the_fingertips(program_world):
    """
    A held tilted glass resists slow numerical contact drift under gravity.
    """
    program_world.run_mixing()
    physics = program_world.physics
    identifier = physics.objects["tube_amber"].body_id
    for _ in range(400):
        program_world.step(100)
        axis = physics.data.xmat[identifier].reshape(3, 3)[:, 2]
        if program_world.program.stage.phase == MixingPhase.POUR and axis[2] < np.cos(
            np.deg2rad(25)
        ):
            break
    assert program_world.program.holding
    assert axis[2] < np.cos(np.deg2rad(25))
    program_world.pause_program()
    program_world.step(500)
    position = physics.data.xpos[identifier].copy()
    axis = physics.data.xmat[identifier].reshape(3, 3)[:, 2].copy()
    program_world.step(5000)
    assert np.linalg.norm(physics.data.xpos[identifier] - position) < 0.002
    final_axis = physics.data.xmat[identifier].reshape(3, 3)[:, 2]
    assert float(axis @ final_axis) > np.cos(np.deg2rad(0.5))
