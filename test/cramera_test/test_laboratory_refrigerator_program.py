"""
Robot demonstrations preserve samples already stored in the refrigerator.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_refrigerator_command import (
    RefrigeratorAction,
    RefrigeratorField,
)
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics


# %% robot ownership and stored sample preservation
@pytest.mark.parametrize("operation", ["transfer", "mixing", "pipetting"])
def test_robot_program_cannot_restore_stored_sample_to_bench(
    robot_physics: LaboratoryRobotPhysics, operation: str
) -> None:
    """
    A new program cannot reset a physically supported refrigerator sample.
    """
    physics = robot_physics
    adapter = LaboratoryProgramPhysics(physics, physics.robot_directory)
    slot = physics.refrigerator.metadata[RefrigeratorField.SLOTS][0]
    body = physics.objects[LaboratoryBody.CLEAR_TUBE].body_id
    coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[body]]
    physics.data.qpos[coordinate : coordinate + 3] = np.asarray(
        slot[RefrigeratorField.POSITION]
    ) + [0, 0, 0.002]
    mujoco.mj_forward(physics.model, physics.data)
    physics.step(1500)
    before = physics.data.qpos.copy()
    program = adapter.program
    with pytest.raises(InvalidPhysicsTarget):
        if operation == "transfer":
            adapter.run_program()
        elif operation == "mixing":
            adapter.run_mixing()
        else:
            adapter.run_pipetting()
    np.testing.assert_array_equal(physics.data.qpos, before)
    assert adapter.program is program


def test_running_robot_preserves_door_ownership(
    robot_physics: LaboratoryRobotPhysics,
) -> None:
    """
    Direct program-adapter calls cannot move a door during active execution.
    """
    adapter = LaboratoryProgramPhysics(robot_physics, robot_physics.robot_directory)
    adapter.program.state = ProgramState.RUNNING
    before = robot_physics.data.qpos.copy()
    with pytest.raises(InvalidPhysicsTarget):
        adapter.refrigerator_command(RefrigeratorAction.OPEN)
    np.testing.assert_array_equal(robot_physics.data.qpos, before)
    adapter.program.state = ProgramState.PAUSED
    adapter.refrigerator_command(RefrigeratorAction.OPEN)
    assert (
        robot_physics.refrigerator.target_angle == robot_physics.refrigerator.open_angle
    )
