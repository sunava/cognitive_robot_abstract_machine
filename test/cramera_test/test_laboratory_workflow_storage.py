"""
Check each cooler destination with earlier batch samples already stored.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_refrigerator_command import (
    RefrigeratorAction,
    RefrigeratorField,
)
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer, SampleGrasp
from cramera.laboratory_workflow import LaboratoryWorkflowProgram, WorkflowParameters

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_scale import place_on_pan


# %% supported storage checkpoints
@pytest.mark.parametrize(
    "sample_index", range(len(LaboratoryWorkflowProgram.SAMPLE_KEYS))
)
def test_scale_sample_reaches_its_cooler_slot_without_displacing_stored_samples(
    mobile_robot: LaboratoryRobotPhysics, sample_index: int
) -> None:
    """
    A real front grasp carries a weighed vial past previously stored samples.
    """
    physics = mobile_robot
    recipe = WorkflowParameters()
    keys = LaboratoryWorkflowProgram.SAMPLE_KEYS
    slots = physics.refrigerator.metadata[RefrigeratorField.SLOTS]
    physics.refrigerator_command(RefrigeratorAction.OPEN)
    for index, key in enumerate(keys[:sample_index]):
        physics.fill_liquid(key, 2 * recipe.dose_ml)
        body = physics.objects[key].body_id
        joint = physics.model.body_jntadr[body]
        coordinate = physics.model.jnt_qposadr[joint]
        velocity = physics.model.jnt_dofadr[joint]
        physics.data.qpos[coordinate : coordinate + 3] = np.asarray(
            slots[index][RefrigeratorField.POSITION]
        ) + np.array([0.0, 0.0, 0.001])
        physics.data.qpos[coordinate + 3 : coordinate + 7] = [1, 0, 0, 0]
        physics.data.qvel[velocity : velocity + 6] = 0
    key = keys[sample_index]
    physics.fill_liquid(key, 2 * recipe.dose_ml)
    place_on_pan(physics, key)
    mujoco.mj_forward(physics.model, physics.data)
    physics.step(5000)
    assert physics.refrigerator.door_open
    for index, stored in enumerate(keys[:sample_index]):
        assert (
            physics.refrigerator.occupants()[slots[index][RefrigeratorField.ID]]
            == stored
        )
    slot = slots[sample_index]
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=key,
        destination=tuple(slot[RefrigeratorField.POSITION]),
        support_body=physics.refrigerator.metadata[RefrigeratorField.BODY],
        grasp=SampleGrasp.FRONT,
    )
    transfer.start()
    for _ in range(int(recipe.transfer_timeout / physics.timestep)):
        before = physics.data.qpos.copy()
        transfer.advance()
        assert np.array_equal(physics.data.qpos, before)
        physics.step()
        transfer.observe()
        if transfer.state != ProgramState.RUNNING:
            break
    assert transfer.state == ProgramState.SUCCEEDED, transfer.snapshot()
    physics.step(1000)
    assert transfer.released
    for index, stored in enumerate(keys[: sample_index + 1]):
        assert (
            physics.refrigerator.occupants()[slots[index][RefrigeratorField.ID]]
            == stored
        )
    assert physics.liquid.tubes[key].volume_ml == pytest.approx(2 * recipe.dose_ml)
    assert physics.target is None
    assert not np.any(physics.data.xfrc_applied)
