"""
Front-row removal clears each later physical rack pickup.
"""

from __future__ import annotations

import numpy as np
import pytest

from cramera.laboratory_centrifuge_command import CentrifugeAction
from cramera.laboratory_mixing_program import MixingPhase
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer
from cramera.laboratory_world import LaboratoryBody
from cramera.laboratory_workflow import LaboratoryWorkflowProgram

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_centrifuge import seat_tube
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% all authored rack positions
@pytest.mark.parametrize("key", LaboratoryWorkflowProgram.SAMPLE_KEYS)
def test_every_workflow_sample_can_be_lifted_from_its_rack_slot(
    mobile_robot: LaboratoryRobotPhysics, key: LaboratoryBody
) -> None:
    """
    The first lift needs opposing contacts without using manual object forces.
    """
    physics = mobile_robot
    for sample in LaboratoryWorkflowProgram.SAMPLE_KEYS:
        physics.fill_liquid(sample, 0.0)
    holders = []
    for identifier, slot in physics.centrifuge.slots.items():
        if identifier not in holders:
            holders.extend((identifier, slot.opposite))
    for earlier, holder in zip(LaboratoryWorkflowProgram.SAMPLE_KEYS, holders):
        if earlier == key:
            break
        seat_tube(physics, earlier, holder)
        physics.centrifuge_command(CentrifugeAction.SECURE, slot=holder, key=earlier)
    body = physics.objects[key].body_id
    initial = physics.data.xpos[body].copy()
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=key,
        destination=tuple(physics.pipetting.target_position),
        release_at_destination=False,
    )
    first_lift = next(
        index
        for index, stage in enumerate(transfer.stages)
        if stage.phase == MixingPhase.LIFT
    )
    allowed_seconds = (
        sum(stage.duration for stage in transfer.stages[: first_lift + 1])
        + transfer.parameters.stage_timeout
    )
    transfer.start()
    for _ in range(int(allowed_seconds / physics.timestep)):
        transfer.advance()
        physics.step()
        transfer.observe()
        if transfer.state != ProgramState.RUNNING or transfer.stage_index > first_lift:
            break
    assert transfer.state == ProgramState.RUNNING, transfer.snapshot()
    assert transfer.grasp_verified, transfer.snapshot()
    assert transfer.holding, transfer.snapshot()
    assert physics.data.xpos[body, 2] - initial[2] >= transfer.parameters.minimum_lift
    assert physics.target is None
    np.testing.assert_array_equal(physics.data.xfrc_applied, 0)
