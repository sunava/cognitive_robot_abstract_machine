"""
Sample transport recovers from low instrument-handle postures.
"""

from __future__ import annotations

import numpy as np
import pytest

from cramera.laboratory_centrifuge_command import CentrifugeAction
from cramera.laboratory_gaze import LaboratoryGaze
from cramera.laboratory_hinge_manipulation import (
    HingeInstrument,
    LaboratoryHingeManipulation,
)
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer, SampleGrasp
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_centrifuge import seat_tube
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% low hand recovery before bench work
def test_upper_transfer_recovers_torso_after_opening_refrigerator(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    A low door grasp does not leave bench transport controlling torso height.
    """
    physics = mobile_robot
    for key, slot in zip(
        (
            LaboratoryBody.CLEAR_TUBE,
            LaboratoryBody.TEAL_TUBE,
            LaboratoryBody.AMBER_TUBE,
            LaboratoryBody.RESERVE_TUBE,
        ),
        physics.centrifuge.slots,
    ):
        physics.fill_liquid(key, 2.0)
        seat_tube(physics, key, slot)
        physics.centrifuge_command(CentrifugeAction.SECURE, slot=slot, key=key)
    gaze = LaboratoryGaze(physics)
    hinge = LaboratoryHingeManipulation(physics, HingeInstrument.REFRIGERATOR, True)
    hinge.start()
    for _ in range(int(150 / physics.timestep)):
        hinge.advance()
        gaze.advance()
        physics.step()
        hinge.observe()
        if hinge.state != ProgramState.RUNNING:
            break
    assert hinge.state == ProgramState.SUCCEEDED, hinge.snapshot()
    physics.centrifuge_command(CentrifugeAction.RELEASE, slot="C1")
    destination = next(
        slot["pose"][:3]
        for slot in physics.scene["laboratory"]["rackSlots"]
        if slot["id"] == "A1"
    )
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=LaboratoryBody.CLEAR_TUBE,
        destination=tuple(destination),
        support_body=LaboratoryBody.RACK,
        grasp=SampleGrasp.TOP,
    )
    torso = "pr2/torso_lift_joint"
    assert transfer.stage.joint_targets == {torso: physics.initial_positions[torso]}
    assert torso not in transfer.arm_names
    transfer.start()
    for _ in range(int(160 / physics.timestep)):
        transfer.advance()
        gaze.advance()
        physics.step()
        transfer.observe()
        if transfer.state != ProgramState.RUNNING:
            break
    assert transfer.state == ProgramState.SUCCEEDED, transfer.snapshot()
    assert LaboratoryBody.CLEAR_TUBE in transfer.grasped_objects
    assert transfer.released
    assert transfer._supported_destination(
        physics.objects[LaboratoryBody.CLEAR_TUBE].body_id,
        transfer.parameters.destination_tolerance,
    )
    assert physics.robot_state()[torso] == pytest.approx(
        physics.initial_positions[torso], abs=transfer.parameters.position_tolerance
    )
    assert not np.any(physics.data.xfrc_applied)
