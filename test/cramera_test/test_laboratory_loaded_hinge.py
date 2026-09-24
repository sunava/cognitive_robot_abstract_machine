"""
Robot clearance after a complete loaded centrifuge cycle.
"""

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
from cramera.laboratory_workflow import LaboratoryWorkflowProgram, WorkflowParameters

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_centrifuge import seat_tube
from .test_laboratory_hinge_manipulation import assert_clear_environment_contacts
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% loaded instrument clearance
@pytest.mark.parametrize("final_sample_by_robot", [False, True])
def test_robot_releases_loaded_lid_after_timed_cycle(
    mobile_robot: LaboratoryRobotPhysics,
    final_sample_by_robot: bool,
) -> None:
    """
    A tracked robot closes, spins, opens and fully clears four secured samples.
    """
    physics = mobile_robot
    parameters = WorkflowParameters()
    keys = LaboratoryWorkflowProgram.SAMPLE_KEYS
    if final_sample_by_robot:
        keys = keys[1:] + keys[:1]
    slots = tuple(physics.centrifuge.slots.values())
    gaze = LaboratoryGaze(physics)
    for index, (key, slot) in enumerate(zip(keys, slots)):
        physics.fill_liquid(key, 2 * parameters.dose_ml)
        if final_sample_by_robot and index == len(keys) - 1:
            transfer = LaboratorySampleTransfer(
                physics=physics,
                object_key=key,
                destination=tuple(physics.centrifuge.slot_position(slot.identifier)),
                support_body=physics.scene["laboratory"]["centrifuge"]["rotorBody"],
                grasp=SampleGrasp.TOP,
            )
            transfer.start()
            for _ in range(int(180 / physics.timestep)):
                transfer.advance()
                gaze.advance()
                physics.step()
                transfer.observe()
                if transfer.state != ProgramState.RUNNING:
                    break
            assert transfer.state == ProgramState.SUCCEEDED, transfer.snapshot()
        else:
            seat_tube(physics, key, slot.identifier)
        physics.centrifuge_command(
            CentrifugeAction.SECURE, slot=slot.identifier, key=key
        )
    for opening in (False, True):
        action = LaboratoryHingeManipulation(
            physics, HingeInstrument.CENTRIFUGE, opening
        )
        action.start()
        for _ in range(int(180 / physics.timestep)):
            action.advance()
            gaze.advance()
            physics.step()
            action.observe()
            assert_clear_environment_contacts(physics, action)
            if action.state != ProgramState.RUNNING:
                break
        assert action.state == ProgramState.SUCCEEDED, action.snapshot()
        assert action.grasp_verified
        assert action.maximum_motor_force == 0
        assert action._hand_clear()
        if not opening:
            physics.centrifuge_command(
                CentrifugeAction.START,
                rpm=parameters.rotor_rpm,
                duration=parameters.spin_seconds,
            )
            for _ in range(int(60 / physics.timestep)):
                gaze.advance()
                physics.step()
                if (
                    physics.centrifuge.remaining_seconds == 0
                    and physics.centrifuge.stopped
                ):
                    break
            assert physics.centrifuge.remaining_seconds == 0
            assert physics.centrifuge.stopped
