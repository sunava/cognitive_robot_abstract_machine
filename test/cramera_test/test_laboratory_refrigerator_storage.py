"""
Sample storage after the robot physically opens the refrigerator door.
"""

import pytest

from cramera.laboratory_gaze import LaboratoryGaze
from cramera.laboratory_hinge_manipulation import (
    HingeInstrument,
    LaboratoryHingeManipulation,
)
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_refrigerator_command import RefrigeratorField
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer, SampleGrasp
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_hinge_manipulation import assert_clear_environment_contacts
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% open-door storage sequence
def test_robot_stores_sample_after_opening_refrigerator(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    The released door handle permits recovery, rack pickup and supported storage.
    """
    physics = mobile_robot
    key = LaboratoryBody.CLEAR_TUBE
    physics.fill_liquid(key, 2.0)
    gaze = LaboratoryGaze(physics)
    door = LaboratoryHingeManipulation(physics, HingeInstrument.REFRIGERATOR, True)
    door.start()
    for _ in range(int(180 / physics.timestep)):
        door.advance()
        gaze.advance()
        physics.step()
        door.observe()
        assert_clear_environment_contacts(physics, door)
        if door.state != ProgramState.RUNNING:
            break
    assert door.state == ProgramState.SUCCEEDED, door.snapshot()
    slot = physics.refrigerator.metadata[RefrigeratorField.SLOTS][0]
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=key,
        destination=tuple(slot[RefrigeratorField.POSITION]),
        support_body=physics.refrigerator.metadata[RefrigeratorField.BODY],
        grasp=SampleGrasp.FRONT,
    )
    transfer.start()
    for _ in range(int(240 / physics.timestep)):
        transfer.advance()
        gaze.advance()
        physics.step()
        transfer.observe()
        if transfer.state != ProgramState.RUNNING:
            break
    assert transfer.state == ProgramState.SUCCEEDED, transfer.snapshot()
    assert transfer.released
    physics.step(2000)
    assert physics.refrigerator.occupants()[slot[RefrigeratorField.ID]] == key
    assert physics.liquid.tubes[key].volume_ml == pytest.approx(2.0)
    assert transfer._supported_destination(
        physics.objects[key].body_id, transfer.parameters.position_tolerance
    )
