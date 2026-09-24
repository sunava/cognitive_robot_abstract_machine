"""
Upper grasp access to the rear rotor remains clear of the open lid.
"""

from __future__ import annotations

import pytest

from cramera.laboratory_centrifuge import CentrifugeGeometry
from cramera.laboratory_centrifuge_command import CentrifugeAction
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer, SampleGrasp
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_sample_transfer import advance_transfer


# %% rear holder clearance
def test_upper_grasp_seats_sample_in_rear_holder_and_clears_the_lid(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    A full sample reaches the rear pocket and the withdrawn hand permits closure.
    """
    physics = mobile_robot
    key = LaboratoryBody.CLEAR_TUBE
    physics.fill_liquid(key, 2.0)
    physics.centrifuge_command(CentrifugeAction.OPEN)
    physics.step(2500)
    holder = "C2"
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=key,
        destination=tuple(physics.centrifuge.slot_position(holder)),
        support_body=physics.scene["laboratory"]["centrifuge"]["rotorBody"],
        grasp=SampleGrasp.TOP,
    )
    advance_transfer(physics, transfer)
    physics.centrifuge_command(CentrifugeAction.SECURE, key=key, slot=holder)
    assert physics.centrifuge.slots[holder].key == key
    assert physics.liquid.tubes[key].volume_ml == pytest.approx(2.0)

    physics.centrifuge_command(CentrifugeAction.CLOSE)
    for _ in range(4000):
        physics.step()
        for contact in physics.data.contact:
            bodies = [
                physics.model.body(physics.model.geom_bodyid[index]).name
                for index in (contact.geom1, contact.geom2)
            ]
            lid_contact = CentrifugeGeometry.LID_BODY in bodies
            robot_contact = any(body.startswith("pr2/") for body in bodies)
            assert not (lid_contact and robot_contact and contact.dist < -0.0001)
    assert physics.centrifuge.lid_closed
