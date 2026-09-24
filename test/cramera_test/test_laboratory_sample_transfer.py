"""
Reusable sample transport relies on gripper and destination contacts.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cramera.laboratory_centrifuge import CentrifugeGeometry
from cramera.laboratory_centrifuge_command import CentrifugeAction
from cramera.laboratory_mixing_program import MixingPhase
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer, SampleGrasp
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_scale import place_on_pan


# %% contact-driven transfers
@pytest.mark.parametrize("axial_rotation", [0.0, np.pi / 2])
def test_pickup_aligns_with_the_measured_axis_of_a_settled_sample(
    mobile_robot: LaboratoryRobotPhysics, axial_rotation: float
) -> None:
    """
    A settled sample's upper grip center follows its observed tilt regardless of yaw.
    """
    physics = mobile_robot
    key = LaboratoryBody.CLEAR_TUBE
    place_on_pan(physics, key)
    body = physics.objects[key].body_id
    coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[body]]
    rotation = Rotation.from_euler("x", -0.1) * Rotation.from_euler("z", axial_rotation)
    physics.data.qpos[coordinate + 3 : coordinate + 7] = rotation.as_quat(
        scalar_first=True
    )
    physics.step(10000)
    assert physics.scale.snapshot()["stable"]
    measured_axis = physics.data.xmat[body].reshape(3, 3)[:, 2]
    assert np.linalg.norm(measured_axis[:2]) > 0.02
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=key,
        destination=tuple(physics.pipetting.target_position),
        release_at_destination=False,
    )
    grasp = next(stage for stage in transfer.stages if stage.phase == MixingPhase.GRASP)
    np.testing.assert_allclose(
        Rotation.from_quat(grasp.rotation).apply([0.0, 0.0, 1.0]),
        measured_axis,
        atol=1e-9,
    )


def test_blocked_torso_recovery_cannot_advance_to_sample_pickup(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    A requested lift is not accepted when the torso actuator cannot move.
    """
    physics = mobile_robot
    torso = "pr2/torso_lift_joint"
    physics.set_joint_targets({torso: 0.1})
    physics.step(2000)
    actuator = physics.model.actuator(torso).id
    physics.model.actuator_forcerange[actuator] = (0.0, 0.0)
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=LaboratoryBody.CLEAR_TUBE,
        destination=tuple(physics.pipetting.target_position),
        release_at_destination=False,
    )
    transfer.start()
    count = int((transfer.stage.duration + 1.0) / physics.timestep)
    for _ in range(count):
        transfer.advance()
        physics.step()
        transfer.observe()
    assert transfer.stage_index == 0
    assert transfer.state == ProgramState.RUNNING


def test_blocked_lift_cannot_begin_lateral_sample_transport(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    A contact-held sample must physically clear its source before transport.
    """
    physics = mobile_robot
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=LaboratoryBody.CLEAR_TUBE,
        destination=tuple(physics.centrifuge.slot_position("C1")),
        support_body=physics.scene["laboratory"]["centrifuge"]["rotorBody"],
        grasp=SampleGrasp.TOP,
    )
    transfer.start()
    for _ in range(20000):
        transfer.advance()
        physics.step()
        transfer.observe()
        if transfer.stage.phase == MixingPhase.LIFT:
            break
    assert transfer.holding
    assert transfer.stage.phase == MixingPhase.LIFT
    for name in transfer.arm_names:
        joint = physics.model.joint(name).id
        position = physics.data.qpos[physics.model.jnt_qposadr[joint]]
        physics.model.jnt_limited[joint] = True
        physics.model.jnt_range[joint] = (position - 0.0001, position + 0.0001)
    stage = transfer.stage_index
    for _ in range(int((transfer.stage.duration + 1.0) / physics.timestep)):
        transfer.advance()
        physics.step()
        transfer.observe()
    assert transfer.holding
    assert transfer.stage_index == stage
    assert transfer.state == ProgramState.RUNNING


def advance_transfer(
    physics: LaboratoryRobotPhysics, transfer: LaboratorySampleTransfer
) -> None:
    """
    Integrate a bounded real simulation until the transfer finishes.
    """
    transfer.start()
    for _ in range(150000):
        transfer.advance()
        physics.step()
        transfer.observe()
        if transfer.state != ProgramState.RUNNING:
            break
    assert transfer.state == ProgramState.SUCCEEDED, transfer.snapshot()


def test_receiver_moves_to_dispenser_then_releases_on_balance(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    An uninterrupted physical grasp connects dispensing to measured weighing.
    """
    physics = mobile_robot
    key = LaboratoryBody.CLEAR_TUBE
    before = physics.data.qpos.copy()
    dispenser = LaboratorySampleTransfer(
        physics=physics,
        object_key=key,
        destination=tuple(physics.pipetting.target_position),
        release_at_destination=False,
    )
    np.testing.assert_array_equal(physics.data.qpos, before)
    advance_transfer(physics, dispenser)
    assert dispenser.holding
    pan = physics.scene["laboratory"]["scale"]["panPosition"]
    balance = LaboratorySampleTransfer(
        physics=physics,
        object_key=key,
        destination=tuple(pan),
        support_body=physics.model.body(
            physics.model.geom_bodyid[physics.scale.pan_id]
        ).name,
        from_held=True,
    )
    advance_transfer(physics, balance)
    physics.step(1000)
    assert balance.released
    assert physics.scale.snapshot()["objects"] == [key]
    assert physics.target is None
    assert not np.any(physics.data.xfrc_applied)


def test_storage_transfer_requires_open_door_and_supported_slot(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    The robot carries a real sample down through the open cabinet front.
    """
    physics = mobile_robot
    physics.refrigerator_command("open")
    physics.step(2500)
    slot = physics.refrigerator.metadata["slots"][0]
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=LaboratoryBody.CLEAR_TUBE,
        destination=tuple(slot["position"]),
        support_body=physics.refrigerator.metadata["body"],
    )
    advance_transfer(physics, transfer)
    physics.step(1000)
    assert physics.refrigerator.occupants()[slot["id"]] == LaboratoryBody.CLEAR_TUBE
    assert transfer.released


def test_upper_grasp_seats_sample_below_centrifuge_rim(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    An upper pinch clears the chamber rim before releasing a filled sample.
    """
    physics = mobile_robot
    physics.fill_liquid(LaboratoryBody.CLEAR_TUBE, 2.0)
    physics.centrifuge_command("open")
    physics.step(2500)
    slot = next(iter(physics.centrifuge.slots))
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=LaboratoryBody.CLEAR_TUBE,
        destination=tuple(physics.centrifuge.slot_position(slot)),
        support_body=physics.scene["laboratory"]["centrifuge"]["rotorBody"],
        grasp=SampleGrasp.TOP,
    )
    advance_transfer(physics, transfer)
    physics.centrifuge_command(
        CentrifugeAction.SECURE, key=LaboratoryBody.CLEAR_TUBE, slot=slot
    )
    assert physics.centrifuge.slots[slot].key == LaboratoryBody.CLEAR_TUBE
    assert physics.liquid.tubes[LaboratoryBody.CLEAR_TUBE].volume_ml == pytest.approx(
        2.0
    )
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
