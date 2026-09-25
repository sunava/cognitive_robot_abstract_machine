"""
Per-step contact auditing for robot transport of an independently loaded rack.
"""

from enum import StrEnum

import mujoco
import pytest

from cramera.laboratory_mixing_program import LaboratoryMixingProgram
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_rack import RackGeometry
from cramera.laboratory_rack_transfer import LaboratoryRackTransfer, RackTransferPhase
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_carried_rack import carried_rack
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% exact intended support surfaces
class RackSupportGeometry(StrEnum):
    """
    Only these external surfaces intentionally support the moving carrier.
    """

    BASE = "laboratory_rack_rack_base"
    """
    Carrier underside contacting the drawer or workbench.
    """

    WORKTOP = "laboratory_room_worktop"
    """
    Placement surface on the bench.
    """


def assert_rack_transfer_contacts(
    physics: LaboratoryRobotPhysics, action: LaboratoryRackTransfer
) -> None:
    """
    Reject unintended robot, rack or carried-sample penetration at every step.
    """
    carried = {RackGeometry.BODY, *action.required_sample_keys}
    for contact in physics.data.contact:
        if contact.dist >= -0.0002:
            continue
        geometries = (contact.geom1, contact.geom2)
        names = {physics.model.geom(geometry).name for geometry in geometries}
        bodies = [
            physics.model.body(physics.model.geom_bodyid[geometry]).name
            for geometry in geometries
        ]
        robots = [body.startswith("pr2/") for body in bodies]
        if all(robots):
            continue
        if not any(robots) and not carried.intersection(bodies):
            continue
        if any(robots) and any(
            physics.model.geom_type[geometry] == mujoco.mjtGeom.mjGEOM_PLANE
            for geometry in geometries
        ):
            continue
        grasp = action.handle in geometries and any(
            body in LaboratoryMixingProgram.FINGERTIPS for body in bodies
        )
        sample_support = RackGeometry.BODY in bodies and any(
            body in action.required_sample_keys for body in bodies
        )
        drawer_support = names == {
            RackSupportGeometry.BASE,
            RackGeometry.DRAWER_FLOOR,
        } and action.phase in (
            RackTransferPhase.STOW,
            RackTransferPhase.DOCK,
            RackTransferPhase.CLEARANCE,
            RackTransferPhase.APPROACH,
            RackTransferPhase.GRASP,
            RackTransferPhase.LIFT,
        )
        bench_support = names == {
            RackSupportGeometry.BASE,
            RackSupportGeometry.WORKTOP,
        } and action.phase in (
            RackTransferPhase.PLACE,
            RackTransferPhase.RELEASE,
            RackTransferPhase.WITHDRAW,
            RackTransferPhase.COMPLETE,
        )
        if not (grasp or sample_support or drawer_support or bench_support):
            raise AssertionError((action.phase, bodies, names, contact.dist))


# %% actual contact detection
def test_contact_audit_rejects_an_obstacle_touching_a_carried_sample(
    carried_rack: LaboratoryRobotPhysics,
) -> None:
    """
    Payload-to-scenery penetration cannot hide behind a clear robot chassis.
    """
    physics = carried_rack
    sample = physics.model.body(LaboratoryBody.CLEAR_TUBE).id
    stopper = physics.model.body(LaboratoryBody.STOPPER).id
    coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[stopper]]
    physics.data.qpos[coordinate : coordinate + 3] = physics.data.xpos[sample] + [
        0.0,
        0.0,
        0.10,
    ]
    mujoco.mj_forward(physics.model, physics.data)
    action = LaboratoryRackTransfer(physics, tuple(physics.liquid.tubes))
    action.phase = RackTransferPhase.TRAVEL
    with pytest.raises(AssertionError) as failure:
        assert_rack_transfer_contacts(physics, action)
    assert set(failure.value.args[0][1]) == {
        LaboratoryBody.CLEAR_TUBE,
        LaboratoryBody.STOPPER,
    }


def test_contact_audit_accepts_real_bench_and_sample_support(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    The placed carrier retains its legitimate worktop and vessel contacts.
    """
    physics = mobile_robot
    physics.step(500)
    action = LaboratoryRackTransfer(physics, tuple(physics.liquid.tubes))
    action.phase = RackTransferPhase.PLACE
    assert physics.rack.supported_by(LaboratoryBody.ROOM)
    assert_rack_transfer_contacts(physics, action)
