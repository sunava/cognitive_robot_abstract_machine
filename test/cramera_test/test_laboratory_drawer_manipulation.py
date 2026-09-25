"""
The PR2 operates the storage drawer through opposing handle contacts.
"""

from __future__ import annotations

import numpy as np
import mujoco

from cramera.laboratory_drawer_manipulation import (
    DrawerPhase,
    LaboratoryDrawerManipulation,
)
from cramera.laboratory_physics_program import JointReference, ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_hinge_manipulation import assert_clear_environment_contacts
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% measured contact manipulation
def test_drawer_withdrawal_clears_the_next_sample_torso_recovery(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    The empty hand finishes its retreat before rising past the worktop edge.
    """
    physics = mobile_robot
    physics.drawer.hold(physics.drawer.open_position)
    physics.step(int(16 / physics.timestep))
    action = LaboratoryDrawerManipulation(physics, False)
    action.start()
    for _ in range(int(180 / physics.timestep)):
        action.advance()
        physics.step()
        action.observe()
        assert_clear_environment_contacts(physics, action)
        if action.phase == DrawerPhase.WITHDRAW:
            assert_no_robot_scenery_force(physics)
        if action.state != ProgramState.RUNNING:
            break
    assert action.state == ProgramState.SUCCEEDED, action.snapshot()
    assert np.linalg.norm(action.hand.position - action.waypoint) < (
        action.parameters.hand_tolerance
    )
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=LaboratoryBody.CLEAR_TUBE,
        destination=physics.pipetting.target_position,
        release_at_destination=False,
    )
    transfer.start()
    recovery = transfer.stage
    assert recovery.joint_targets
    for _ in range(
        int(
            (recovery.duration + transfer.parameters.stage_timeout + 1)
            / physics.timestep
        )
    ):
        transfer.advance()
        physics.step()
        assert_no_robot_scenery_force(physics)
        transfer.observe()
        if transfer.stage_index > 0 or transfer.state != ProgramState.RUNNING:
            break
    assert transfer.state == ProgramState.RUNNING, transfer.snapshot()
    assert transfer.stage_index == 1
    for name, target in recovery.joint_targets.items():
        assert abs(physics.robot_state()[name] - target) <= (
            transfer.parameters.position_tolerance
        )


def assert_no_robot_scenery_force(physics: LaboratoryRobotPhysics) -> None:
    """
    An unladen withdrawn hand must not press against any non-ground fixture.
    """
    for index, contact in enumerate(physics.data.contact):
        geometries = (contact.geom1, contact.geom2)
        bodies = [
            physics.model.body(physics.model.geom_bodyid[geometry]).name
            for geometry in geometries
        ]
        robots = [body.startswith("pr2/") for body in bodies]
        if all(robots) or not any(robots):
            continue
        if any(
            physics.model.geom_type[geometry] == mujoco.mjtGeom.mjGEOM_PLANE
            for geometry in geometries
        ):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(physics.model, physics.data, index, force)
        assert force[0] == 0, (bodies, contact.dist, force)


def test_robot_opens_drawer_with_four_contact_supported_samples(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    The loaded drawer moves through the robot grasp while its contents stay seated.
    """
    physics = mobile_robot
    physics.prepare_rack_in_drawer()
    physics.step(1000)
    occupants = physics.rack.occupants()
    assert set(occupants.values()) - {None} == set(physics.liquid.tubes)
    action = LaboratoryDrawerManipulation(physics, True)
    before = physics.data.qpos.copy()
    action.start()
    np.testing.assert_array_equal(physics.data.qpos, before)
    for _ in range(int(180 / physics.timestep)):
        before = physics.data.qpos.copy()
        action.advance()
        assert np.array_equal(physics.data.qpos, before)
        physics.step()
        before = physics.data.qpos.copy()
        action.observe()
        assert np.array_equal(physics.data.qpos, before)
        assert_clear_environment_contacts(physics, action)
        if action.state != ProgramState.RUNNING:
            break
    assert action.state == ProgramState.SUCCEEDED, action.snapshot()
    assert action.grasp_verified
    assert action.maximum_motor_force == 0
    assert abs(action.position - action.target_position) <= (
        action.parameters.position_tolerance
    )
    assert physics.rack.supported_by(LaboratoryBody.DRAWER)
    assert physics.rack.occupants() == occupants


def test_robot_opens_and_closes_drawer_using_its_handle(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    A physical front grasp pulls and pushes the drawer with its motor disabled.
    """
    physics = mobile_robot
    for opening in (True, False):
        before = physics.data.qpos.copy()
        action = LaboratoryDrawerManipulation(physics, opening)
        action.start()
        np.testing.assert_array_equal(physics.data.qpos, before)
        previous = None
        for _ in range(int(180 / physics.timestep)):
            before = physics.data.qpos.copy()
            action.advance()
            assert np.array_equal(physics.data.qpos, before)
            physics.step()
            before = physics.data.qpos.copy()
            action.observe()
            assert np.array_equal(physics.data.qpos, before)
            assert_clear_environment_contacts(physics, action)
            if action.phase != previous:
                print(action.snapshot(), flush=True)
                previous = action.phase
            if action.state != ProgramState.RUNNING:
                break
        assert action.state == ProgramState.SUCCEEDED, action.snapshot()
        assert action.grasp_verified
        assert action.maximum_handle_force > action.parameters.contact_force
        assert action.maximum_motor_force == 0
        assert (
            abs(action.position - action.target_position)
            <= action.parameters.position_tolerance
        )
        assert action.handle_force < action.parameters.contact_force
        assert not np.any(physics.data.xfrc_applied)


def test_cancelled_drawer_operation_restores_handle_and_finger_actuators(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Cancellation leaves the drawer held and restores the glass-safe finger profile.
    """
    physics = mobile_robot
    original_ranges = physics.model.actuator_forcerange.copy()
    finger = physics.model.actuator(JointReference.GRIPPER_JOINT).id
    action = LaboratoryDrawerManipulation(physics, True)
    action.start()
    assert not np.array_equal(
        physics.model.actuator_forcerange[finger], original_ranges[finger]
    )
    action.cancel()
    assert action.state == ProgramState.FAILED
    np.testing.assert_array_equal(physics.model.actuator_forcerange, original_ranges)


def test_cancellation_restores_glass_safe_finger_stiffness(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    The stronger drawer-bar grip does not leak into subsequent glass handling.
    """
    physics = mobile_robot
    finger = physics.model.actuator(JointReference.GRIPPER_JOINT).id
    gain = physics.model.actuator_gainprm[finger].copy()
    bias = physics.model.actuator_biasprm[finger].copy()
    action = LaboratoryDrawerManipulation(physics, True)
    action.start()
    assert physics.model.actuator_gainprm[finger, 0] > gain[0]
    action.cancel()
    np.testing.assert_array_equal(physics.model.actuator_gainprm[finger], gain)
    np.testing.assert_array_equal(physics.model.actuator_biasprm[finger], bias)
