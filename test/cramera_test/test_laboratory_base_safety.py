"""
Robot travel respects the visible bench and measured swept geometry.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from cramera.laboratory_centrifuge_command import CentrifugeAction
from cramera.laboratory_hinge_manipulation import (
    HingeInstrument,
    LaboratoryHingeManipulation,
)
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer, SampleGrasp
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics, RobotBaseJoint
from cramera.laboratory_robot_base import LaboratoryBasePath, UnsafeBasePath
from cramera.laboratory_world import CabinetStructure, LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% cabinet contact geometry
def test_visible_cabinet_panels_exist_in_contact_world(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    All authored cabinet panels also prevent physical robot penetration.
    """
    cabinet = mobile_robot.model.body(LaboratoryBody.CABINET)
    assert cabinet.geomnum[0] == len(CabinetStructure().panels())
    for geometry, panel in zip(
        range(cabinet.geomadr[0], cabinet.geomadr[0] + cabinet.geomnum[0]),
        CabinetStructure().panels(),
    ):
        np.testing.assert_allclose(
            mobile_robot.model.geom_pos[geometry], panel.origin.to_np()[:3, 3]
        )
        np.testing.assert_allclose(
            mobile_robot.model.geom_size[geometry], panel.size.to_np() / 2
        )


def test_swept_base_path_rejects_forward_cabinet_intrusion_without_mutation(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    A collision ahead cancels travel before any physical target or pose changes.
    """
    physics = mobile_robot
    before = physics.data.qpos.copy()
    targets = physics.joint_targets.copy()
    with pytest.raises(UnsafeBasePath):
        LaboratoryBasePath(physics).validate({RobotBaseJoint.DEPTH: 0.20})
    np.testing.assert_array_equal(physics.data.qpos, before)
    assert physics.joint_targets == targets


def test_swept_base_path_accepts_retreat_and_lateral_front_corridor(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    The same checker allows ordinary front-side base travel.
    """
    physics = mobile_robot
    for target in (
        {RobotBaseJoint.DEPTH: -0.25},
        {RobotBaseJoint.HORIZONTAL: -0.30},
    ):
        before = physics.data.qpos.copy()
        LaboratoryBasePath(physics).validate(target)
        np.testing.assert_array_equal(physics.data.qpos, before)


def test_cabinet_physically_stops_a_force_driven_base(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Even a direct drive command cannot carry the chassis through cabinet doors.
    """
    physics = mobile_robot
    physics.set_joint_targets({RobotBaseJoint.DEPTH: 0.20})
    physics.step(2500)
    assert physics.robot_state()[RobotBaseJoint.DEPTH] < 0.08
    cabinet = physics.model.body(LaboratoryBody.CABINET).id
    contacts = [
        contact
        for contact in physics.data.contact[: physics.data.ncon]
        if cabinet
        in {
            physics.model.geom_bodyid[contact.geom1],
            physics.model.geom_bodyid[contact.geom2],
        }
    ]
    assert contacts
    assert min(contact.dist for contact in contacts) > -0.005


# %% whole-robot swept collision checks
def test_swept_base_path_checks_arm_geometry_above_the_chassis(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    A small obstacle in the carried arm's path blocks otherwise clear chassis travel.
    """
    physics = mobile_robot
    palm = physics.model.body("pr2/l_gripper_palm_link").id
    stopper = physics.objects[LaboratoryBody.STOPPER].body_id
    coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[stopper]]
    physics.data.qpos[coordinate : coordinate + 3] = physics.data.xpos[palm] + np.array(
        [-0.2, 0.0, 0.0]
    )
    mujoco.mj_forward(physics.model, physics.data)
    with pytest.raises(UnsafeBasePath) as obstruction:
        LaboratoryBasePath(physics).validate({RobotBaseJoint.HORIZONTAL: -0.3})
    assert obstruction.value.robot_geometry.startswith("pr2/l_")
    assert obstruction.value.obstacle_geometry.startswith(LaboratoryBody.STOPPER)


def test_swept_base_path_detects_obstacle_between_clear_endpoints(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Endpoint-only collision checks cannot miss a thin obstruction crossed en route.
    """
    physics = mobile_robot
    palm = physics.model.body("pr2/l_gripper_palm_link").id
    stopper = physics.objects[LaboratoryBody.STOPPER].body_id
    coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[stopper]]
    physics.data.qpos[coordinate : coordinate + 3] = physics.data.xpos[palm] + np.array(
        [-0.2, 0.0, 0.0]
    )
    mujoco.mj_forward(physics.model, physics.data)
    checker = LaboratoryBasePath(physics)
    target = {RobotBaseJoint.HORIZONTAL: -0.5}
    with pytest.raises(UnsafeBasePath):
        checker.validate(target)
    scratch = mujoco.MjData(physics.model)
    scratch.qpos[:] = physics.data.qpos
    scratch.qpos[physics.robot_joints[RobotBaseJoint.HORIZONTAL].position_index] = (
        target[RobotBaseJoint.HORIZONTAL]
    )
    mujoco.mj_kinematics(physics.model, scratch)
    stopper_geometries = np.flatnonzero(physics.model.geom_bodyid == stopper)
    for robot_geometry in checker.moving_geometry:
        for stopper_geometry in stopper_geometries:
            assert (
                mujoco.mj_geomDistance(
                    physics.model,
                    scratch,
                    int(robot_geometry),
                    int(stopper_geometry),
                    checker.clearance,
                    None,
                )
                >= checker.clearance
            )


def test_centrifuge_transfer_drives_with_robot_and_vial_clear_of_environment(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    A successful loaded transfer completes its base stages without unintended contact.
    """
    physics = mobile_robot
    key = LaboratoryBody.CLEAR_TUBE
    physics.fill_liquid(key, 2.0)
    physics.centrifuge_command(CentrifugeAction.OPEN)
    physics.step(2500)
    slot = next(iter(physics.centrifuge.slots))
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=key,
        destination=tuple(physics.centrifuge.slot_position(slot)),
        support_body=physics.scene["laboratory"]["centrifuge"]["rotorBody"],
        grasp=SampleGrasp.TOP,
    )
    transfer.start()
    travel_steps = 0
    for _ in range(150000):
        driving = transfer.stage.base_targets is not None
        transfer.advance()
        physics.step()
        transfer.observe()
        if driving:
            travel_steps += 1
            for contact in physics.data.contact[: physics.data.ncon]:
                first, second = [
                    physics.model.body(physics.model.geom_bodyid[index]).name
                    for index in (contact.geom1, contact.geom2)
                ]
                first_moves = first.startswith("pr2/") or (
                    transfer.stage.closed and first == key
                )
                second_moves = second.startswith("pr2/") or (
                    transfer.stage.closed and second == key
                )
                if first_moves == second_moves:
                    continue
                other = second if first_moves else first
                if other == "world":
                    continue
                assert contact.dist >= -0.0001, (first, second, contact.dist)
        if transfer.state != ProgramState.RUNNING:
            break
    assert travel_steps > 0
    assert transfer.state == ProgramState.SUCCEEDED, transfer.snapshot()


def test_base_rotation_checks_the_swept_arm_arc(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Turning the chassis cannot sweep an extended arm through a bench instrument.
    """
    physics = mobile_robot
    before = physics.data.qpos.copy()
    with pytest.raises(UnsafeBasePath):
        LaboratoryBasePath(physics).validate({RobotBaseJoint.HEADING: np.pi})
    np.testing.assert_array_equal(physics.data.qpos, before)


def test_refrigerator_dock_clears_the_complete_door_swing(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    The opening door must stay outside the stationary robot chassis throughout its arc.
    """
    physics = mobile_robot
    manipulation = LaboratoryHingeManipulation(
        physics, HingeInstrument.REFRIGERATOR, True
    )
    scratch = mujoco.MjData(physics.model)
    scratch.qpos[:] = physics.data.qpos
    for joint, target in zip(
        (RobotBaseJoint.HORIZONTAL, RobotBaseJoint.DEPTH), manipulation.dock
    ):
        scratch.qpos[physics.robot_joints[joint].position_index] = target
    chassis = physics.model.body("pr2/base_link").id
    for angle in np.linspace(0.0, manipulation.target_angle, 100):
        scratch.qpos[physics.model.jnt_qposadr[manipulation.joint]] = angle
        mujoco.mj_forward(physics.model, scratch)
        for contact in scratch.contact[: scratch.ncon]:
            bodies = {
                physics.model.geom_bodyid[contact.geom1],
                physics.model.geom_bodyid[contact.geom2],
            }
            if chassis in bodies:
                assert contact.dist >= -0.0001, (angle, contact.dist)
