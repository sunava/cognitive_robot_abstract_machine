"""
Fixed handle frames preserve the laboratory's compiled contact dynamics.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ElementTree
from dataclasses import replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cramera.laboratory_handles import LaboratoryHandle
from cramera.laboratory_physics import LaboratoryPhysics
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics import physics
from .test_laboratory_physics_robot import robot_physics


# %% physical model equivalence
def test_handle_frames_preserve_laboratory_contact_dynamics(
    physics: LaboratoryPhysics,
) -> None:
    """
    Adding fixed annotation frames changes neither contact geometry nor motion.
    """
    assert_handle_frames_preserve_physics(physics)


def test_handle_frames_preserve_robot_inertial_bounds_and_actuators(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Robot compiler mass bounds do not add inertia to empty handle frames.
    """
    assert_handle_frames_preserve_physics(mobile_robot)


def test_massless_handle_sites_follow_their_moving_parents(
    physics: LaboratoryPhysics,
) -> None:
    """
    Named MuJoCo sites retain the authored frames throughout real hinge motion.
    """
    description = ElementTree.parse(physics.bundle_directory / "environment.urdf")
    joints = [
        joint
        for joint in description.findall("joint")
        if joint.find("child").attrib["link"] in LaboratoryHandle
    ]
    assert len(joints) == len(LaboratoryHandle)
    before = {}
    for joint in joints:
        name = joint.find("child").attrib["link"]
        site = mujoco.mj_name2id(physics.model, mujoco.mjtObj.mjOBJ_SITE, name)
        assert site >= 0
        before[name] = physics.data.site_xpos[site].copy()
        assert_handle_site_pose(physics, joint)
    physics.centrifuge_command("close")
    physics.refrigerator_command("open")
    physics.step(500)
    for joint in joints:
        assert_handle_site_pose(physics, joint)
    for name in (
        LaboratoryHandle.CENTRIFUGE_LID,
        LaboratoryHandle.REFRIGERATOR_DOOR,
    ):
        site = physics.model.site(name).id
        assert not np.allclose(physics.data.site_xpos[site], before[name])


def assert_handle_frames_preserve_physics(physics: LaboratoryPhysics) -> None:
    """
    Compare one generated world with the same description stripped of handles.
    """
    description = ElementTree.parse(physics.bundle_directory / "environment.urdf")
    root = description.getroot()
    handles = set(LaboratoryHandle)
    links = [link for link in root.findall("link") if link.attrib["name"] in handles]
    joints = [
        joint
        for joint in root.findall("joint")
        if joint.find("child").attrib["link"] in handles
    ]
    assert {link.attrib["name"] for link in links} == handles
    assert len(joints) == len(handles)
    for link in links:
        assert len(link) == 0
        root.remove(link)
    for joint in joints:
        assert joint.attrib["type"] == "fixed"
        root.remove(joint)
    baseline_directory = physics.bundle_directory.with_name(
        physics.bundle_directory.name + "-without-handles"
    )
    shutil.copytree(physics.bundle_directory, baseline_directory)
    description.write(baseline_directory / "environment.urdf", encoding="utf-8")
    baseline = replace(physics, bundle_directory=baseline_directory)
    assert_physical_model_equal(physics.model, baseline.model)
    for joint in joints:
        assert_handle_site_pose(physics, joint)
    for world in (physics, baseline):
        world.centrifuge_command("close")
        world.refrigerator_command("open")
        world.step(500)
    np.testing.assert_allclose(
        physics.data.qpos, baseline.data.qpos, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        physics.data.qvel, baseline.data.qvel, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        physics.data.qacc, baseline.data.qacc, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(physics.data.qM, baseline.data.qM, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        physics.data.actuator_force, baseline.data.actuator_force, rtol=0, atol=1e-12
    )
    for joint in joints:
        assert_handle_site_pose(physics, joint)


def assert_physical_model_equal(
    model: mujoco.MjModel, baseline: mujoco.MjModel
) -> None:
    """
    Require equal named collision geometry, inertials, joints and drive settings.
    """
    assert (model.ngeom, model.njnt, model.nq, model.nv, model.nu) == (
        baseline.ngeom,
        baseline.njnt,
        baseline.nq,
        baseline.nv,
        baseline.nu,
    )
    assert [model.geom(index).name for index in range(model.ngeom)] == [
        baseline.geom(index).name for index in range(baseline.ngeom)
    ]
    assert [model.joint(index).name for index in range(model.njnt)] == [
        baseline.joint(index).name for index in range(baseline.njnt)
    ]
    assert [model.actuator(index).name for index in range(model.nu)] == [
        baseline.actuator(index).name for index in range(baseline.nu)
    ]
    for actual, expected in (
        (model.geom_type, baseline.geom_type),
        (model.geom_size, baseline.geom_size),
        (model.geom_pos, baseline.geom_pos),
        (model.geom_quat, baseline.geom_quat),
        (model.geom_friction, baseline.geom_friction),
        (model.geom_contype, baseline.geom_contype),
        (model.geom_conaffinity, baseline.geom_conaffinity),
        (model.geom_condim, baseline.geom_condim),
        (model.geom_solref, baseline.geom_solref),
        (model.geom_solimp, baseline.geom_solimp),
        (model.geom_margin, baseline.geom_margin),
        (model.geom_gap, baseline.geom_gap),
        (model.jnt_type, baseline.jnt_type),
        (model.jnt_axis, baseline.jnt_axis),
        (model.jnt_pos, baseline.jnt_pos),
        (model.jnt_range, baseline.jnt_range),
        (model.jnt_stiffness, baseline.jnt_stiffness),
        (model.dof_damping, baseline.dof_damping),
        (model.dof_armature, baseline.dof_armature),
        (model.dof_frictionloss, baseline.dof_frictionloss),
        (model.actuator_trnid, baseline.actuator_trnid),
        (model.actuator_gear, baseline.actuator_gear),
        (model.actuator_gainprm, baseline.actuator_gainprm),
        (model.actuator_biasprm, baseline.actuator_biasprm),
        (model.actuator_dynprm, baseline.actuator_dynprm),
        (model.actuator_ctrlrange, baseline.actuator_ctrlrange),
        (model.actuator_forcerange, baseline.actuator_forcerange),
    ):
        np.testing.assert_array_equal(actual, expected)
    for index in range(baseline.nbody):
        body = model.body(baseline.body(index).name).id
        for actual, expected in (
            (model.body_mass[body], baseline.body_mass[index]),
            (model.body_inertia[body], baseline.body_inertia[index]),
            (model.body_ipos[body], baseline.body_ipos[index]),
            (model.body_iquat[body], baseline.body_iquat[index]),
            (model.body_subtreemass[body], baseline.body_subtreemass[index]),
        ):
            np.testing.assert_array_equal(actual, expected)
    for handle in LaboratoryHandle:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, handle)
        if body >= 0:
            assert model.body_mass[body] == 0
            np.testing.assert_array_equal(model.body_inertia[body], np.zeros(3))
            assert model.body_jntnum[body] == 0


def assert_handle_site_pose(
    physics: LaboratoryPhysics, joint: ElementTree.Element
) -> None:
    """
    Check an invisible, massless site against its URDF parent transform.
    """
    name = joint.find("child").attrib["link"]
    site = physics.model.site(name).id
    parent = physics.model.body(joint.find("parent").attrib["link"]).id
    assert physics.model.site_bodyid[site] == parent
    origin = joint.find("origin")
    parent_T_handle = np.eye(4)
    parent_T_handle[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ")
    parent_T_handle[:3, :3] = Rotation.from_euler(
        "xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    ).as_matrix()
    world_T_parent = np.eye(4)
    world_T_parent[:3, 3] = physics.data.xpos[parent]
    world_T_parent[:3, :3] = physics.data.xmat[parent].reshape(3, 3)
    world_T_handle = world_T_parent @ parent_T_handle
    np.testing.assert_allclose(
        physics.data.site_xpos[site], world_T_handle[:3, 3], atol=1e-12
    )
    np.testing.assert_allclose(
        physics.data.site_xmat[site].reshape(3, 3), world_T_handle[:3, :3], atol=1e-12
    )
    assert physics.model.site_rgba[site, 3] == 0
