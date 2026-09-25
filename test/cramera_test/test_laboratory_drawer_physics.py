"""
Physical drawer travel and an independently supported sample rack.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree

import mujoco
import numpy as np

from cramera.laboratory_physics import LaboratoryPhysics
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics import physics


# %% authored articulation
def test_drawer_imports_its_authored_prismatic_joint(
    physics: LaboratoryPhysics,
) -> None:
    """
    The drawer retains its URDF travel limits and passive rail resistance.
    """
    description = ElementTree.parse(physics.bundle_directory / "environment.urdf")
    name = physics.scene["laboratory"]["drawer"]["joint"]
    authored = description.find(f"joint[@name='{name}']")
    joint = mujoco.mj_name2id(physics.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint >= 0
    assert physics.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE
    np.testing.assert_array_equal(
        physics.model.jnt_axis[joint],
        np.fromstring(authored.find("axis").attrib["xyz"], sep=" "),
    )
    limits = authored.find("limit").attrib
    np.testing.assert_array_equal(
        physics.model.jnt_range[joint],
        [float(limits["lower"]), float(limits["upper"])],
    )
    velocity = physics.model.jnt_dofadr[joint]
    dynamics = authored.find("dynamics").attrib
    assert physics.model.dof_damping[velocity] == float(dynamics["damping"])
    assert physics.model.dof_frictionloss[velocity] == float(dynamics["friction"])


def test_rack_and_samples_have_independent_free_bodies(
    physics: LaboratoryPhysics,
) -> None:
    """
    Rack motion can carry separate samples only through physical contacts.
    """
    rack = physics.model.body(LaboratoryBody.RACK).id
    assert physics.model.body_jntnum[rack] == 1
    joint = physics.model.body_jntadr[rack]
    assert physics.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
    assert physics.model.body_mass[rack] > 0
    for key in physics.liquid.tubes:
        body = physics.objects[key].body_id
        assert physics.model.body_parentid[body] == 0
        assert physics.model.jnt_type[physics.model.body_jntadr[body]] == (
            mujoco.mjtJoint.mjJNT_FREE
        )
        assert body != rack


# %% bounded drawer control
def test_drawer_hold_moves_through_physics_without_resetting_samples(
    physics: LaboratoryPhysics,
) -> None:
    """
    A requested opening only changes controls before solver-driven rail travel.
    """
    drawer = physics.drawer
    before = physics.data.qpos.copy()
    drawer.hold(drawer.open_position)
    np.testing.assert_array_equal(physics.data.qpos, before)
    physics.step(1000)
    assert 0 < drawer.position < drawer.open_position
    physics.step(5000)
    assert drawer.open
    assert abs(physics.data.actuator_force[drawer.actuator]) <= (
        drawer.parameters.maximum_force
    )
    drawer.hold(0.0)
    physics.step(6000)
    assert drawer.closed


def test_drawer_hand_control_disables_motor_without_changing_state(
    physics: LaboratoryPhysics,
) -> None:
    """
    The physical hand exclusively drives the rail until its endpoint is held.
    """
    drawer = physics.drawer
    drawer.hold(drawer.open_position)
    physics.step(1500)
    before = physics.data.qpos.copy()
    drawer.release_to_hand()
    np.testing.assert_array_equal(physics.data.qpos, before)
    physics.step(200)
    assert physics.data.actuator_force[drawer.actuator] == 0.0
    measured = drawer.position
    before = physics.data.qpos.copy()
    drawer.restore_hold(measured)
    np.testing.assert_array_equal(physics.data.qpos, before)
    assert drawer.reference_position == measured
    assert drawer.target_position == measured
    physics.step(1000)
    assert abs(drawer.position - measured) < drawer.parameters.position_tolerance


def test_drawer_endpoint_hold_accepts_small_measured_stop_deflection(
    physics: LaboratoryPhysics,
) -> None:
    """
    Compliant rail endpoints produce valid bounded holding references.
    """
    drawer = physics.drawer
    deflection = drawer.parameters.position_tolerance / 2
    before = physics.data.qpos.copy()
    drawer.restore_hold(-deflection)
    assert drawer.reference_position == 0.0
    drawer.restore_hold(drawer.open_position + deflection)
    assert drawer.reference_position == drawer.open_position
    np.testing.assert_array_equal(physics.data.qpos, before)


# %% contact-supported rack preparation
def test_prepared_rack_and_four_separate_samples_follow_drawer_contacts(
    physics: LaboratoryPhysics,
) -> None:
    """
    Supported contents follow slow rail travel without attachment constraints.
    """
    physics.prepare_rack_in_drawer()
    physics.step(1000)
    rack = physics.model.body(LaboratoryBody.RACK).id
    start = physics.data.xpos[rack].copy()
    occupants = physics.rack.occupants()
    assert set(occupants.values()) - {None} == set(physics.liquid.tubes)
    before = physics.data.qpos.copy()
    physics.drawer.hold(physics.drawer.open_position)
    np.testing.assert_array_equal(physics.data.qpos, before)
    physics.step(6000)
    assert physics.drawer.open
    assert physics.rack.supported_by(LaboratoryBody.DRAWER)
    np.testing.assert_allclose(
        physics.data.xpos[rack],
        start + np.array([0.0, -physics.drawer.open_position, 0.0]),
        atol=0.004,
    )
    assert physics.rack.occupants() == occupants
    assert physics.data.xpos[rack, 1] + physics.rack.parameters.depth / 2 < -0.375
    before = physics.data.qpos.copy()
    snapshot = physics.snapshot()
    np.testing.assert_array_equal(physics.data.qpos, before)
    np.testing.assert_array_equal(
        snapshot["bodyPoses"][LaboratoryBody.RACK], physics.rack.pose
    )
