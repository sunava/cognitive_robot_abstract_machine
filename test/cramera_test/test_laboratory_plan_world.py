"""A native plan world observes detached measurements from the physical laboratory."""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import FrozenInstanceError, dataclass, field
from enum import StrEnum

import mujoco
import numpy as np
import pytest

from cramera.laboratory_centrifuge import CentrifugeGeometry
from cramera.laboratory_centrifuge_command import CentrifugeAction
from cramera.laboratory_physics_robot import (
    LaboratoryRobotPhysics,
    RobotBaseJoint,
    RobotHeadJoint,
)
from cramera.laboratory_plan_world import LaboratoryPlanWorld
from cramera.laboratory_refrigerator import RefrigeratorGeometry
from cramera.laboratory_refrigerator_command import RefrigeratorAction
from cramera.laboratory_world import LaboratoryBody, LaboratoryWorld
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Pose, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_world import laboratory_bundle


# %% portable measured articulation
class MeasurementFixture(StrEnum):
    """Names shared by the portable physical and semantic test worlds."""

    ROBOT = "bot"
    """The robot already supplied by the shared cylinder fixture."""
    ARM = "arm"
    """An articulated robot body."""
    HINGE = "arm_hinge"
    """A scalar joint measuring arm rotation."""
    SAMPLE = "sample"
    """An independently floating sample body."""


@dataclass
class DetachedArticulation:
    """Observe a real small MuJoCo robot without external model resources."""

    model: mujoco.MjModel = field(init=False)
    """A free robot root, scalar arm joint and free sample."""
    data: mujoco.MjData = field(init=False)
    """Independent physical positions copied by the semantic observer."""

    def __post_init__(self) -> None:
        """Create measured articulation without network or optional mesh assets."""
        root = ElementTree.Element("mujoco")
        bodies = ElementTree.SubElement(root, "worldbody")
        robot = ElementTree.SubElement(
            bodies, "body", name=MeasurementFixture.ROBOT, pos="0.4 0.3 0"
        )
        ElementTree.SubElement(robot, "freejoint")
        ElementTree.SubElement(robot, "geom", type="sphere", size="0.1")
        arm = ElementTree.SubElement(robot, "body", name=MeasurementFixture.ARM)
        ElementTree.SubElement(
            arm, "joint", name=MeasurementFixture.HINGE, axis="0 1 0"
        )
        ElementTree.SubElement(arm, "geom", type="sphere", size="0.05")
        sample = ElementTree.SubElement(
            bodies, "body", name=MeasurementFixture.SAMPLE, pos="1 2 0.8"
        )
        ElementTree.SubElement(sample, "freejoint")
        ElementTree.SubElement(sample, "geom", type="sphere", size="0.02")
        self.model = mujoco.MjModel.from_xml_string(
            ElementTree.tostring(root, encoding="unicode")
        )
        self.data = mujoco.MjData(self.model)
        self.data.qpos[self.model.joint(MeasurementFixture.HINGE).qposadr[0]] = 0.4
        mujoco.mj_forward(self.model, self.data)

    def robot_state(self) -> dict[str, float]:
        """Read the actual arm coordinate from the independent physical model."""
        return {
            MeasurementFixture.HINGE: float(
                self.data.qpos[self.model.joint(MeasurementFixture.HINGE).qposadr[0]]
            )
        }


def test_portable_measurement_updates_real_native_coordinates(
    cylinder_bot_world: World,
):
    world = cylinder_bot_world
    physics = DetachedArticulation()
    robot = world.get_body_by_name(MeasurementFixture.ROBOT)
    with world.modify_world():
        arm = Body(name=PrefixedName(MeasurementFixture.ARM))
        hinge = RevoluteConnection.create_with_dofs(
            world=world,
            parent=robot,
            child=arm,
            name=PrefixedName(MeasurementFixture.HINGE),
            axis=Vector3.Y(),
        )
        world.add_connection(hinge)
        sample = Body(name=PrefixedName(MeasurementFixture.SAMPLE))
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world, parent=world.root, child=sample
            )
        )
    before_physics = physics.data.qpos.copy()
    measurement = LaboratoryPlanWorld.capture(physics)

    measurement.apply(world)

    assert hinge.position == physics.robot_state()[MeasurementFixture.HINGE]
    for body in (robot, sample):
        np.testing.assert_allclose(
            body.global_pose.to_position().to_np()[:3],
            physics.data.xpos[physics.model.body(str(body.name)).id],
            atol=1e-12,
        )
    np.testing.assert_array_equal(physics.data.qpos, before_physics)


# %% authoritative measurements
def test_capture_does_not_modify_physical_positions_or_commands(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    before_positions = mobile_robot.data.qpos.copy()
    before_controls = mobile_robot.data.ctrl.copy()

    measurement = LaboratoryPlanWorld.capture(mobile_robot)

    np.testing.assert_array_equal(mobile_robot.data.qpos, before_positions)
    np.testing.assert_array_equal(mobile_robot.data.ctrl, before_controls)
    assert measurement.simulation_time == mobile_robot.data.time
    with pytest.raises(FrozenInstanceError):
        measurement.simulation_time = 1.0


def test_semantic_robot_base_head_and_samples_match_detached_measurements(
    mobile_robot: LaboratoryRobotPhysics,
    laboratory_bundle,
) -> None:
    world = LaboratoryWorld(bundle_directory=laboratory_bundle).build()
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    mobile_robot.set_joint_targets(
        {
            RobotBaseJoint.HORIZONTAL: -0.1,
            RobotHeadJoint.PAN: 0.25,
            RobotHeadJoint.TILT: 0.55,
        }
    )
    mobile_robot.step(500)
    measurement = LaboratoryPlanWorld.capture(mobile_robot)
    captured_positions = mobile_robot.data.qpos.copy()
    captured_controls = mobile_robot.data.ctrl.copy()
    captured_poses = {pose.body_name: pose for pose in measurement.poses}
    captured_joints = {joint.joint_name: joint.position for joint in measurement.joints}
    collisions = {body.id: body.collision for body in world.bodies}

    measurement.apply(world)

    for name in RobotHeadJoint:
        prefix, local_name = name.split("/", 1)
        connection = world.get_connection_by_name(PrefixedName(local_name, prefix))
        assert connection.position == captured_joints[name]
    for body in (robot.root, world.get_body_by_name(LaboratoryBody.CLEAR_TUBE)):
        actual = captured_poses[str(body.name)]
        expected = Pose.from_xyz_quaternion(*actual.position, *actual.orientation)
        np.testing.assert_allclose(
            body.global_pose.to_np(), expected.to_np(), atol=1e-10
        )
    assert all(body.collision is collisions[body.id] for body in world.bodies)
    np.testing.assert_array_equal(mobile_robot.data.qpos, captured_positions)
    np.testing.assert_array_equal(mobile_robot.data.ctrl, captured_controls)


def test_semantic_instrument_angles_use_captured_state_after_physics_advances(
    mobile_robot: LaboratoryRobotPhysics,
    laboratory_bundle,
) -> None:
    world = LaboratoryWorld(bundle_directory=laboratory_bundle).build()
    mobile_robot.centrifuge_command(CentrifugeAction.CLOSE)
    mobile_robot.refrigerator_command(RefrigeratorAction.OPEN)
    mobile_robot.step(200)
    measurement = LaboratoryPlanWorld.capture(mobile_robot)
    captured = {joint.joint_name: joint.position for joint in measurement.joints}
    mobile_robot.step(200)

    measurement.apply(world)

    for name in (CentrifugeGeometry.LID_JOINT, RefrigeratorGeometry.JOINT):
        assert world.get_connection_by_name(name).position == captured[name]
        physical_joint = mobile_robot.model.joint(name)
        assert captured[name] != float(
            mobile_robot.data.qpos[physical_joint.qposadr[0]]
        )
