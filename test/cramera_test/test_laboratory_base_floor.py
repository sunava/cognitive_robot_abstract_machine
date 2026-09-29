"""Authored floor support does not disable obstacles in the same room body."""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np
import pytest
from typing_extensions import Any

from cramera.laboratory_physics_robot import RobotBaseJoint
from cramera.laboratory_robot_base import LaboratoryBasePath, UnsafeBasePath


# %% floor and wall share one authored room shell
class RoomGeometry(StrEnum):
    """Collision names separating ground support from room obstructions."""

    FLOOR = "room_floor"
    """Horizontal support whose top is the navigation plane."""
    WALL = "room_wall"
    """Vertical wall belonging to the same body as the floor."""
    ROBOT = "robot_chassis"
    """Robot collider whose lower surface rests on the floor."""


@dataclass(frozen=True)
class MeasuredBaseJoint:
    """Position coordinate required by swept base interpolation."""

    position_index: int
    """MuJoCo scalar coordinate of the planar joint."""


@dataclass
class SupportedBaseWorld:
    """A measured chassis resting on a collidable finite floor slab."""

    model: mujoco.MjModel
    """Room, chassis and planar articulation compiled for collision checks."""
    data: mujoco.MjData = field(init=False)
    """Current measured pose, preserved by navigation queries."""
    robot_joints: dict[str, MeasuredBaseJoint] = field(init=False)
    """Position offsets of the physical planar base."""
    scene: dict[str, Any] = field(
        default_factory=lambda: {
            "navigation": {"supportGeometries": [RoomGeometry.FLOOR]}
        }
    )
    """Exact authored support geometry declarations."""

    def __post_init__(self) -> None:
        """Initialize the stationary chassis at the floor's support height."""
        self.data = mujoco.MjData(self.model)
        self.robot_joints = {
            name: MeasuredBaseJoint(
                int(self.model.jnt_qposadr[self.model.joint(name).id])
            )
            for name in RobotBaseJoint
        }
        mujoco.mj_forward(self.model, self.data)

    def robot_state(self) -> dict[str, float]:
        """Read the three measured planar coordinates."""
        return {
            name: float(self.data.qpos[joint.position_index])
            for name, joint in self.robot_joints.items()
        }


@pytest.fixture
def supported_base() -> SupportedBaseWorld:
    """Build a finite floor slab and an obstructing wall on the same room body."""
    root = ElementTree.Element("mujoco")
    world = ElementTree.SubElement(root, "worldbody")
    room = ElementTree.SubElement(world, "body", name="room_shell")
    ElementTree.SubElement(
        room,
        "geom",
        name=RoomGeometry.FLOOR,
        type="box",
        pos="0 0 -0.04",
        size="3 3 0.04",
    )
    ElementTree.SubElement(
        room,
        "geom",
        name=RoomGeometry.WALL,
        type="box",
        pos="0.8 0 0.5",
        size="0.05 1 0.5",
    )
    mount = ElementTree.SubElement(world, "body", name="robot_mount")
    chassis = ElementTree.SubElement(mount, "body", name="pr2/base_link")
    for joint, axis, kind in (
        (RobotBaseJoint.HORIZONTAL, "1 0 0", "slide"),
        (RobotBaseJoint.DEPTH, "0 1 0", "slide"),
        (RobotBaseJoint.HEADING, "0 0 1", "hinge"),
    ):
        ElementTree.SubElement(chassis, "joint", name=joint, type=kind, axis=axis)
    ElementTree.SubElement(
        chassis,
        "geom",
        name=RoomGeometry.ROBOT,
        type="box",
        pos="0 0 0.05",
        size="0.1 0.1 0.05",
    )
    return SupportedBaseWorld(
        mujoco.MjModel.from_xml_string(ElementTree.tostring(root, encoding="unicode"))
    )


# %% precise support filtering
def test_finite_authored_floor_permits_normal_base_travel(
    supported_base: SupportedBaseWorld,
) -> None:
    """Ground contact must not be reported as an obstacle to planar navigation."""
    before = supported_base.data.qpos.copy()
    LaboratoryBasePath(supported_base).validate({RobotBaseJoint.HORIZONTAL: 0.3})
    np.testing.assert_array_equal(supported_base.data.qpos, before)


def test_wall_on_same_body_as_floor_still_blocks_base(
    supported_base: SupportedBaseWorld,
) -> None:
    """Floor designation never exempts the rest of its room shell from collision."""
    with pytest.raises(UnsafeBasePath) as failure:
        LaboratoryBasePath(supported_base).validate({RobotBaseJoint.HORIZONTAL: 1.0})
    assert failure.value.obstacle_geometry == RoomGeometry.WALL


def test_navigation_support_filter_preserves_actual_floor_contacts(
    supported_base: SupportedBaseWorld,
) -> None:
    """Support declarations affect planning only; MuJoCo still resolves floor contact."""
    before = supported_base.model.geom_contype.copy()
    LaboratoryBasePath(supported_base)
    np.testing.assert_array_equal(supported_base.model.geom_contype, before)
    pairs = {
        frozenset(
            (
                supported_base.model.geom(contact.geom1).name,
                supported_base.model.geom(contact.geom2).name,
            )
        )
        for contact in supported_base.data.contact[: supported_base.data.ncon]
    }
    assert frozenset((RoomGeometry.ROBOT, RoomGeometry.FLOOR)) in pairs
