"""Independent sample-rack dynamics, measured slots and contact support."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import Any

from cramera.laboratory_drawer import DrawerGeometry


# %% authored geometry and measured state
class RackGeometry(StrEnum):
    """Authored names of the independently movable rack."""

    BODY = "laboratory_rack"
    """Rigid carrier supporting separate sample bodies."""
    JOINT = "laboratory_rack_joint"
    """Six-degree-of-freedom carrier motion."""
    HANDLE = "laboratory_rack_rack_handle"
    """Contact geometry of the front grasp bar."""
    DRAWER_FLOOR = "laboratory_drawer_drawer_bottom"
    """Upward support surface used during explicit drawer preparation."""


class RackField(StrEnum):
    """Rack metadata and observations shared by preparation and visualization."""

    RACK = "rack"
    """Top-level measured rack observation."""
    BODY_POSES = "bodyPoses"
    """Measured world poses of free bodies authored inside the environment."""
    DRAWER_OFFSET = "drawerOffset"
    """Authored rack offset from the drawer floor centre."""
    POSE = "pose"
    """Measured world position and XYZW quaternion."""
    SUPPORTED = "supported"
    """Whether upward external contacts support the carrier weight."""
    SUPPORT_BODIES = "supportBodies"
    """Bodies supplying measured upward rack support."""
    SLOTS = "slots"
    """Measured insertion poses and contact-supported occupants."""
    RACK_SLOTS = "rackSlots"
    """Authored insertion poses at the initial workbench placement."""
    ID = "id"
    """Authored slot identifier."""
    KEY = "key"
    """Independent sample identifier, or None for an empty slot."""
    SLOT = "slot"
    """Authored initial sample slot."""
    TUBES = "tubes"
    """Initial assignments of independent sample bodies."""


@dataclass(frozen=True)
class RackParameters:
    """Measured support and insertion tolerances for the sample carrier."""

    depth: float = 0.11
    """Carrier base depth in metres."""
    position_tolerance: float = 0.008
    """Maximum tube-root distance from a measured slot in metres."""
    upright_tolerance: float = 0.15
    """Maximum sample tilt relative to the carrier in radians."""
    support_fraction: float = 0.2
    """Minimum weight fraction carried by upward support contacts."""


# %% free-body model
@dataclass
class RackCollisionModel:
    """Import a floating carrier without attaching its individual samples."""

    @staticmethod
    def dynamic(joint: ElementTree.Element | None) -> bool:
        """Whether the authored rack explicitly requests floating motion."""
        return (
            joint is not None
            and joint.attrib["name"] == RackGeometry.JOINT
            and joint.attrib["type"] == "floating"
        )

    @staticmethod
    def world_pose(
        name: str, children: dict[str, ElementTree.Element]
    ) -> dict[str, str]:
        """Compose fixed ancestors when promoting a floating rack to the world body."""
        world_T_rack = np.eye(4)
        while name in children:
            joint = children[name]
            origin = joint.find("origin")
            parent_T_child = np.eye(4)
            if origin is not None:
                parent_T_child[:3, 3] = np.fromstring(
                    origin.attrib.get("xyz", "0 0 0"), sep=" "
                )
                parent_T_child[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            world_T_rack = parent_T_child @ world_T_rack
            name = joint.find("parent").attrib["link"]
        return {
            "pos": " ".join(map(str, world_T_rack[:3, 3])),
            "quat": " ".join(
                map(
                    str,
                    Rotation.from_matrix(world_T_rack[:3, :3]).as_quat(
                        scalar_first=True
                    ),
                )
            ),
        }

    @staticmethod
    def add_joint(
        body: ElementTree.Element,
        link: ElementTree.Element,
        joint: ElementTree.Element | None,
    ) -> None:
        """Retain authored mass and inertia with an independent free joint."""
        if not RackCollisionModel.dynamic(joint):
            return
        inertial = link.find("inertial")
        inertia = inertial.find("inertia").attrib
        ElementTree.SubElement(
            body,
            "inertial",
            mass=inertial.find("mass").attrib["value"],
            fullinertia=" ".join(
                inertia[key] for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            ),
            pos=inertial.find("origin").attrib.get("xyz", "0 0 0"),
        )
        ElementTree.SubElement(body, "freejoint", name=RackGeometry.JOINT)


# %% measured support and preparation
@dataclass
class LaboratoryRack:
    """Report moving slots and supported samples on an independent carrier."""

    model: mujoco.MjModel
    """Compiled rack, drawer and sample contact model."""
    data: mujoco.MjData
    """Authoritative body poses and contact forces."""
    laboratory: dict[str, Any]
    """Authored slots, sample assignments and drawer preparation offset."""
    parameters: RackParameters = field(default_factory=RackParameters)
    """Measured support and occupancy tolerances."""
    body: int = field(init=False)
    """Rack body index."""
    joint: int = field(init=False)
    """Floating rack joint index, or minus one in older fixed-rack bundles."""
    local_positions: dict[str, np.ndarray] = field(init=False)
    """Authored tube-root positions expressed in the rack frame."""
    assignments: dict[str, str] = field(init=False)
    """Initial slot identifier for each separate sample body."""

    def __post_init__(self) -> None:
        """Express authored workbench insertion poses in the moving rack frame."""
        self.body = self.model.body(RackGeometry.BODY).id
        self.joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, RackGeometry.JOINT
        )
        initial_position = self.model.body_pos[self.body]
        initial_rotation = Rotation.from_quat(
            self.model.body_quat[self.body], scalar_first=True
        ).as_matrix()
        self.local_positions = {
            slot[RackField.ID]: initial_rotation.T
            @ (np.asarray(slot[RackField.POSE][:3]) - initial_position)
            for slot in self.laboratory[RackField.RACK_SLOTS]
        }
        self.assignments = {
            item[RackField.KEY]: item[RackField.SLOT]
            for item in self.laboratory[RackField.TUBES]
        }

    @property
    def movable(self) -> bool:
        """Whether the carrier has an independently integrated free joint."""
        return (
            self.joint >= 0
            and self.model.jnt_type[self.joint] == mujoco.mjtJoint.mjJNT_FREE
        )

    @property
    def pose(self) -> list[float]:
        """Measured world root position followed by the XYZW quaternion."""
        quaternion = self.data.xquat[self.body]
        return [*self.data.xpos[self.body], *quaternion[1:], quaternion[0]]

    def slot_position(self, name: str) -> np.ndarray:
        """Return the measured insertion position for a slot or its initial sample."""
        slot = self.assignments.get(name, name)
        return (
            self.data.xpos[self.body]
            + self.data.xmat[self.body].reshape(3, 3) @ self.local_positions[slot]
        )

    def support_forces(self, body: int) -> dict[int, float]:
        """Read upward normal forces acting on one body from surrounding contacts."""
        forces: dict[int, float] = {}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first, second = (
                int(self.model.geom_bodyid[geometry])
                for geometry in (contact.geom1, contact.geom2)
            )
            if body not in (first, second) or contact.efc_address < 0:
                continue
            effort = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, effort)
            direction = 1.0 if second == body else -1.0
            force = max(0.0, direction * float(contact.frame[2]) * float(effort[0]))
            support = first if second == body else second
            forces[support] = forces.get(support, 0.0) + force
        return forces

    def supported_by(self, name: str) -> bool:
        """Require upward contacts from the named body carrying carrier weight."""
        support = self.model.body(name).id
        weight = self.model.body_mass[self.body] * np.linalg.norm(
            self.model.opt.gravity
        )
        return self.support_forces(self.body).get(support, 0.0) > (
            weight * self.parameters.support_fraction
        )

    def occupants(self) -> dict[str, str | None]:
        """Match independent upright samples to contact-supported moving slots."""
        occupants = dict.fromkeys(self.local_positions)
        rack_rotation = self.data.xmat[self.body].reshape(3, 3)
        for key in self.assignments:
            body = self.model.body(key).id
            sample_rotation = self.data.xmat[body].reshape(3, 3)
            if (rack_rotation.T @ sample_rotation)[2, 2] < math.cos(
                self.parameters.upright_tolerance
            ):
                continue
            weight = self.model.body_mass[body] * np.linalg.norm(self.model.opt.gravity)
            if (
                self.support_forces(body).get(self.body, 0.0)
                <= weight * self.parameters.support_fraction
            ):
                continue
            for slot in occupants:
                if np.linalg.norm(self.data.xpos[body] - self.slot_position(slot)) <= (
                    self.parameters.position_tolerance
                ):
                    occupants[slot] = key
                    break
        return occupants

    def prepare_in_drawer(self) -> None:
        """Initialize separate rack and sample coordinates for an explicit demo reset."""
        if not self.movable:
            raise ValueError("The authored rack has no floating joint")
        drawer = self.model.body(DrawerGeometry.BODY).id
        floor = self.model.geom(RackGeometry.DRAWER_FLOOR).id
        drawer_rotation = self.data.xmat[drawer].reshape(3, 3)
        position = self.data.geom_xpos[floor] + drawer_rotation @ (
            np.asarray(self.laboratory[RackField.RACK][RackField.DRAWER_OFFSET])
            + [0.0, 0.0, self.model.geom_size[floor, 2]]
        )
        quaternion = self.data.xquat[drawer]
        self._initialize_body(self.body, position, quaternion)
        for key, slot in self.assignments.items():
            self._initialize_body(
                self.model.body(key).id,
                position + drawer_rotation @ self.local_positions[slot],
                quaternion,
            )
        mujoco.mj_forward(self.model, self.data)

    def _initialize_body(
        self, body: int, position: np.ndarray, quaternion: np.ndarray
    ) -> None:
        """Set one free body's initial coordinates and clear its velocity at reset."""
        joint = self.model.body_jntadr[body]
        coordinate = self.model.jnt_qposadr[joint]
        velocity = self.model.jnt_dofadr[joint]
        self.data.qpos[coordinate : coordinate + 3] = position
        self.data.qpos[coordinate + 3 : coordinate + 7] = quaternion
        self.data.qvel[velocity : velocity + 6] = 0.0

    def snapshot(self) -> dict[str, Any]:
        """Publish measured carrier placement and physically supported slot contents."""
        forces = self.support_forces(self.body)
        weight = self.model.body_mass[self.body] * np.linalg.norm(
            self.model.opt.gravity
        )
        supports = [
            self.model.body(body).name
            for body, force in forces.items()
            if force > weight * self.parameters.support_fraction
        ]
        occupants = self.occupants()
        return {
            RackField.POSE: self.pose,
            RackField.SUPPORTED: bool(supports),
            RackField.SUPPORT_BODIES: supports,
            RackField.SLOTS: [
                {
                    RackField.ID: slot,
                    RackField.POSE: [*self.slot_position(slot), *self.pose[3:]],
                    RackField.KEY: occupants[slot],
                }
                for slot in self.local_positions
            ],
        }
