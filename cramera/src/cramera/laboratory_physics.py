"""Rigid contact dynamics for the authored laboratory's loose objects.

The collision model retains the bundle's square rack apertures and compound glass
walls. A reduced free-surface model follows measured tube poses and contributes
contained liquid mass; liquid pressure and sloshing torques are not coupled.
Visual meshes remain owned by CRAMERA.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import Any

from cramera.laboratory_centrifuge import CentrifugeCollisionModel, LaboratoryCentrifuge
from cramera.laboratory_centrifuge_command import CentrifugeAction, CentrifugeField
from cramera.laboratory_liquid import LaboratoryLiquid
from cramera.laboratory_mass import LaboratoryLiquidMass
from cramera.laboratory_pipetting import (
    InvalidPipettingCommand,
    LaboratoryPipetting,
    PipettingField,
    PipettingState,
)
from cramera.laboratory_refrigerator import (
    RefrigeratorCollisionModel,
    LaboratoryRefrigerator,
)
from cramera.laboratory_refrigerator_command import (
    RefrigeratorAction,
    RefrigeratorField,
)
from cramera.laboratory_scale import LaboratoryScale, ScaleGeometry
from cramera.laboratory_world import (
    CabinetStructure,
    LaboratoryAsset,
    LaboratoryBody,
    LaboratoryWorld,
)


# %% public state
class PhysicsField(StrEnum):
    """Fields of a contact simulation snapshot."""

    OBJECTS = "objects"
    """Current object poses in meters and XYZW quaternion order."""
    TARGET = "target"
    """Current compliant drag target, or None."""
    KEY = "key"
    """Authored loose-object identifier."""
    POSITION = "position"
    """World coordinates in the laboratory's Z-up frame."""
    ORIENTATION = "orientation"
    """Desired orientation as an XYZW quaternion."""
    LIQUID = "liquid"
    """Authoritative reduced free-surface volumes and escaped droplets."""
    SCALE = "scale"
    """Measured weighing-pan contact load and tare state."""
    CONTACTS = "contacts"
    """Current active collision contacts."""
    NORMAL = "normal"
    """World contact normal from the first geometry toward the second."""
    FORCE = "force"
    """Contact normal force in Newtons."""
    BODY_A = "bodyA"
    """First contacting body's authored name."""
    BODY_B = "bodyB"
    """Second contacting body's authored name."""
    TIME = "time"
    """Simulated time since reset, in seconds."""
    ENERGY = "energy"
    """Combined kinetic and potential energy, in Joules."""
    APPLIED_FORCE = "appliedForce"
    """Current drag force magnitude in Newtons."""
    MAXIMUM_APPLIED_FORCE = "maximumAppliedForce"
    """Largest applied drag force magnitude since reset."""
    MAXIMUM_CONTACT_FORCE = "maximumContactForce"
    """Largest individual contact normal force since reset."""


@dataclass(frozen=True)
class LaboratoryPhysicsParameters:
    """Numerical contact and compliant mouse-grasp parameters."""

    timestep: float = 0.001
    """Physics integration period in seconds."""
    stiffness: float = 12.0
    """Translation spring stiffness in Newtons per meter."""
    damping: float = 0.8
    """Translation damping in Newton seconds per meter."""
    maximum_force: float = 0.6
    """Maximum resultant mouse-grasp force, including gravity compensation."""
    angular_stiffness: float = 0.04
    """Upright orientation spring stiffness in Newton meters per radian."""
    angular_damping: float = 0.0016
    """Orientation damping in Newton meter seconds per radian."""
    maximum_torque: float = 0.02
    """Maximum mouse-grasp torque in Newton meters."""
    contact_time_constant: float = 0.004
    """Contact constraint response time in seconds."""
    scale_contact_time_constant: float = 0.008
    """Compliant pan and holder response avoiding unstable friction corrections."""
    scale_contact_priority: int = 1
    """Priority retaining the scale response when a vessel touches its surface."""
    liquid_timestep: float = 0.02
    """Interval between liquid updates from measured rigid-body poses, in seconds."""


@dataclass(frozen=True)
class PhysicsTarget:
    """One selected object and its desired force-driven world pose."""

    key: str
    """Authored loose-object name."""
    position: tuple[float, float, float]
    """Desired object-root position in meters."""
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Desired normalized XYZW orientation; defaults to upright."""


@dataclass
class InvalidPhysicsTarget(ValueError):
    """A drag target cannot be applied to the contact simulation."""

    reason: str
    """Description of the invalid object or coordinate input."""

    def __str__(self) -> str:
        """Present the request validation failure to an HTTP caller."""
        return self.reason


@dataclass(frozen=True)
class DynamicLaboratoryObject:
    """Compiled indices for one free object."""

    key: str
    """Authored loose-object identifier."""
    body_id: int
    """MuJoCo body index."""


# %% collision-only scene conversion
@dataclass
class LaboratoryCollisionModel:
    """Build in-memory MJCF from authored collision elements and inertials."""

    directory: Path
    """Root directory of the authored laboratory bundle."""
    scene: dict[str, Any]
    """Authored scene manifest defining movable objects and initial poses."""
    parameters: LaboratoryPhysicsParameters
    """Numerical integration and contact settings."""
    root: ElementTree.Element = field(init=False)
    """Generated MJCF root."""
    assets: ElementTree.Element = field(init=False)
    """Collision mesh declarations."""
    world: ElementTree.Element = field(init=False)
    """Top-level MJCF body container."""

    def build(self) -> mujoco.MjModel:
        """Compile contacts without importing the scene's visual meshes."""
        self.root = ElementTree.Element("mujoco", model="laboratory_contact")
        ElementTree.SubElement(
            self.root, "compiler", angle="radian", inertiafromgeom="false"
        )
        option = ElementTree.SubElement(
            self.root,
            "option",
            timestep=str(self.parameters.timestep),
            integrator="implicitfast",
            cone="elliptic",
            iterations="80",
        )
        ElementTree.SubElement(option, "flag", energy="enable")
        defaults = ElementTree.SubElement(self.root, "default")
        ElementTree.SubElement(
            defaults,
            "geom",
            friction="0.4 0.005 0.0001",
            condim="4",
            solref=f"{self.parameters.contact_time_constant} 1",
            solimp="0.95 0.99 0.0001",
        )
        self.assets = ElementTree.SubElement(self.root, "asset")
        self.world = ElementTree.SubElement(self.root, "worldbody")
        ElementTree.SubElement(
            self.world, "geom", name="floor", type="plane", size="3 3 0.1"
        )
        for item in self.scene["models"]:
            if not item["robot"]:
                self._add_environment(self.directory / item["urdf"])
        self._add_cabinet()
        for item in self.scene["objects"]:
            self._add_object(item)
        CentrifugeCollisionModel(
            self.root,
            tuple(item["key"] for item in self.scene["laboratory"]["tubes"]),
        ).complete()
        RefrigeratorCollisionModel(self.root).complete()
        return mujoco.MjModel.from_xml_string(
            ElementTree.tostring(self.root, encoding="unicode")
        )

    def _add_cabinet(self) -> None:
        """Match visible closed cabinet panels to the semantic collision world."""
        cabinet = ElementTree.SubElement(
            self.world, "body", name=LaboratoryBody.CABINET
        )
        for index, panel in enumerate(CabinetStructure().panels()):
            ElementTree.SubElement(
                cabinet,
                "geom",
                name=f"{LaboratoryBody.CABINET}_{index}",
                type="box",
                pos=" ".join(map(str, panel.origin.to_np()[:3, 3])),
                size=" ".join(map(str, panel.size.to_np() / 2)),
            )

    @staticmethod
    def _pose(origin: ElementTree.Element | None) -> dict[str, str]:
        """Translate an optional URDF pose to MJCF's quaternion convention."""
        if origin is None:
            return {}
        result = {"pos": origin.attrib.get("xyz", "0 0 0")}
        angles = [float(value) for value in origin.attrib.get("rpy", "0 0 0").split()]
        quaternion = Rotation.from_euler("xyz", angles).as_quat(scalar_first=True)
        result["quat"] = " ".join(map(str, quaternion))
        return result

    def _add_environment(self, description: Path) -> None:
        """Import the fixed environment and its explicitly supported instrument joints."""
        robot = ElementTree.parse(description).getroot()
        links = {link.attrib["name"]: link for link in robot.findall("link")}
        children = {
            joint.find("child").attrib["link"]: joint
            for joint in robot.findall("joint")
        }
        for name in links.keys() - children.keys():
            self._add_environment_link(name, links, children, self.world, description)

    def _add_environment_link(
        self,
        name: str,
        links: dict[str, ElementTree.Element],
        children: dict[str, ElementTree.Element],
        parent: ElementTree.Element,
        description: Path,
    ) -> None:
        """Preserve named contact bodies and the instrument's physical hinges."""
        joint = children.get(name)
        pose = self._pose(joint.find("origin") if joint is not None else None)
        descendants = [
            child
            for child, child_joint in children.items()
            if child_joint.find("parent").attrib["link"] == name
        ]
        if (
            joint is not None
            and joint.attrib["type"] == "fixed"
            and len(links[name]) == 0
            and not descendants
        ):
            # Annotation frames must not acquire the robot compiler's minimum mass.
            ElementTree.SubElement(
                parent, "site", name=name, size="0.001", rgba="0 0 0 0", **pose
            )
            return
        body = ElementTree.SubElement(parent, "body", name=name, **pose)
        self._add_collisions(body, links[name], description)
        CentrifugeCollisionModel.add_joint(body, links[name], joint)
        RefrigeratorCollisionModel.add_joint(body, links[name], joint)
        for child in descendants:
            self._add_environment_link(child, links, children, body, description)

    def _add_object(self, item: dict[str, Any]) -> None:
        """Give an authored loose object its own six-degree-of-freedom joint."""
        description = self.directory / item["urdf"]
        [link] = ElementTree.parse(description).getroot().findall("link")
        pose = item["spawn"]
        body = ElementTree.SubElement(
            self.world,
            "body",
            name=item["key"],
            pos=" ".join(map(str, pose[:3])),
            quat=" ".join(map(str, [pose[6], *pose[3:6]])),
        )
        ElementTree.SubElement(body, "freejoint", name=f"{item['key']}_free")
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
        self._add_collisions(body, link, description)

    def _add_collisions(
        self,
        body: ElementTree.Element,
        link: ElementTree.Element,
        description: Path,
    ) -> None:
        """Transfer primitive and convex mesh collision pieces independently."""
        for collision in link.findall("collision"):
            [geometry] = list(collision.find("geometry"))
            name = f"{link.attrib['name']}_{collision.attrib['name']}"
            properties = self._pose(collision.find("origin"))
            if name == ScaleGeometry.PAN or name.startswith(
                ScaleGeometry.HOLDER_PREFIX
            ):
                properties.update(
                    priority=str(self.parameters.scale_contact_priority),
                    solref=f"{self.parameters.scale_contact_time_constant} 1",
                )
            if geometry.tag == "box":
                properties.update(
                    type="box",
                    size=" ".join(
                        str(float(value) / 2)
                        for value in geometry.attrib["size"].split()
                    ),
                )
            elif geometry.tag == "cylinder":
                properties.update(
                    type="cylinder",
                    size=f"{geometry.attrib['radius']} {float(geometry.attrib['length']) / 2}",
                )
            elif geometry.tag == "sphere":
                properties.update(type="sphere", size=geometry.attrib["radius"])
            elif geometry.tag == "mesh":
                filename = str(
                    (description.parent / geometry.attrib["filename"]).resolve()
                )
                ElementTree.SubElement(
                    self.assets,
                    "mesh",
                    name=name,
                    file=filename,
                    scale=geometry.attrib.get("scale", "1 1 1"),
                )
                properties.update(type="mesh", mesh=name)
            else:
                raise ValueError(
                    f"Unsupported laboratory collision geometry: {geometry.tag}"
                )
            ElementTree.SubElement(body, "geom", name=name, **properties)


# %% force-driven interaction
@dataclass
class LaboratoryPhysics:
    """Advance laboratory contacts and a bounded compliant cursor grasp.

    The owner serializes calls to this object. Only reset writes free-joint poses;
    dragging applies forces and receives positions from MuJoCo's contact solver.
    """

    bundle_directory: Path | None = None
    """Authored scene directory, resolved locally when omitted."""
    parameters: LaboratoryPhysicsParameters = field(
        default_factory=LaboratoryPhysicsParameters
    )
    """Physical integration and cursor spring settings."""
    scene: dict[str, Any] = field(init=False)
    """Original manifest, available for preserving authored presentation."""
    model: mujoco.MjModel = field(init=False, repr=False)
    """Compiled collision-only model."""
    data: mujoco.MjData = field(init=False, repr=False)
    """Authoritative dynamic state."""
    objects: dict[str, DynamicLaboratoryObject] = field(init=False)
    """Movable objects indexed by their authored identifiers."""
    target: PhysicsTarget | None = field(init=False, default=None)
    """Active cursor grasp, if any."""
    applied_force: float = field(init=False, default=0.0)
    """Current applied cursor force magnitude."""
    maximum_applied_force: float = field(init=False, default=0.0)
    """Peak applied cursor force since reset."""
    maximum_contact_force: float = field(init=False, default=0.0)
    """Peak individual normal contact force since reset."""
    liquid: LaboratoryLiquid = field(init=False)
    """Contained and spilled volumes advanced from observed glass motion."""
    liquid_elapsed: float = field(init=False, default=0.0)
    """Physics time accumulated since the last liquid update."""
    liquid_mass: LaboratoryLiquidMass = field(init=False)
    """Contained liquid coupled into rigid-body mass and inertia."""
    scale: LaboratoryScale = field(init=False)
    """Contact-force balance measurement and deliberate tare."""
    centrifuge: LaboratoryCentrifuge = field(init=False)
    """Articulated rotor, measured holder clamps and operating interlocks."""

    refrigerator: LaboratoryRefrigerator = field(init=False)
    """Physical cabinet door, contact-grounded inventory and reduced cooling."""

    pipetting: LaboratoryPipetting = field(init=False)
    """Finite reagent delivery with measured vessel alignment interlocks."""

    def __post_init__(self) -> None:
        """Compile authored collisions and initialize gravity-driven objects."""
        self.bundle_directory = (
            LaboratoryWorld.resolve_directory()
            if self.bundle_directory is None
            else Path(self.bundle_directory).expanduser().resolve()
        )
        self.scene = json.loads(
            (self.bundle_directory / LaboratoryAsset.SCENE).read_text()
        )
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.objects = {
            item["key"]: DynamicLaboratoryObject(
                item["key"], self.model.body(item["key"]).id
            )
            for item in self.scene["objects"]
        }
        self.liquid = LaboratoryLiquid(self.scene)
        self.pipetting = LaboratoryPipetting(self.scene, self.liquid)
        self.liquid_mass = LaboratoryLiquidMass(
            self.model,
            {key: self.objects[key].body_id for key in self.liquid.tubes},
        )
        self.scale = LaboratoryScale(self.model, self.data)
        self.centrifuge = LaboratoryCentrifuge(
            self.model,
            self.data,
            self.scene["laboratory"].get(CentrifugeField.CENTRIFUGE, {}),
            tuple(self.liquid.tubes),
        )
        self.refrigerator = LaboratoryRefrigerator(
            self.model,
            self.data,
            self.scene["laboratory"].get(RefrigeratorField.REFRIGERATOR, {}),
            tuple(self.liquid.tubes),
        )
        self.reset()

    def _build_model(self) -> mujoco.MjModel:
        """Compile the collision model for the configured laboratory."""
        return LaboratoryCollisionModel(
            self.bundle_directory, self.scene, self.parameters
        ).build()

    @property
    def timestep(self) -> float:
        """Integration period used by the owner's real-time scheduler."""
        return self.parameters.timestep

    def reset(self) -> None:
        """Restore authored spawn poses, zero velocity and clear force history."""
        mujoco.mj_resetData(self.model, self.data)
        self.target = None
        self.applied_force = 0.0
        self.maximum_applied_force = 0.0
        self.maximum_contact_force = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.liquid_elapsed = 0.0
        self.liquid.reset(self._object_poses())
        self.liquid_mass.synchronize(self.liquid)
        self.centrifuge.reset()
        self.pipetting.reset()
        self.refrigerator.reset()
        mujoco.mj_forward(self.model, self.data)
        self.scale.reset()

    def set_target(
        self, key: str, position: list[float], orientation: list[float] | None = None
    ) -> None:
        """Select a loose object and set its bounded force and torque target."""
        if key not in self.objects:
            raise InvalidPhysicsTarget(f"Unknown movable laboratory object: {key}")
        self.centrifuge.validate_manipulation(key)
        self.refrigerator.validate_manipulation(key)
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise InvalidPhysicsTarget("Position must contain three finite numbers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in position
        ):
            raise InvalidPhysicsTarget("Position must contain three finite numbers")
        quaternion = (0.0, 0.0, 0.0, 1.0)
        if orientation is not None:
            if (
                not isinstance(orientation, (list, tuple))
                or len(orientation) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in orientation
                )
            ):
                raise InvalidPhysicsTarget(
                    "Orientation must contain four finite numbers"
                )
            length = math.hypot(*orientation)
            if not math.isfinite(length) or length < 1e-8:
                raise InvalidPhysicsTarget(
                    "Orientation quaternion must have nonzero length"
                )
            quaternion = tuple(float(value) / length for value in orientation)
        self.target = PhysicsTarget(
            key, tuple(float(value) for value in position), quaternion
        )

    def fill_liquid(self, key: str, volume_ml: float) -> None:
        """Set a tube's liquid quantity without changing its physical pose."""
        self.centrifuge.validate_manipulation(key)
        self.refrigerator.validate_manipulation(key)
        if (
            self.pipetting.state == PipettingState.DISPENSING
            and self.pipetting.key == key
        ):
            raise InvalidPipettingCommand(
                "Stop the active dose before changing the receiver liquid."
            )
        self.liquid.fill(key, volume_ml)
        self.liquid_mass.synchronize(self.liquid)
        mujoco.mj_forward(self.model, self.data)
        self.scale.observe(force=True)

    def pipetting_command(
        self,
        action: str,
        key: str | None = None,
        source: str | None = None,
        volume_ml: float | None = None,
    ) -> None:
        """Operate the stationary pump against the current measured receiver pose."""
        self.pipetting.command(
            action,
            self._object_poses(),
            self._secured_tubes(),
            key=key,
            source=source,
            volume_ml=volume_ml,
        )

    def _secured_tubes(self) -> frozenset[str]:
        """Return glasses currently owned by a centrifuge holder fixture."""
        return frozenset(
            slot.key for slot in self.centrifuge.slots.values() if slot.key is not None
        )

    def refrigerator_command(self, action: RefrigeratorAction | str) -> None:
        """Request bounded cabinet door motion without moving samples."""
        self.refrigerator.command(action)

    def tare_scale(self) -> None:
        """Zero the balance at its current settled load."""
        self.scale.tare()

    def read_centrifuge_manual(self, document: str) -> None:
        """Read operating rules without altering any object or actuator state."""
        self.centrifuge.read_centrifuge_manual(document)

    def centrifuge_command(
        self,
        action: str,
        slot: str | None = None,
        key: str | None = None,
        rpm: float | None = None,
        duration: float | None = None,
    ) -> None:
        """Operate the physical centrifuge after checking its measured interlocks."""
        self.centrifuge.command(action, slot, key, rpm, duration)
        if (
            action == CentrifugeAction.SECURE
            and self.target is not None
            and self.target.key == key
        ):
            self.release()

    def release(self) -> None:
        """Remove the cursor grasp and let gravity and contact determine motion."""
        self.target = None
        self.data.xfrc_applied[:] = 0
        self.applied_force = 0.0

    @staticmethod
    def _limited(vector: np.ndarray, maximum: float) -> np.ndarray:
        """Limit resultant magnitude while preserving force or torque direction."""
        magnitude = np.linalg.norm(vector)
        return vector * min(1.0, maximum / magnitude) if magnitude else vector

    def _apply_grasp(self) -> None:
        """Apply damped, bounded forces and torques toward the desired pose."""
        self.data.xfrc_applied[:] = 0
        self.applied_force = 0.0
        if self.target is None:
            return
        body = self.objects[self.target.key]
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body.body_id, velocity, 0
        )
        force = (
            self.parameters.stiffness
            * (np.asarray(self.target.position) - self.data.xpos[body.body_id])
            - self.parameters.damping * velocity[3:]
            - self.model.body_mass[body.body_id] * self.model.opt.gravity
        )
        force = self._limited(force, self.parameters.maximum_force)
        rotation = (
            Rotation.from_quat(self.target.orientation)
            * Rotation.from_quat(self.data.xquat[body.body_id], scalar_first=True).inv()
        ).as_rotvec()
        torque = (
            self.parameters.angular_stiffness * rotation
            - self.parameters.angular_damping * velocity[:3]
        )
        inertial_rotation = self.data.ximat[body.body_id].reshape(3, 3)
        inertia = self.model.body_inertia[body.body_id]
        # Integrate the cursor's angular spring and damping implicitly in the
        # principal frame so small axial moments remain dissipative.
        response = inertia / (
            inertia
            + self.timestep * self.parameters.angular_damping
            + self.timestep**2 * self.parameters.angular_stiffness
        )
        torque = inertial_rotation @ (response * (inertial_rotation.T @ torque))
        self.data.xfrc_applied[body.body_id, :3] = force
        self.data.xfrc_applied[body.body_id, 3:] = self._limited(
            torque, self.parameters.maximum_torque
        )
        self.applied_force = min(
            float(np.linalg.norm(force)), self.parameters.maximum_force
        )
        self.maximum_applied_force = max(self.maximum_applied_force, self.applied_force)

    def step(self, count: int = 1) -> None:
        """Integrate a fixed number of real contact-physics steps."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Physics step count must be a positive integer")
        for _ in range(count):
            self._apply_grasp()
            self.centrifuge.advance(self.timestep)
            self.refrigerator.advance(self.timestep)
            mujoco.mj_step(self.model, self.data)
            self._observe_contacts()
            self.liquid_elapsed += self.timestep
            if self.liquid_elapsed + 1e-12 >= self.parameters.liquid_timestep:
                poses = self._object_poses()
                self.liquid.step(self.liquid_elapsed, poses)
                self.pipetting.advance(
                    self.liquid_elapsed, poses, self._secured_tubes()
                )
                if self.liquid_mass.synchronize(self.liquid):
                    mujoco.mj_forward(self.model, self.data)
                self.liquid_elapsed = 0.0
            self.scale.observe()
        mujoco.mj_forward(self.model, self.data)

    def _observe_contacts(self) -> list[dict[str, Any]]:
        """Read actual solver contact forces with authored body names."""
        contacts = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.efc_address < 0:
                continue
            effort = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, effort)
            force = max(0.0, float(effort[0]))
            if force <= 0:
                continue
            self.maximum_contact_force = max(self.maximum_contact_force, force)
            contacts.append(
                {
                    PhysicsField.POSITION: contact.pos.tolist(),
                    PhysicsField.NORMAL: contact.frame[:3].tolist(),
                    PhysicsField.FORCE: force,
                    PhysicsField.BODY_A: self.model.body(
                        self.model.geom_bodyid[contact.geom1]
                    ).name,
                    PhysicsField.BODY_B: self.model.body(
                        self.model.geom_bodyid[contact.geom2]
                    ).name,
                }
            )
        return contacts

    def _object_poses(self) -> dict[str, list[float]]:
        """Read measured loose-object transforms in the laboratory frame."""
        objects = {}
        for key, body in self.objects.items():
            quaternion = self.data.xquat[body.body_id]
            objects[key] = [
                *self.data.xpos[body.body_id].tolist(),
                *quaternion[1:].tolist(),
                float(quaternion[0]),
            ]
        return objects

    def snapshot(self) -> dict[str, Any]:
        """Return authoritative body poses, liquid state, contacts and cursor feedback."""
        return {
            PhysicsField.OBJECTS: self._object_poses(),
            PhysicsField.LIQUID: self.liquid.snapshot(),
            PipettingField.PIPETTING: self.pipetting.snapshot(),
            PhysicsField.SCALE: self.scale.snapshot(),
            CentrifugeField.CENTRIFUGE: self.centrifuge.snapshot(),
            RefrigeratorField.REFRIGERATOR: self.refrigerator.snapshot(),
            PhysicsField.TARGET: (
                None
                if self.target is None
                else {
                    PhysicsField.KEY: self.target.key,
                    PhysicsField.POSITION: list(self.target.position),
                    PhysicsField.ORIENTATION: list(self.target.orientation),
                }
            ),
            PhysicsField.CONTACTS: self._observe_contacts(),
            PhysicsField.TIME: float(self.data.time),
            PhysicsField.ENERGY: float(np.sum(self.data.energy)),
            PhysicsField.APPLIED_FORCE: self.applied_force,
            PhysicsField.MAXIMUM_APPLIED_FORCE: self.maximum_applied_force,
            PhysicsField.MAXIMUM_CONTACT_FORCE: self.maximum_contact_force,
        }
