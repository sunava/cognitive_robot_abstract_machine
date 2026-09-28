"""MuJoCo contacts and an ideal threaded joint for the portable workshop."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing_extensions import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# %% physical limits


class InvalidWorkshopCommand(ValueError):
    """A requested interaction violates the workshop command contract."""


class WorkshopSimulationFailure(RuntimeError):
    """The contact solver lost a finite, monotonically advancing state."""


class ThreadState(StrEnum):
    """Observable phases of the explicitly idealized thread joint."""

    FREE = "free"
    SEATED = "seated"
    DRIVING = "driving"
    STOPPED = "stopped"
    LIMIT = "limit"


@dataclass(frozen=True)
class WorkshopPhysicsParameters:
    """Contact integration and force limits for direct workshop manipulation."""

    timestep: float = 0.001
    """MuJoCo integration interval in seconds."""
    spring: float = 350.0
    """Translational grasp stiffness in newtons per metre."""
    maximum_force: float = 35.0
    """Maximum resultant virtual-grasp force in newtons."""
    angular_spring: float = 2.5
    """Rotational grasp stiffness in newton metres per radian."""
    maximum_torque: float = 1.2
    """Maximum virtual-grasp torque in newton metres."""
    contact_time: float = 0.004
    """Contact response time, kept longer than two integration steps."""
    friction: tuple[float, float, float] = (0.35, 0.005, 0.0001)
    """Sliding, torsional and rolling coefficients for rigid contacts."""
    drawer_spring: float = 800.0
    """Drawer target stiffness in newtons per metre."""
    drawer_force: float = 15.0
    """Maximum drawer motor force; obstructions can stop a drawer."""
    thread_spring: float = 2000.0
    """Thread position stiffness in newtons per metre."""
    thread_force: float = 20.0
    """Maximum thread constraint reaction force."""
    thread_angular_spring: float = 20.0
    """Pilot-axis rotational stiffness in newton metres per radian."""
    thread_torque: float = 1.2
    """Maximum passive rotational reaction at the ideal pilot constraint."""
    driver_speed: float = 2.0 * math.pi
    """Requested driver angular speed in radians per second."""
    maximum_contacts: int = 64
    """Maximum number of measured contact records in a state response."""
    blade_preload: float = 0.001
    """Desired blade compression against the slot floor, resolved by contacts."""
    contact_gap: float = 0.08
    """Maximum transient contact gap after a blade has physically engaged."""
    driver_distance: float = 0.006
    """Maximum blade displacement from the slot goal during engagement."""
    driver_angle: float = 0.15
    """Maximum driver shaft alignment error in radians."""
    pilot_distance: float = 0.002
    """Maximum radial shaft offset at the ideal thread entrance."""
    pilot_angle: float = 0.08
    """Maximum shaft tilt in radians when entering the pilot thread."""
    pilot_clearance: float = 0.0002
    """Initial screw-tip goal above the measured pilot mouth."""
    contact_margin: float = 0.0005
    """Maximum measured separation accepted as a tool-slot contact."""
    pilot_capture_depth: float = 0.004
    """Maximum screw-tip depth at which the ideal thread can first capture."""
    pilot_capture_height: float = 0.0005
    """Maximum screw-tip height above the pilot entrance during capture."""
    pilot_release_distance: float = 0.003
    """Radial displacement releasing a screw that is manually extracted."""
    pilot_release_height: float = 0.006
    """Axial displacement releasing a screw that is manually extracted."""


@dataclass(frozen=True)
class WorkshopThreadGeometry:
    """Authored dimensions and frames of the ideal screw-driving assembly."""

    assembly: str
    """Independent workpiece object key."""
    driver: str
    """Independent screwdriver object key."""
    mouth: np.ndarray
    """Pilot entrance expressed in the workpiece frame."""
    maximum_depth: float
    """Maximum axial insertion before the screw head seats."""
    slot_floor: float
    """Slot floor measured from the screw tip along its shaft."""
    bit_tip: np.ndarray
    """Blade contact point expressed in the tool frame."""
    drive_axis: np.ndarray
    """Positive driving axis expressed in the aligned tool frame."""
    pitch: float
    """Authored ideal thread advance per revolution."""
    screws: tuple[str, ...]
    """Independent screws eligible for the pilot joint."""

    @classmethod
    def from_json(cls, workshop: dict[str, Any]) -> WorkshopThreadGeometry:
        """Read geometry exclusively from the portable workshop manifest."""
        assembly, driver, screw = (
            workshop["assembly"],
            workshop["driver"],
            workshop["screw"],
        )
        return cls(
            assembly=assembly["key"],
            driver=driver["key"],
            mouth=np.asarray(assembly["localMouth"], dtype=float),
            maximum_depth=float(assembly["maximumDepth"]),
            slot_floor=float(screw["headTop"] - screw["slotSize"][2]),
            bit_tip=np.asarray(driver["bitTip"], dtype=float),
            drive_axis=np.asarray(driver["axis"], dtype=float),
            pitch=float(screw["threadPitch"]),
            screws=tuple(
                item["key"] for item in workshop["parts"] if item["kind"] == "screw"
            ),
        )


# %% collision descriptions


@dataclass
class WorkshopCollisionModel:
    """Convert portable URDF collision primitives into articulated MuJoCo bodies."""

    directory: Path
    """Directory containing the workshop scene and URDF descriptions."""
    parameters: WorkshopPhysicsParameters
    """Integration and contact coefficients."""
    scene: dict[str, Any]
    """Authored scene manifest."""

    @staticmethod
    def values(values: Any) -> str:
        """Format numerical vectors for an XML attribute."""
        return " ".join(str(float(value)) for value in values)

    @classmethod
    def origin(cls, element: ElementTree.Element | None) -> dict[str, str]:
        """Convert an optional URDF origin to a MuJoCo pose."""
        if element is None:
            return {}
        result = {"pos": element.get("xyz", "0 0 0")}
        angles = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        result["quat"] = cls.values(
            Rotation.from_euler("xyz", angles).as_quat(scalar_first=True)
        )
        return result

    def build(self) -> mujoco.MjModel:
        """Compile independent free bodies and bounded sliding drawers."""
        root = ElementTree.Element("mujoco", model="aicor_workshop_contact")
        ElementTree.SubElement(
            root, "compiler", angle="radian", inertiafromgeom="false"
        )
        ElementTree.SubElement(
            root,
            "option",
            timestep=str(self.parameters.timestep),
            integrator="implicitfast",
            cone="elliptic",
            iterations="80",
        )
        defaults = ElementTree.SubElement(root, "default")
        ElementTree.SubElement(
            defaults,
            "geom",
            friction=self.values(self.parameters.friction),
            condim="4",
            solref=f"{self.parameters.contact_time} 1",
            solimp=".95 .99 .0001",
        )
        world = ElementTree.SubElement(root, "worldbody")
        for model in self.scene["models"]:
            self.add_description(world, self.directory / model["urdf"])
        for item in self.scene["objects"]:
            self.add_description(world, self.directory / item["urdf"], item["spawn"])
        return mujoco.MjModel.from_xml_string(
            ElementTree.tostring(root, encoding="unicode")
        )

    def add_description(
        self, parent: ElementTree.Element, path: Path, pose: list[float] | None = None
    ) -> None:
        """Preserve a URDF hierarchy, with a free joint for a movable root."""
        description = ElementTree.parse(path).getroot()
        links = {link.attrib["name"]: link for link in description.findall("link")}
        joints = description.findall("joint")
        children = {joint.find("child").attrib["link"] for joint in joints}

        def add_link(
            container: ElementTree.Element,
            name: str,
            joint: ElementTree.Element | None = None,
        ) -> None:
            attributes = self.origin(joint.find("origin")) if joint is not None else {}
            if joint is None and pose is not None:
                attributes = {
                    "pos": self.values(pose[:3]),
                    "quat": self.values([pose[6], *pose[3:6]]),
                }
            body = ElementTree.SubElement(container, "body", name=name, **attributes)
            if joint is None and pose is not None:
                ElementTree.SubElement(body, "freejoint", name=f"{name}_free")
            elif joint is not None and joint.attrib["type"] != "fixed":
                limit = joint.find("limit")
                ElementTree.SubElement(
                    body,
                    "joint",
                    name=joint.attrib["name"],
                    type="slide" if joint.attrib["type"] == "prismatic" else "hinge",
                    axis=joint.find("axis").get("xyz", "0 0 1"),
                    range=f"{limit.attrib['lower']} {limit.attrib['upper']}",
                    damping="2",
                )
            inertial = links[name].find("inertial")
            if inertial is not None:
                inertia = inertial.find("inertia").attrib
                ElementTree.SubElement(
                    body,
                    "inertial",
                    pos=inertial.find("origin").get("xyz", "0 0 0"),
                    mass=inertial.find("mass").attrib["value"],
                    fullinertia=" ".join(
                        inertia[key]
                        for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                    ),
                )
            for index, collision in enumerate(links[name].findall("collision")):
                primitive = collision.find("geometry")[0]
                if primitive.tag == "box":
                    attributes = {
                        "type": "box",
                        "size": self.values(
                            np.fromstring(primitive.attrib["size"], sep=" ") / 2
                        ),
                    }
                elif primitive.tag == "cylinder":
                    attributes = {
                        "type": "cylinder",
                        "size": f"{primitive.attrib['radius']} {float(primitive.attrib['length']) / 2}",
                    }
                elif primitive.tag == "sphere":
                    attributes = {"type": "sphere", "size": primitive.attrib["radius"]}
                else:
                    raise InvalidWorkshopCommand(
                        f"Unsupported collision primitive: {primitive.tag}"
                    )
                ElementTree.SubElement(
                    body,
                    "geom",
                    name=f"{name}_collision_{index}",
                    **self.origin(collision.find("origin")),
                    **attributes,
                )
            for child_joint in joints:
                if child_joint.find("parent").attrib["link"] == name:
                    add_link(
                        body, child_joint.find("child").attrib["link"], child_joint
                    )

        for name in links.keys() - children:
            add_link(parent, name)


# %% simulation and compliant grasp


@dataclass
class WorkshopPhysics:
    """Integrate loose objects, frictional contacts and force-limited manipulation."""

    bundle_directory: Path | None = None
    """Portable bundle directory, defaulting to packaged workshop resources."""
    parameters: WorkshopPhysicsParameters = field(
        default_factory=WorkshopPhysicsParameters
    )
    """Integration, manipulation and contact limits."""
    scene: dict[str, Any] = field(init=False)
    """Scene metadata defining initial poses and instrument frames."""
    geometry: WorkshopThreadGeometry = field(init=False)
    """Portable tool, screw and workpiece dimensions."""
    model: mujoco.MjModel = field(init=False)
    """Compiled articulated collision world."""
    data: mujoco.MjData = field(init=False)
    """Measured MuJoCo state."""
    objects: dict[str, int] = field(init=False)
    """Independent object keys mapped to MuJoCo body indices."""
    drawers: dict[str, float] = field(init=False)
    """Bounded drawer target positions."""
    held: str | None = field(default=None, init=False)
    """Object currently controlled by the virtual compliant grasp."""
    target: np.ndarray | None = field(default=None, init=False)
    """Desired held-object pose as position and xyzw quaternion."""
    seated_screw: str | None = field(default=None, init=False)
    """Screw captured by the explicitly idealized pilot thread."""
    thread_turns: float = field(default=0.0, init=False)
    """Accumulated measured driver turns since pilot capture."""
    direction: int = field(default=0, init=False)
    """Requested motor direction, with zero holding the measured orientation."""
    previous_driver_rotation: Rotation | None = field(default=None, init=False)
    """Previous measured driver orientation for unwrapped angular increments."""
    initial_screw_rotation: Rotation | None = field(default=None, init=False)
    """Screw orientation relative to its workpiece when the thread seats."""
    thread_phase: ThreadState = field(default=ThreadState.FREE, init=False)
    """Current ideal-thread interaction phase."""
    elapsed_remainder: float = field(default=0.0, init=False)
    """Requested fractional integration time carried into the next step."""
    contact_gap_time: float = field(default=0.0, init=False)
    """Duration of a transient tool-slot contact gap during rotation."""
    guided_driver: bool = field(default=False, init=False)
    """Keep an engaged blade preloaded while the motor is stopped."""

    def __post_init__(self) -> None:
        """Load collision descriptions without opening graphics or robot middleware."""
        self.bundle_directory = (
            Path(self.bundle_directory)
            if self.bundle_directory is not None
            else Path(__file__).with_name("resources") / "aicor_workshop"
        )
        self.scene = json.loads((self.bundle_directory / "scene.json").read_text())
        self.geometry = WorkshopThreadGeometry.from_json(self.scene["workshop"])
        self.model = WorkshopCollisionModel(
            self.bundle_directory, self.parameters, self.scene
        ).build()
        self.data = mujoco.MjData(self.model)
        self.objects = {
            item["key"]: self.model.body(item["key"]).id
            for item in self.scene["objects"]
        }
        self.drawers = {
            item["joint"]: 0.0 for item in self.scene["workshop"]["drawers"]
        }
        self.reset()

    def validate_state(self) -> None:
        """Reject invalid numerical states before they can be reset or published."""
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or not math.isfinite(self.data.time)
        ):
            raise WorkshopSimulationFailure(
                "The workshop contact solver stopped with an invalid state."
            )

    def reset(self) -> None:
        """Restore authored poses and clear all grasp and thread constraints."""
        mujoco.mj_resetData(self.model, self.data)
        self.held = None
        self.target = None
        self.direction = 0
        self.seated_screw = None
        self.thread_turns = 0.0
        self.thread_phase = ThreadState.FREE
        self.previous_driver_rotation = None
        self.initial_screw_rotation = None
        self.elapsed_remainder = 0.0
        self.contact_gap_time = 0.0
        self.guided_driver = False
        self.drawers = dict.fromkeys(self.drawers, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def pose(self, key: str) -> np.ndarray:
        """Return a measured body-frame pose using the viewer's xyzw convention."""
        body = self.model.body(key).id
        return np.concatenate(
            (self.data.xpos[body], self.data.xquat[body][[1, 2, 3, 0]])
        )

    @staticmethod
    def bounded(vector: np.ndarray, maximum: float) -> np.ndarray:
        """Clip the magnitude of a physical force or torque."""
        return vector * min(1.0, maximum / max(float(np.linalg.norm(vector)), 1e-12))

    def grab(self, key: str) -> None:
        """Attach a bounded virtual spring to one independent body."""
        if key not in self.objects:
            raise InvalidWorkshopCommand(f"Unknown movable object: {key}")
        self.direction = 0
        self.guided_driver = False
        self.held = key
        self.target = self.pose(key)
        if key == self.seated_screw:
            self.seated_screw = None
            self.thread_turns = 0.0
            self.thread_phase = ThreadState.FREE

    def move(self, pose: list[float]) -> None:
        """Move a grasp target; the actual body remains subject to contacts."""
        value = np.asarray(pose, dtype=float)
        if self.held is None:
            raise InvalidWorkshopCommand("Grab an object before moving its target.")
        if (
            value.shape != (7,)
            or not np.isfinite(value).all()
            or np.linalg.norm(value[3:]) < 1e-6
        ):
            raise InvalidWorkshopCommand(
                "A target needs seven finite values and a nonzero quaternion."
            )
        if np.any(np.abs(value[:2]) > 2.1) or not 0.05 <= value[2] <= 2.5:
            raise InvalidWorkshopCommand(
                "The grasp target is outside the workshop workspace."
            )
        if abs(np.linalg.norm(value[3:]) - 1.0) > 1e-12:
            value[3:] /= np.linalg.norm(value[3:])
        self.target = value.copy()
        self.direction = 0
        self.guided_driver = False

    def release(self) -> None:
        """Remove the virtual grasp so gravity and contacts determine motion."""
        self.held = None
        self.target = None
        self.direction = 0
        self.guided_driver = False

    def set_drawer(self, joint: str, value: float) -> None:
        """Request a drawer position through its force-limited motor."""
        if joint not in self.drawers or not math.isfinite(value):
            raise InvalidWorkshopCommand("Unknown drawer or non-finite position.")
        limits = self.model.jnt_range[self.model.joint(joint).id]
        if not limits[0] <= value <= limits[1]:
            raise InvalidWorkshopCommand("Drawer target exceeds its travel limits.")
        self.drawers[joint] = float(value)

    def apply_grasp(
        self, key: str, target: np.ndarray, spring: float, maximum_force: float
    ) -> None:
        """Apply a damped Cartesian wrench at the measured body origin."""
        body = self.objects[key]
        measured = self.pose(key)
        world_R_body = Rotation.from_quat(measured[3:])
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
        )
        mass = self.model.body_mass[body]
        damping = 2.0 * math.sqrt(mass * spring)
        origin_offset = self.data.xpos[body] - self.data.xipos[body]
        origin_velocity = velocity[3:] + np.cross(velocity[:3], origin_offset)
        origin_force = spring * (target[:3] - measured[:3]) - damping * origin_velocity
        gravity_compensation = -mass * self.model.opt.gravity
        requested_force = origin_force + gravity_compensation
        force = self.bounded(requested_force, maximum_force)
        requested_norm = float(np.linalg.norm(requested_force))
        scale = float(np.linalg.norm(force)) / requested_norm if requested_norm else 1.0
        error = (Rotation.from_quat(target[3:]) * world_R_body.inv()).as_rotvec()
        angular_damping = 2.0 * math.sqrt(
            float(max(self.model.body_inertia[body])) * self.parameters.angular_spring
        )
        torque = self.parameters.angular_spring * error - angular_damping * velocity[:3]
        torque = self.bounded(torque, self.parameters.maximum_torque)
        self.data.xfrc_applied[body, :3] += force
        self.data.xfrc_applied[body, 3:] += torque + np.cross(
            origin_offset, scale * origin_force
        )

    def step(self, duration: float) -> None:
        """Advance a finite duration using contact-resolved integration steps."""
        if not math.isfinite(duration) or not 0 <= duration <= 30:
            raise InvalidWorkshopCommand(
                "Integration duration must be between zero and 30 seconds."
            )
        self.elapsed_remainder += duration
        steps = int((self.elapsed_remainder + 1e-12) / self.parameters.timestep)
        self.elapsed_remainder -= steps * self.parameters.timestep
        for _ in range(steps):
            self.validate_state()
            self.data.xfrc_applied[:] = 0
            self.data.qfrc_applied[:] = 0
            self.update_thread()
            if self.held is not None:
                self.apply_grasp(
                    self.held,
                    self.target,
                    self.parameters.spring,
                    self.parameters.maximum_force,
                )
            for name, target in self.drawers.items():
                joint = self.model.joint(name)
                position = self.data.qpos[joint.qposadr[0]]
                velocity = self.data.qvel[joint.dofadr[0]]
                force = (
                    self.parameters.drawer_spring * (target - position)
                    - 60.0 * velocity
                )
                self.data.qfrc_applied[joint.dofadr[0]] = np.clip(
                    force, -self.parameters.drawer_force, self.parameters.drawer_force
                )
            previous_time = float(self.data.time)
            mujoco.mj_step(self.model, self.data)
            invalid_warning = any(
                self.data.warning[warning].number
                for warning in (
                    mujoco.mjtWarning.mjWARN_BADQPOS,
                    mujoco.mjtWarning.mjWARN_BADQVEL,
                    mujoco.mjtWarning.mjWARN_BADQACC,
                )
            )
            if (
                invalid_warning
                or self.data.time <= previous_time
                or not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
            ):
                raise WorkshopSimulationFailure(
                    "The workshop contact solver stopped with an invalid state."
                )
        mujoco.mj_forward(self.model, self.data)

    # %% ideal threaded contact

    def socket(self) -> tuple[np.ndarray, Rotation]:
        """Return the measured pilot mouth and its outward shaft orientation."""
        pose = self.pose(self.geometry.assembly)
        rotation = Rotation.from_quat(pose[3:])
        return pose[:3] + rotation.apply(self.geometry.mouth), rotation

    def driver_goal(self, key: str) -> np.ndarray:
        """Place the blade in the measured screw-head slot."""
        screw = self.pose(key)
        rotation = Rotation.from_quat(screw[3:])
        depth = self.geometry.slot_floor - self.parameters.blade_preload
        return np.concatenate(
            (
                screw[:3]
                + rotation.apply(np.asarray([0, 0, depth]) - self.geometry.bit_tip),
                screw[3:],
            )
        )

    def driver_within_slot(self) -> bool:
        """Check the measured blade's position and shaft-axis alignment."""
        if self.seated_screw is None or self.geometry.driver not in self.objects:
            return False
        goal = self.driver_goal(self.seated_screw)
        driver = self.pose(self.geometry.driver)
        angle = (
            Rotation.from_quat(goal[3:]).inv() * Rotation.from_quat(driver[3:])
        ).as_rotvec()
        return bool(
            np.linalg.norm(driver[:3] - goal[:3]) < self.parameters.driver_distance
            and np.linalg.norm(angle[:2]) < self.parameters.driver_angle
        )

    def driver_aligned(self) -> bool:
        """Require actual tool-head contact and a seated, axially aligned blade."""
        if not self.driver_within_slot():
            return False
        pair = {self.objects[self.geometry.driver], self.objects[self.seated_screw]}
        contact = any(
            {
                int(self.model.geom_bodyid[item.geom1]),
                int(self.model.geom_bodyid[item.geom2]),
            }
            == pair
            and item.dist < self.parameters.contact_margin
            for item in self.data.contact[: self.data.ncon]
        )
        return bool(contact)

    def driver_engaged(self) -> bool:
        """Retain recent contact only through a short, geometrically bounded gap."""
        return self.driver_aligned() or (
            self.guided_driver
            and self.contact_gap_time <= self.parameters.contact_gap
            and self.driver_within_slot()
        )

    def drive(self, direction: int) -> None:
        """Rotate a physically engaged held driver, forward, reverse or stop."""
        if direction not in (-1, 0, 1):
            raise InvalidWorkshopCommand("Drive direction must be -1, 0 or 1.")
        if direction == 0:
            self.direction = 0
            if self.held == self.geometry.driver:
                self.target = self.pose(self.geometry.driver)
                if (
                    self.seated_screw is not None
                    and np.linalg.norm(
                        self.target[:3] - self.driver_goal(self.seated_screw)[:3]
                    )
                    < self.parameters.driver_distance
                ):
                    self.target[:3] = self.driver_goal(self.seated_screw)[:3]
            return
        if self.held != self.geometry.driver or not self.driver_engaged():
            raise InvalidWorkshopCommand(
                "Hold and align the driver in a seated screw's head slot first."
            )
        if self.parameters.maximum_torque <= 0:
            self.drive(0)
            self.thread_phase = ThreadState.STOPPED
            return
        self.direction = direction
        self.guided_driver = True
        self.contact_gap_time = 0.0
        self.previous_driver_rotation = self.socket()[1].inv() * Rotation.from_quat(
            self.pose(self.geometry.driver)[3:]
        )

    def update_thread(self) -> None:
        """Apply the bounded ideal pilot joint and integrate measured tool rotation."""
        mouth, world_R_assembly = self.socket()
        axis = world_R_assembly.apply([0, 0, 1])
        if self.seated_screw is None:
            for key in self.geometry.screws:
                screw = self.pose(key)
                relative = world_R_assembly.inv().apply(screw[:3] - mouth)
                shaft = Rotation.from_quat(screw[3:]).apply([0, 0, 1])
                if (
                    np.linalg.norm(relative[:2]) < self.parameters.pilot_distance
                    and -self.parameters.pilot_capture_depth
                    < relative[2]
                    <= self.parameters.pilot_capture_height
                    and np.dot(shaft, axis) > math.cos(self.parameters.pilot_angle)
                ):
                    self.seated_screw = key
                    self.thread_turns = 0.0
                    relative_rotation = (
                        world_R_assembly.inv() * Rotation.from_quat(screw[3:])
                    ).as_quat()
                    self.initial_screw_rotation = Rotation.from_quat(
                        [0, 0, relative_rotation[2], relative_rotation[3]]
                    )
                    self.thread_phase = ThreadState.SEATED
                    break
        if self.seated_screw is None:
            return
        if self.held == self.seated_screw:
            relative = world_R_assembly.inv().apply(
                self.pose(self.seated_screw)[:3] - mouth
            )
            if (
                np.linalg.norm(relative[:2]) > self.parameters.pilot_release_distance
                or abs(relative[2]) > self.parameters.pilot_release_height
            ):
                self.seated_screw = None
                self.thread_phase = ThreadState.FREE
            return
        engaged = self.driver_aligned()
        self.contact_gap_time = (
            0.0 if engaged else self.contact_gap_time + self.parameters.timestep
        )
        if (
            engaged
            and self.held == self.geometry.driver
            and np.linalg.norm(
                self.target[:3] - self.driver_goal(self.seated_screw)[:3]
            )
            < self.parameters.driver_distance
        ):
            self.guided_driver = True
        if self.guided_driver and (
            not self.driver_within_slot()
            or self.contact_gap_time > self.parameters.contact_gap
        ):
            self.direction = 0
            self.guided_driver = False
        if self.geometry.driver in self.objects:
            actual_rotation = world_R_assembly.inv() * Rotation.from_quat(
                self.pose(self.geometry.driver)[3:]
            )
            if self.guided_driver and self.held == self.geometry.driver:
                self.target[:3] = self.driver_goal(self.seated_screw)[:3]
            if self.direction and self.previous_driver_rotation is not None:
                delta = (
                    actual_rotation * self.previous_driver_rotation.inv()
                ).as_rotvec()
                maximum_depth = self.geometry.maximum_depth
                self.thread_turns = float(
                    np.clip(
                        self.thread_turns
                        + np.dot(delta, self.geometry.drive_axis) / (2 * math.pi),
                        0.0,
                        maximum_depth / self.geometry.pitch,
                    )
                )
                increment = Rotation.from_rotvec(
                    world_R_assembly.apply(self.geometry.drive_axis)
                    * self.direction
                    * self.parameters.driver_speed
                    * self.parameters.timestep
                )
                self.target[3:] = (
                    increment * Rotation.from_quat(self.target[3:])
                ).as_quat()
                self.target[:3] = self.driver_goal(self.seated_screw)[:3]
                self.thread_phase = ThreadState.DRIVING
                reached_bottom = (
                    self.direction > 0
                    and self.thread_turns >= maximum_depth / self.geometry.pitch
                )
                reached_top = self.direction < 0 and self.thread_turns <= 0
                if reached_bottom or reached_top:
                    self.direction = 0
                    self.target[3:] = self.pose(self.geometry.driver)[3:]
                    self.thread_phase = ThreadState.LIMIT
            elif self.thread_turns and self.thread_phase != ThreadState.LIMIT:
                self.thread_phase = ThreadState.STOPPED
            self.previous_driver_rotation = actual_rotation
        depth = self.thread_turns * self.geometry.pitch
        target = np.concatenate(
            (
                mouth - depth * axis,
                (
                    world_R_assembly
                    * Rotation.from_rotvec([0, 0, -2 * math.pi * self.thread_turns])
                    * self.initial_screw_rotation
                ).as_quat(),
            )
        )
        self.apply_thread_constraint(target)

    def apply_thread_constraint(self, target: np.ndarray) -> None:
        """Apply equal and opposite compliant reactions between screw and workpiece."""
        screw_body = self.objects[self.seated_screw]
        assembly_body = self.objects[self.geometry.assembly]
        screw = self.pose(self.seated_screw)
        screw_velocity, assembly_velocity = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            screw_body,
            screw_velocity,
            0,
        )
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            assembly_body,
            assembly_velocity,
            0,
        )
        assembly_point_velocity = assembly_velocity[3:] + np.cross(
            assembly_velocity[:3], screw[:3] - self.data.xipos[assembly_body]
        )
        damping = 2.0 * math.sqrt(
            self.model.body_mass[screw_body] * self.parameters.thread_spring
        )
        force = self.parameters.thread_spring * (target[:3] - screw[:3]) - damping * (
            screw_velocity[3:] - assembly_point_velocity
        )
        error = (
            Rotation.from_quat(target[3:]) * Rotation.from_quat(screw[3:]).inv()
        ).as_rotvec()
        angular_damping = 2.0 * math.sqrt(
            float(max(self.model.body_inertia[screw_body]))
            * self.parameters.thread_angular_spring
        )
        torque = self.parameters.thread_angular_spring * error - angular_damping * (
            screw_velocity[:3] - assembly_velocity[:3]
        )
        force = self.bounded(force, self.parameters.thread_force)
        torque = self.bounded(torque, self.parameters.thread_torque)
        for body, sign in ((screw_body, 1.0), (assembly_body, -1.0)):
            self.data.xfrc_applied[body, :3] += sign * force
            self.data.xfrc_applied[body, 3:] += sign * (
                torque + np.cross(screw[:3] - self.data.xipos[body], force)
            )

    # %% measured state

    def contacts(self) -> list[dict[str, Any]]:
        """Expose bounded contact records with solver-measured reaction forces."""
        result = []
        for index, contact in enumerate(self.data.contact[: self.data.ncon]):
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if force[0] <= 1e-8:
                continue
            result.append(
                {
                    "bodyA": self.model.body(
                        int(
                            self.model.body_weldid[
                                self.model.geom_bodyid[contact.geom1]
                            ]
                        )
                    ).name,
                    "bodyB": self.model.body(
                        int(
                            self.model.body_weldid[
                                self.model.geom_bodyid[contact.geom2]
                            ]
                        )
                    ).name,
                    "position": contact.pos.tolist(),
                    "normal": contact.frame[:3].tolist(),
                    "force": float(force[0]),
                    "distance": float(contact.dist),
                }
            )
        return sorted(result, key=lambda contact: contact["force"], reverse=True)[
            : self.parameters.maximum_contacts
        ]

    def state(self) -> dict[str, Any]:
        """Return measured poses, joints, contacts and transparent thread status."""
        self.validate_state()
        mouth, rotation = self.socket()
        candidate = self.seated_screw or (
            self.held if self.held in self.geometry.screws else self.geometry.screws[0]
        )
        depth = (
            0.0
            if self.seated_screw is None
            else float(
                np.dot(
                    mouth - self.pose(self.seated_screw)[:3], rotation.apply([0, 0, 1])
                )
            )
        )
        return {
            "objects": {key: self.pose(key).tolist() for key in self.objects},
            "joints": {
                key: float(self.data.qpos[self.model.joint(key).qposadr[0]])
                for key in self.drawers
            },
            "held": self.held,
            "target": None if self.target is None else self.target.tolist(),
            "contacts": self.contacts(),
            "time": float(self.data.time),
            "thread": {
                "engaged": self.driver_engaged(),
                "screw": candidate,
                "driver": self.geometry.driver,
                "depth": depth,
                "turns": self.thread_turns,
                "pitch": self.geometry.pitch,
                "state": self.thread_phase.value,
                "driverAligned": self.driver_engaged(),
                "driverTargetPose": (
                    None if candidate is None else self.driver_goal(candidate).tolist()
                ),
                "socketPose": np.concatenate((mouth, rotation.as_quat())).tolist(),
                "screwTargetPose": np.concatenate(
                    (
                        mouth + rotation.apply([0, 0, self.parameters.pilot_clearance]),
                        rotation.as_quat(),
                    )
                ).tolist(),
            },
        }

    def snapshot(self) -> dict[str, Any]:
        """Expose the same measured state for session persistence and polling."""
        return self.state()
