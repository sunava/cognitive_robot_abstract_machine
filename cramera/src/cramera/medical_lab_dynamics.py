"""Force-limited articulation for independently authored laboratory instruments."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np

from cramera.medical_lab_description import (
    MedicalLabDescription,
    MedicalLabJoint,
    MedicalLabJointKind,
)


# %% physical instrument vocabulary
class MedicalPhysicsField(StrEnum):
    """Manifest and measured-state sections of the expanded laboratory."""

    DESCRIPTION = "medicalLab"
    """Authored instruments and workstations."""
    INSTRUMENTS = "instruments"
    """Measured instrument articulation and demo state."""
    JOINTS = "joints"
    """Observed scalar joint positions."""
    CONTACT_EXCLUSIONS = "contactExclusions"
    """Explicitly connected bearing parts separated by multiple joints."""
    FIRST = "first"
    """First body of a mechanical mating pair."""
    SECOND = "second"
    """Second body of a mechanical mating pair."""


@dataclass(frozen=True)
class MedicalDriveParameters:
    """Conservative inspection servos, independent of scientific process models."""

    angular_stiffness: float = 40.0
    """Hinge position stiffness in Newton metres per radian."""
    angular_damping: float = 4.0
    """Hinge damping in Newton metre seconds per radian."""
    linear_stiffness: float = 400.0
    """Slide position stiffness in Newtons per metre."""
    linear_damping: float = 20.0
    """Slide damping in Newton seconds per metre."""
    angular_tolerance: float = 0.008
    """Maximum accepted hinge displacement in radians."""
    linear_tolerance: float = 0.0005
    """Maximum accepted slide displacement in metres."""
    settling_speed: float = 0.02
    """Largest joint speed accepted as settled in its native units per second."""


# %% collision and actuator import
@dataclass
class MedicalLabCollision:
    """Retain explicitly declared URDF joints and their bounded physical motors."""

    description: MedicalLabDescription
    """The instrument metadata paired with the composed URDF."""
    parameters: MedicalDriveParameters = field(default_factory=MedicalDriveParameters)
    """Inspection servo stiffness and damping."""
    contact_exclusions: tuple[dict[str, str], ...] = ()
    """Explicit authored mating pairs in addition to directly connected bodies."""
    joints: dict[str, MedicalLabJoint] = field(init=False)
    """Declared joint definitions indexed by their unique composed names."""
    imported: set[str] = field(default_factory=set, init=False)
    """Declared joints actually encountered in the URDF tree."""
    mating_bodies: dict[str, str] = field(default_factory=dict, init=False)
    """Directly joined moving bodies and their physical mounting bodies."""

    def __post_init__(self) -> None:
        """Index only the joints belonging to the authored instruments."""
        self.joints = {
            joint.name: joint
            for instrument in self.description.instruments
            for joint in instrument.joints
        }

    @staticmethod
    def motor_name(joint: str) -> str:
        """Return the actuator name distinct from the URDF joint namespace."""
        return f"{joint}__drive"

    def add_joint(
        self,
        body: ElementTree.Element,
        link: ElementTree.Element,
        joint: ElementTree.Element | None,
    ) -> None:
        """Preserve one declared joint, its inertia and its URDF physical limits."""
        if joint is None or joint.attrib["name"] not in self.joints:
            return
        name = joint.attrib["name"]
        definition = self.joints[name]
        kind = MedicalLabJointKind(joint.attrib["type"])
        inertial = link.find("inertial")
        if inertial is None:
            raise ValueError(f"Moving instrument body has no inertia: {name}")
        origin = inertial.find("origin")
        if origin is not None and origin.attrib.get("rpy", "0 0 0") != "0 0 0":
            raise ValueError(f"Instrument inertia must use its link axes: {name}")
        inertia = inertial.find("inertia").attrib
        ElementTree.SubElement(
            body,
            "inertial",
            mass=inertial.find("mass").attrib["value"],
            fullinertia=" ".join(
                inertia[key] for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            ),
            pos=origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
        )
        properties = {
            "name": name,
            "type": "slide" if kind is MedicalLabJointKind.PRISMATIC else "hinge",
            "axis": joint.find("axis").attrib["xyz"],
            "armature": "0.0001",
        }
        if kind is not MedicalLabJointKind.CONTINUOUS:
            limits = joint.find("limit").attrib
            properties.update(
                limited="true", range=f"{limits['lower']} {limits['upper']}"
            )
            if (float(limits["lower"]), float(limits["upper"])) != (
                definition.lower,
                definition.upper,
            ):
                raise ValueError(f"Instrument limits differ from its URDF: {name}")
        if float(joint.find("limit").attrib["effort"]) != definition.effort:
            raise ValueError(f"Instrument effort differs from its URDF: {name}")
        body.set("gravcomp", "1")
        ElementTree.SubElement(body, "joint", **properties)
        self.imported.add(name)
        self.mating_bodies[joint.find("child").attrib["link"]] = joint.find(
            "parent"
        ).attrib["link"]

    def complete(self, root: ElementTree.Element) -> None:
        """Add physical motors only after every declared joint was imported."""
        missing = self.joints.keys() - self.imported
        if missing:
            raise ValueError(
                f"Instrument joints missing from environment: {sorted(missing)}"
            )
        if not self.joints:
            return
        contacts = ElementTree.SubElement(root, "contact")
        bodies = {body.attrib["name"]: body for body in root.findall(".//body")}
        excluded: set[frozenset[str]] = set()
        for child, parent in self.mating_bodies.items():
            for moving in self._fixed_descendants(bodies[child]):
                for mounting in self._fixed_descendants(bodies[parent]):
                    pair = frozenset((moving, mounting))
                    if len(pair) == 2 and pair not in excluded:
                        ElementTree.SubElement(
                            contacts, "exclude", body1=mounting, body2=moving
                        )
                        excluded.add(pair)
        for entry in self.contact_exclusions:
            first, second = (
                entry[MedicalPhysicsField.FIRST],
                entry[MedicalPhysicsField.SECOND],
            )
            if (
                first not in bodies
                or second not in bodies
                or first.split("__")[0] != second.split("__")[0]
            ):
                raise ValueError(
                    "Mechanical mating exclusions must reference bodies of the same instrument"
                )
            pair = frozenset((first, second))
            if len(pair) == 2 and pair not in excluded:
                ElementTree.SubElement(contacts, "exclude", body1=first, body2=second)
                excluded.add(pair)
        actuators = ElementTree.SubElement(root, "actuator")
        for joint in self.joints.values():
            linear = joint.kind == MedicalLabJointKind.PRISMATIC
            ElementTree.SubElement(
                actuators,
                "position",
                name=self.motor_name(joint.name),
                joint=joint.name,
                kp=str(
                    self.parameters.linear_stiffness
                    if linear
                    else self.parameters.angular_stiffness
                ),
                kv=str(
                    self.parameters.linear_damping
                    if linear
                    else self.parameters.angular_damping
                ),
                forcelimited="true",
                forcerange=f"{-joint.effort} {joint.effort}",
            )

    @classmethod
    def _fixed_descendants(cls, body: ElementTree.Element) -> tuple[str, ...]:
        """Group rigidly attached geometry with its mechanical mounting link."""
        names = [body.attrib["name"]]
        for child in body.findall("body"):
            if child.find("joint") is None and child.find("freejoint") is None:
                names.extend(cls._fixed_descendants(child))
        return tuple(names)


# %% measured motion control
@dataclass(frozen=True)
class MedicalJointIndices:
    """Compiled state offsets of a single independently driven instrument joint."""

    position: int
    """Scalar coordinate offset in MuJoCo's position state."""
    velocity: int
    """Scalar coordinate offset in MuJoCo's velocity state."""
    actuator: int
    """Position servo index in MuJoCo's control array."""


@dataclass
class MedicalLabDrives:
    """Ramp bounded demands while contacts determine actual instrument motion."""

    model: mujoco.MjModel
    """Compiled contact model containing the instrument motors."""
    data: mujoco.MjData
    """Authoritative dynamic state owned by the simulation thread."""
    description: MedicalLabDescription
    """Authored articulation, rates and inspection poses."""
    parameters: MedicalDriveParameters = field(default_factory=MedicalDriveParameters)
    """Measured completion tolerances."""
    joints: dict[str, MedicalLabJoint] = field(init=False)
    """Joint definitions indexed by their unique names."""
    indices: dict[str, MedicalJointIndices] = field(init=False)
    """Compiled position, velocity and actuator offsets."""
    targets: dict[str, float] = field(default_factory=dict, init=False)
    """Requested final positions, never substituted for measurements."""
    demands: dict[str, float] = field(default_factory=dict, init=False)
    """Rate-limited servo positions between measurements and final targets."""
    velocities: dict[str, float] = field(default_factory=dict, init=False)
    """Requested reference speeds bounded by the authored drive limits."""

    def __post_init__(self) -> None:
        """Resolve the imported articulation and initialize bounded demands."""
        self.joints = {
            joint.name: joint
            for instrument in self.description.instruments
            for joint in instrument.joints
        }
        self.indices = {}
        for name in self.joints:
            joint = self.model.joint(name).id
            self.indices[name] = MedicalJointIndices(
                int(self.model.jnt_qposadr[joint]),
                int(self.model.jnt_dofadr[joint]),
                self.model.actuator(MedicalLabCollision.motor_name(name)).id,
            )
        self.reset()

    def reset(self) -> None:
        """Restore authored rest poses only at the explicit simulation reset boundary."""
        for name, joint in self.joints.items():
            indices = self.indices[name]
            self.data.qpos[indices.position] = joint.rest
            self.data.qvel[indices.velocity] = 0.0
            self.data.ctrl[indices.actuator] = joint.rest
            self.targets[name] = joint.rest
            self.demands[name] = joint.rest
            self.velocities[name] = joint.velocity
        mujoco.mj_forward(self.model, self.data)

    def set_targets(
        self, targets: dict[str, float], velocities: dict[str, float] | None = None
    ) -> None:
        """Validate all requested targets before changing any servo demand."""
        for name, value in targets.items():
            if name not in self.joints:
                raise ValueError(f"Unknown instrument joint: {name}")
            joint = self.joints[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Instrument target must be finite: {name}")
            if (
                joint.kind != MedicalLabJointKind.CONTINUOUS
                and not joint.lower <= value <= joint.upper
            ):
                raise ValueError(f"Instrument target is outside its limits: {name}")
        if velocities is not None:
            if velocities.keys() != targets.keys() or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= self.joints[name].velocity
                for name, value in velocities.items()
            ):
                raise ValueError("Instrument speeds must match bounded joint targets")
        self.targets.update(targets)
        self.velocities.update(
            velocities or {name: self.joints[name].velocity for name in targets}
        )

    def advance(self) -> None:
        """Apply a bounded step of position demand to the physical motors."""
        for name, joint in self.joints.items():
            maximum = self.velocities[name] * self.model.opt.timestep
            delta = float(
                np.clip(self.targets[name] - self.demands[name], -maximum, maximum)
            )
            self.demands[name] += delta
            self.data.ctrl[self.indices[name].actuator] = self.demands[name]

    def positions(self) -> dict[str, float]:
        """Read actual articulated positions from the contact solver."""
        return {
            name: float(self.data.qpos[indices.position])
            for name, indices in self.indices.items()
        }

    def hold(self) -> None:
        """Stop further demand travel at each joint's measured position."""
        self.targets = self.positions()
        self.demands = self.targets.copy()
        for name, value in self.targets.items():
            self.data.ctrl[self.indices[name].actuator] = value

    def reached(self, targets: dict[str, float]) -> bool:
        """Require settled measured positions for every commanded joint."""
        for name, value in targets.items():
            indices = self.indices[name]
            tolerance = (
                self.parameters.linear_tolerance
                if self.joints[name].kind == MedicalLabJointKind.PRISMATIC
                else self.parameters.angular_tolerance
            )
            if abs(float(self.data.qpos[indices.position]) - value) > tolerance:
                return False
            if (
                abs(float(self.data.qvel[indices.velocity]))
                > self.parameters.settling_speed
            ):
                return False
        return True
