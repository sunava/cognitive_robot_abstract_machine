"""Physical sample-cabinet articulation and reduced simulated cooling."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np
from typing_extensions import Any

from cramera.laboratory_refrigerator_command import (
    RefrigeratorAction,
    RefrigeratorField,
    RefrigeratorTemperatureModel,
)


# %% physical cabinet contract
class RefrigeratorGeometry(StrEnum):
    """Stable authored names of the cabinet and its bounded door actuator."""

    BASE = "laboratory_refrigerator_base"
    """Stationary cabinet and shelf collision body."""
    DOOR = "laboratory_refrigerator_door"
    """Moving glass door body."""
    JOINT = "laboratory_refrigerator_door_joint"
    """Vertical door hinge."""
    MOTOR = "laboratory_refrigerator_door_motor"
    """Torque-limited position actuator acting on the door hinge."""


@dataclass(frozen=True)
class RefrigeratorParameters:
    """Door effort, supported-placement tolerances and reduced thermal response."""

    maximum_torque: float = 2.0
    """Maximum door actuator torque in Newton metres."""
    stiffness: float = 15.0
    """Door position gain in Newton metres per radian."""
    damping: float = 4.0
    """Door actuator damping in Newton metre seconds per radian."""
    hinge_damping: float = 0.1
    """Passive hinge friction in Newton metre seconds per radian."""
    angular_speed: float = 0.65
    """Maximum door reference speed in radians per simulated second."""
    angle_tolerance: float = 0.05
    """Angular tolerance for fully open and closed states."""
    position_tolerance: float = 0.008
    """Maximum tube-root displacement from the authored storage position."""
    upright_tolerance: float = 0.15
    """Maximum storage tilt in radians from the vertical."""
    support_fraction: float = 0.2
    """Minimum fraction of tube weight supported by upward cabinet contacts."""
    ambient_temperature: float = 22.0
    """Assumed laboratory air temperature in degrees Celsius."""
    cooling_seconds: float = 45.0
    """Closed-cabinet exponential cooling time constant in simulated seconds."""
    warming_seconds: float = 20.0
    """Open-cabinet exponential warming time constant in simulated seconds."""


@dataclass
class InvalidRefrigeratorOperation(ValueError):
    """A cabinet operation cannot be performed in the current physical state."""

    reason: str
    """Explanation returned to the operator."""

    def __str__(self) -> str:
        """Return the measured-state restriction."""
        return self.reason


@dataclass
class RefrigeratorCollisionModel:
    """Preserve the authored hinge and add a bounded simulated door drive."""

    root: ElementTree.Element
    """MJCF tree containing the cabinet collision bodies."""
    parameters: RefrigeratorParameters = field(default_factory=RefrigeratorParameters)
    """Door hinge damping and actuator limits."""

    @staticmethod
    def add_joint(
        body: ElementTree.Element,
        link: ElementTree.Element,
        joint: ElementTree.Element | None,
    ) -> None:
        """Import the refrigerator's inertial frame and bounded vertical hinge."""
        if joint is None or joint.attrib["name"] != RefrigeratorGeometry.JOINT:
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
        limits = joint.find("limit").attrib
        ElementTree.SubElement(
            body,
            "joint",
            name=RefrigeratorGeometry.JOINT,
            type="hinge",
            axis=joint.find("axis").attrib["xyz"],
            limited="true",
            range=f"{limits['lower']} {limits['upper']}",
            damping=str(RefrigeratorParameters().hinge_damping),
        )

    def complete(self) -> None:
        """Add one torque-limited actuator when the authored cabinet is present."""
        if self.root.find(f".//joint[@name='{RefrigeratorGeometry.JOINT}']") is None:
            return
        actuators = self.root.find("actuator")
        if actuators is None:
            actuators = ElementTree.SubElement(self.root, "actuator")
        ElementTree.SubElement(
            actuators,
            "position",
            name=RefrigeratorGeometry.MOTOR,
            joint=RefrigeratorGeometry.JOINT,
            kp=str(self.parameters.stiffness),
            kv=str(self.parameters.damping),
            forcelimited="true",
            forcerange=f"{-self.parameters.maximum_torque} {self.parameters.maximum_torque}",
        )


# %% measured storage and reduced cooling
@dataclass
class LaboratoryRefrigerator:
    """Operate a physical door and report supported samples with simulated cooling."""

    model: mujoco.MjModel
    """Compiled collision model containing the cabinet and loose samples."""
    data: mujoco.MjData
    """Authoritative physical positions, contacts and actuator state."""
    metadata: dict[str, Any]
    """Authored storage positions and operating geometry, empty for older bundles."""
    tube_keys: tuple[str, ...]
    """Loose sample identifiers eligible for cabinet storage."""
    parameters: RefrigeratorParameters = field(default_factory=RefrigeratorParameters)
    """Bounded door motion and reduced thermal response parameters."""
    body: int = field(init=False)
    """Cabinet body index, or minus one when absent."""
    joint: int = field(init=False)
    """Door hinge index, or minus one when absent."""
    actuator: int = field(init=False)
    """Door actuator index, or minus one when absent."""
    target_angle: float = field(init=False, default=0.0)
    """Requested final door opening angle in radians."""
    reference_angle: float = field(init=False, default=0.0)
    """Rate-limited actuator reference in radians."""
    temperature: float = field(init=False, default=0.0)
    """Current reduced-model cabinet temperature in degrees Celsius."""

    def __post_init__(self) -> None:
        """Resolve optional cabinet indices without making older bundles mandatory."""
        self.body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, RefrigeratorGeometry.BASE
        )
        self.joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, RefrigeratorGeometry.JOINT
        )
        self.actuator = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, RefrigeratorGeometry.MOTOR
        )
        self.temperature = self.target_temperature

    @property
    def available(self) -> bool:
        """Whether the authored cabinet has a physically actuated door."""
        return bool(self.metadata) and min(self.body, self.joint, self.actuator) >= 0

    @property
    def target_temperature(self) -> float:
        """Cooling target supplied by the authored sample cabinet."""
        return float(
            self.metadata.get(
                RefrigeratorField.AUTHORED_TEMPERATURE,
                self.parameters.ambient_temperature,
            )
        )

    @property
    def open_angle(self) -> float:
        """Authored maximum opening angle of the door."""
        return float(self.metadata.get(RefrigeratorField.DOOR_OPEN, 0.0))

    @property
    def door_angle(self) -> float:
        """Measured door hinge angle without advancing physical state."""
        return (
            float(self.data.qpos[self.model.jnt_qposadr[self.joint]])
            if self.available
            else 0.0
        )

    @property
    def door_open(self) -> bool:
        """Whether the measured aperture permits the sample-loading path."""
        return (
            self.available
            and self.door_angle >= self.open_angle - self.parameters.angle_tolerance
        )

    @property
    def door_closed(self) -> bool:
        """Whether the measured hinge is near its closed mechanical stop."""
        return (
            self.available and abs(self.door_angle) <= self.parameters.angle_tolerance
        )

    def reset(self) -> None:
        """Restore a closed, pre-cooled cabinet without moving any sample."""
        self.target_angle = self.reference_angle = 0.0
        self.temperature = self.target_temperature
        if self.available:
            self.data.ctrl[self.actuator] = 0.0
            self.data.qpos[self.model.jnt_qposadr[self.joint]] = 0.0
            self.data.qvel[self.model.jnt_dofadr[self.joint]] = 0.0

    def command(self, action: RefrigeratorAction | str) -> None:
        """Request a door destination without directly setting any physical pose."""
        if not self.available:
            raise InvalidRefrigeratorOperation(
                "This laboratory has no sample refrigerator"
            )
        if action not in RefrigeratorAction:
            raise InvalidRefrigeratorOperation("Unknown refrigerator operation")
        self.target_angle = (
            self.open_angle if action == RefrigeratorAction.OPEN else 0.0
        )

    def advance(self, timestep: float) -> None:
        """Advance the bounded door reference and reduced cabinet heat exchange."""
        if not self.available:
            return
        maximum_change = self.parameters.angular_speed * timestep
        self.reference_angle += float(
            np.clip(
                self.target_angle - self.reference_angle,
                -maximum_change,
                maximum_change,
            )
        )
        self.data.ctrl[self.actuator] = self.reference_angle
        aperture = float(np.clip(self.door_angle / self.open_angle, 0.0, 1.0))
        equilibrium = self.target_temperature + aperture * (
            self.parameters.ambient_temperature - self.target_temperature
        )
        time_constant = (
            self.parameters.warming_seconds
            if equilibrium > self.temperature
            else self.parameters.cooling_seconds
        )
        self.temperature += (equilibrium - self.temperature) * (
            -math.expm1(-timestep / time_constant)
        )

    def supported(self, body: int) -> bool:
        """Require upward cabinet contact carrying part of the sample's weight."""
        supported_force = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first, second = (
                int(self.model.geom_bodyid[geometry])
                for geometry in (contact.geom1, contact.geom2)
            )
            if {first, second} != {self.body, body} or contact.efc_address < 0:
                continue
            effort = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, effort)
            direction = 1.0 if second == body else -1.0
            supported_force += max(
                0.0, direction * float(contact.frame[2]) * float(effort[0])
            )
        weight = self.model.body_mass[body] * float(
            np.linalg.norm(self.model.opt.gravity)
        )
        return supported_force > weight * self.parameters.support_fraction

    def occupants(self) -> dict[str, str | None]:
        """Match supported, upright tube roots to the authored storage positions."""
        slots = self.metadata.get(RefrigeratorField.SLOTS, [])
        occupants = {slot[RefrigeratorField.ID]: None for slot in slots}
        if not self.available:
            return occupants
        for key in self.tube_keys:
            body = self.model.body(key).id
            if self.data.xmat[body].reshape(3, 3)[2, 2] < math.cos(
                self.parameters.upright_tolerance
            ) or not self.supported(body):
                continue
            for slot in slots:
                position = np.asarray(slot[RefrigeratorField.POSITION])
                if (
                    np.linalg.norm(self.data.xpos[body] - position)
                    <= self.parameters.position_tolerance
                ):
                    occupants[slot[RefrigeratorField.ID]] = key
                    break
        return occupants

    def validate_manipulation(self, key: str) -> None:
        """Require the door to be fully open before retrieving a stored sample."""
        if not self.door_open and key in self.occupants().values():
            raise InvalidRefrigeratorOperation(
                "Open the refrigerator door before moving the stored sample"
            )

    def snapshot(self) -> dict[str, Any]:
        """Publish measured door and contacts beside explicitly simulated temperature."""
        occupants = self.occupants()
        return {
            RefrigeratorField.AVAILABLE: self.available,
            RefrigeratorField.DOOR_ANGLE: self.door_angle,
            RefrigeratorField.DOOR_OPEN: self.door_open,
            RefrigeratorField.DOOR_CLOSED: self.door_closed,
            RefrigeratorField.TEMPERATURE: round(self.temperature, 2),
            RefrigeratorField.TARGET_TEMPERATURE: self.target_temperature,
            RefrigeratorField.TEMPERATURE_MODEL: RefrigeratorTemperatureModel.SIMULATED,
            RefrigeratorField.JOINTS: (
                {RefrigeratorGeometry.JOINT: self.door_angle} if self.available else {}
            ),
            RefrigeratorField.SLOTS: [
                {
                    RefrigeratorField.ID: slot[RefrigeratorField.ID],
                    RefrigeratorField.POSITION: list(slot[RefrigeratorField.POSITION]),
                    RefrigeratorField.KEY: occupants[slot[RefrigeratorField.ID]],
                }
                for slot in self.metadata.get(RefrigeratorField.SLOTS, [])
            ],
        }
