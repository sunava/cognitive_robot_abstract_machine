"""Articulated laboratory centrifuge with measured loading and drive interlocks."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import Any

from cramera.laboratory_centrifuge_command import CentrifugeAction, CentrifugeField
from cramera.laboratory_manual import ManualKnowledge


# %% authored mechanics
class CentrifugeGeometry(StrEnum):
    """Shared names of the articulated instrument and its clamp constraints."""

    LID_JOINT = "laboratory_centrifuge_lid_joint"
    """Physical cover hinge."""
    ROTOR_JOINT = "laboratory_centrifuge_rotor_joint"
    """Continuous drive hinge."""
    ROTOR_BODY = "laboratory_centrifuge_rotor"
    """Rotating assembly carrying the sample holders."""
    LID_BODY = "laboratory_centrifuge_lid"
    """Moving protective cover."""
    LID_MOTOR = "laboratory_centrifuge_lid_motor"
    """Bounded cover position actuator."""
    ROTOR_MOTOR = "laboratory_centrifuge_rotor_motor"
    """Bounded rotor velocity actuator."""
    CLAMP_PREFIX = "laboratory_centrifuge_clamp_"
    """Prefix of the per-glass mechanical fixture constraint."""
    ROTOR_SITE_PREFIX = "laboratory_centrifuge_rotor_anchor_"
    """Prefix of each movable clamp attachment frame on the rotor."""
    TUBE_SITE_PREFIX = "laboratory_centrifuge_tube_anchor_"
    """Prefix of each glass's fixed attachment frame."""


class CentrifugeMetadataField(StrEnum):
    """Authored geometry bounds used by measured instrument interlocks."""

    OPEN_ANGLE = "openAngle"
    """Mechanical limit of the lid hinge in radians."""
    CHAMBER_RADIUS = "chamberRadius"
    """Clear radial extent of the instrument chamber."""
    CHAMBER_HEIGHT = "chamberHeight"
    """Upper chamber edge in rotor-local coordinates."""


@dataclass(frozen=True)
class CentrifugeParameters:
    """Conservative mechanical limits of the simulated bench instrument."""

    motor_torque: float = 0.12
    """Maximum rotor drive and braking torque in Newton meters."""
    motor_gain: float = 0.05
    """Velocity servo gain in Newton meter seconds per radian."""
    rotor_damping: float = 0.0001
    """Bearing damping in Newton meter seconds per radian."""
    acceleration_rpm: float = 60.0
    """Maximum commanded change of speed per simulated second."""
    lid_speed: float = 0.6
    """Maximum change of lid command in radians per second."""
    lid_stiffness: float = 20.0
    """Position stiffness of the lid actuator in Newton meters per radian."""
    lid_damping: float = 1.0
    """Damping of the lid actuator in Newton meter seconds per radian."""
    lid_torque: float = 2.0
    """Maximum lid opening or closing torque in Newton meters."""
    stopped_rpm: float = 0.5
    """Largest measured speed permitting a lid or clamp operation."""
    lid_tolerance: float = 0.02
    """Joint-angle tolerance for the fully open and fully closed states."""
    seating_distance: float = 0.003
    """Largest tube-bottom displacement from a measured holder position."""
    seating_angle: float = 0.04
    """Largest tube-axis tilt permitting mechanical clamp engagement."""
    seating_speed: float = 0.02
    """Largest tube linear speed permitting mechanical clamp engagement."""
    balance_grams: float = 0.25
    """Maximum mass difference between each occupied opposing pair."""
    support_fraction: float = 0.5
    """Minimum fraction of vessel weight measured on the holder floor."""
    default_rpm: float = 120.0
    """Default requested rotor speed in revolutions per minute."""
    default_duration: float = 20.0
    """Default cycle duration after attaining commanded speed."""
    maximum_duration: float = 300.0
    """Longest allowed timed cycle in simulation seconds."""
    speed_tolerance: float = 0.05
    """Relative speed tolerance that starts the timed cycle."""
    chamber_lower_margin: float = 0.04
    """Allowance below slot floors when checking for loose objects in meters."""


@dataclass
class InvalidCentrifugeOperation(ValueError):
    """A requested operation violates the instrument's mechanical interlocks."""

    reason: str
    """Condition that must be satisfied before retrying the command."""

    def __str__(self) -> str:
        """Expose the unmet condition to the operator."""
        return self.reason


@dataclass
class CentrifugeCollisionModel:
    """Add motor actuation and inactive physical fixture constraints to MJCF."""

    root: ElementTree.Element
    """MJCF containing the authored instrument bodies and loose vessels."""
    tube_keys: tuple[str, ...]
    """Glass bodies that can be mechanically secured to the rotor."""
    parameters: CentrifugeParameters = field(default_factory=CentrifugeParameters)
    """Force limits of the instrument actuators."""

    @staticmethod
    def add_joint(
        body: ElementTree.Element,
        link: ElementTree.Element,
        joint: ElementTree.Element | None,
    ) -> None:
        """Preserve the instrument's lid and rotor articulation from its URDF."""
        if joint is None or joint.attrib["name"] not in (
            CentrifugeGeometry.LID_JOINT,
            CentrifugeGeometry.ROTOR_JOINT,
        ):
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
        properties = {
            "name": joint.attrib["name"],
            "type": "hinge",
            "axis": joint.find("axis").attrib["xyz"],
            "damping": str(CentrifugeParameters().rotor_damping),
        }
        if joint.attrib["type"] == "revolute":
            limits = joint.find("limit").attrib
            properties.update(
                limited="true", range=f"{limits['lower']} {limits['upper']}"
            )
            body.set("gravcomp", "1")
        ElementTree.SubElement(body, "joint", **properties)

    def complete(self) -> None:
        """Install bounded motors and measured-pose clamps when an instrument exists."""
        rotor = self.root.find(f".//body[@name='{CentrifugeGeometry.ROTOR_BODY}']")
        if rotor is None:
            return
        actuators = ElementTree.SubElement(self.root, "actuator")
        ElementTree.SubElement(
            actuators,
            "velocity",
            name=CentrifugeGeometry.ROTOR_MOTOR,
            joint=CentrifugeGeometry.ROTOR_JOINT,
            kv=str(self.parameters.motor_gain),
            forcelimited="true",
            forcerange=f"{-self.parameters.motor_torque} {self.parameters.motor_torque}",
        )
        ElementTree.SubElement(
            actuators,
            "position",
            name=CentrifugeGeometry.LID_MOTOR,
            joint=CentrifugeGeometry.LID_JOINT,
            kp=str(self.parameters.lid_stiffness),
            kv=str(self.parameters.lid_damping),
            forcelimited="true",
            forcerange=f"{-self.parameters.lid_torque} {self.parameters.lid_torque}",
        )
        equalities = ElementTree.SubElement(self.root, "equality")
        for key in self.tube_keys:
            tube = self.root.find(f".//body[@name='{key}']")
            rotor_site = f"{CentrifugeGeometry.ROTOR_SITE_PREFIX}{key}"
            tube_site = f"{CentrifugeGeometry.TUBE_SITE_PREFIX}{key}"
            ElementTree.SubElement(rotor, "site", name=rotor_site, size="0.001")
            ElementTree.SubElement(tube, "site", name=tube_site, size="0.001")
            ElementTree.SubElement(
                equalities,
                "weld",
                name=f"{CentrifugeGeometry.CLAMP_PREFIX}{key}",
                site1=rotor_site,
                site2=tube_site,
                active="false",
                solref="0.008 1",
                torquescale="0.02",
            )


@dataclass
class CentrifugeSlot:
    """One opposed holder and its deliberately engaged mechanical fixture."""

    identifier: str
    """User-visible holder identifier."""
    opposite: str
    """Holder on the opposite side of the rotor."""
    position: np.ndarray
    """Desired tube-bottom position in rotor-local coordinates."""
    floor_id: int
    """Collision surface that must support a vessel before engagement."""
    key: str | None = None
    """Secured glass identifier, or None for an unlatched holder."""


# %% measured interlocks and actuation
@dataclass
class LaboratoryCentrifuge:
    """Operate physical hinges and secured glass holders under measured interlocks."""

    model: mujoco.MjModel
    """Compiled contact and actuator model."""
    data: mujoco.MjData
    """Current authoritative solver state."""
    metadata: dict[str, Any]
    """Authored instrument geometry, or an empty mapping for older bundles."""
    tube_keys: tuple[str, ...]
    """Existing glass bodies eligible for a rotor holder."""
    parameters: CentrifugeParameters = field(default_factory=CentrifugeParameters)
    """Measured safety margins and motor command limits."""
    knowledge: ManualKnowledge = field(default_factory=ManualKnowledge)
    """Validated operating document retained independently of physical resets."""
    rotor_body: int = field(init=False)
    """Compiled rotor body index, or minus one when absent."""
    rotor_joint: int = field(init=False, default=-1)
    """Compiled hinge coordinate of the physical rotor."""
    lid_joint: int = field(init=False, default=-1)
    """Compiled hinge coordinate of the physical lid."""
    rotor_actuator: int = field(init=False, default=-1)
    """Bounded velocity motor driving the rotor."""
    lid_actuator: int = field(init=False, default=-1)
    """Bounded position motor driving the lid."""
    slots: dict[str, CentrifugeSlot] = field(init=False, default_factory=dict)
    """Authored pocket layout and currently secured objects."""
    target_rpm: float = field(init=False, default=0.0)
    """Final requested rotor speed before acceleration limiting."""
    commanded_rpm: float = field(init=False, default=0.0)
    """Acceleration-limited velocity command sent to the motor."""
    lid_target: float = field(init=False, default=0.0)
    """Requested closed or fully open lid position."""
    remaining_seconds: float = field(init=False, default=0.0)
    """Time remaining after measured speed first reaches its target band."""
    timer_started: bool = field(init=False, default=False)
    """Whether this cycle has attained its target speed at least once."""

    def __post_init__(self) -> None:
        """Resolve the optional instrument and its physical contact surfaces."""
        self.rotor_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, CentrifugeGeometry.ROTOR_BODY
        )
        if not self.available:
            return
        self.rotor_joint = self.model.joint(CentrifugeGeometry.ROTOR_JOINT).id
        self.lid_joint = self.model.joint(CentrifugeGeometry.LID_JOINT).id
        self.rotor_actuator = self.model.actuator(CentrifugeGeometry.ROTOR_MOTOR).id
        self.lid_actuator = self.model.actuator(CentrifugeGeometry.LID_MOTOR).id
        for index, slot in enumerate(self.metadata[CentrifugeField.SLOTS], start=1):
            identifier = slot[CentrifugeField.ID]
            self.slots[identifier] = CentrifugeSlot(
                identifier,
                slot[CentrifugeField.OPPOSITE],
                np.asarray(slot[CentrifugeField.POSITION], dtype=float),
                self.model.geom(
                    f"{CentrifugeGeometry.ROTOR_BODY}_slot_{index}_floor"
                ).id,
            )

    @property
    def available(self) -> bool:
        """Whether the bundle includes the articulated centrifuge."""
        return self.rotor_body >= 0 and bool(self.metadata)

    @property
    def open_angle(self) -> float:
        """Authored mechanical opening limit of the lid."""
        return float(self.metadata.get(CentrifugeMetadataField.OPEN_ANGLE, 0.0))

    @property
    def maximum_rpm(self) -> float:
        """Authored operating speed limit in revolutions per minute."""
        return float(self.metadata.get(CentrifugeField.MAXIMUM_RPM, 0.0))

    @property
    def rpm(self) -> float:
        """Measured rotor speed in revolutions per minute."""
        if not self.available:
            return 0.0
        return (
            float(self.data.qvel[self.model.jnt_dofadr[self.rotor_joint]])
            * 60
            / math.tau
        )

    @property
    def lid_angle(self) -> float:
        """Measured lid position in radians from its closed configuration."""
        return (
            float(self.data.qpos[self.model.jnt_qposadr[self.lid_joint]])
            if self.available
            else 0.0
        )

    @property
    def stopped(self) -> bool:
        """Whether all commanded and measured rotor motion has ceased."""
        return (
            self.target_rpm == 0
            and self.commanded_rpm == 0
            and abs(self.rpm) < self.parameters.stopped_rpm
        )

    @property
    def lid_open(self) -> bool:
        """Whether the lid has reached its fully open loading position."""
        return (
            self.available
            and self.lid_angle >= self.open_angle - self.parameters.lid_tolerance
        )

    @property
    def lid_closed(self) -> bool:
        """Whether the measured lid is at its closed latch position."""
        return (
            self.available
            and abs(self.lid_angle) <= self.parameters.lid_tolerance
            and self.lid_target == 0
        )

    @property
    def balanced(self) -> bool:
        """Whether every occupied slot has a secured mass-matched opposing vessel."""
        occupied = [slot for slot in self.slots.values() if slot.key is not None]
        if not occupied or self.has_unsecured_load():
            return False
        return all(
            self.slots[slot.opposite].key is not None
            and abs(self.slot_mass(slot) - self.slot_mass(self.slots[slot.opposite]))
            <= self.parameters.balance_grams
            for slot in occupied
        )

    def has_unsecured_load(self) -> bool:
        """Detect an unclamped vessel inside the chamber or touching the rotor."""
        if not self.available:
            return False
        secured = {slot.key for slot in self.slots.values()}
        rotor_rotation = self.data.xmat[self.rotor_body].reshape(3, 3)
        radius = float(self.metadata[CentrifugeMetadataField.CHAMBER_RADIUS])
        ceiling = float(self.metadata[CentrifugeMetadataField.CHAMBER_HEIGHT])
        floor = (
            min(slot.position[2] for slot in self.slots.values())
            - self.parameters.chamber_lower_margin
        )
        loose_bodies = set()
        for key in self.tube_keys:
            if key in secured:
                continue
            body = self.model.body(key).id
            loose_bodies.add(body)
            local = rotor_rotation.T @ (
                self.data.xpos[body] - self.data.xpos[self.rotor_body]
            )
            if np.linalg.norm(local[:2]) <= radius and floor <= local[2] <= ceiling:
                return True
        return any(
            self.rotor_body
            in (
                self.model.geom_bodyid[contact.geom1],
                self.model.geom_bodyid[contact.geom2],
            )
            and bool(
                loose_bodies
                & {
                    self.model.geom_bodyid[contact.geom1],
                    self.model.geom_bodyid[contact.geom2],
                }
            )
            for contact in self.data.contact
        )

    def slot_mass(self, slot: CentrifugeSlot) -> float:
        """Return secured glass and contained liquid mass in grams."""
        return (
            0.0
            if slot.key is None
            else float(self.model.body_mass[self.model.body(slot.key).id]) * 1000
        )

    def slot_position(self, identifier: str) -> np.ndarray:
        """Return a holder's tube-bottom position from the measured rotor transform."""
        if identifier not in self.slots:
            raise InvalidCentrifugeOperation("Unknown centrifuge holder")
        return (
            self.data.xpos[self.rotor_body]
            + self.data.xmat[self.rotor_body].reshape(3, 3)
            @ self.slots[identifier].position
        )

    def reset(self) -> None:
        """Restore the open, stopped instrument and disengage every mechanical clamp."""
        self.target_rpm = self.commanded_rpm = self.remaining_seconds = 0.0
        self.timer_started = False
        if not self.available:
            return
        self.lid_target = self.open_angle
        self.data.qpos[self.model.jnt_qposadr[self.lid_joint]] = self.open_angle
        self.data.ctrl[self.lid_actuator] = self.open_angle
        self.data.ctrl[self.rotor_actuator] = 0
        for key in self.tube_keys:
            self.data.eq_active[
                self.model.equality(f"{CentrifugeGeometry.CLAMP_PREFIX}{key}").id
            ] = False
        for slot in self.slots.values():
            slot.key = None

    def validate_manipulation(self, key: str) -> None:
        """Reject cursor or liquid commands for a mechanically clamped vessel."""
        if any(slot.key == key for slot in self.slots.values()):
            raise InvalidCentrifugeOperation(
                "Release this glass from the centrifuge holder first"
            )

    def read_centrifuge_manual(self, document: str) -> None:
        """Replace operating knowledge only while the physical rotor is stopped."""
        if not self.available:
            raise InvalidCentrifugeOperation("This laboratory has no centrifuge")
        if not self.stopped:
            raise InvalidCentrifugeOperation(
                "Stop the rotor before reading another operating manual"
            )
        self.knowledge.read(document, simulation_time=float(self.data.time))

    def _require_manual_load(self) -> None:
        """Enforce the document's minimum load before enabling rotor actuation."""
        secured = sum(slot.key is not None for slot in self.slots.values())
        required = self.knowledge.active.minimum_samples
        if secured < required:
            raise InvalidCentrifugeOperation(
                f"Manual {self.knowledge.active.version}: at least {required} secured samples required; currently {secured}."
            )

    def _require_loading(self) -> None:
        """Require a stopped rotor and fully open lid for fixture changes."""
        if not self.stopped or not self.lid_open:
            raise InvalidCentrifugeOperation(
                "Stop the rotor and fully open the lid before loading"
            )

    def _secure(self, identifier: str | None, key: str | None) -> None:
        """Engage a holder at its measured seated pose without moving the glass."""
        self._require_loading()
        if identifier not in self.slots or key not in self.tube_keys:
            raise InvalidCentrifugeOperation(
                "Select an existing glass and centrifuge holder"
            )
        slot = self.slots[identifier]
        if slot.key is not None or any(
            other.key == key for other in self.slots.values()
        ):
            raise InvalidCentrifugeOperation("The holder or glass is already secured")
        body = self.model.body(key).id
        rotation = self.data.xmat[body].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
        )
        seated = (
            np.linalg.norm(self.data.xpos[body] - self.slot_position(identifier))
            <= self.parameters.seating_distance
            and rotation[2, 2] >= math.cos(self.parameters.seating_angle)
            and np.linalg.norm(velocity[3:]) <= self.parameters.seating_speed
        )
        support = 0.0
        effort = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geometries = (contact.geom1, contact.geom2)
            if slot.floor_id not in geometries or contact.efc_address < 0:
                continue
            other = contact.geom2 if contact.geom1 == slot.floor_id else contact.geom1
            if self.model.geom_bodyid[other] != body:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, effort)
            support += max(0.0, float(effort[0]))
        weight = self.model.body_mass[body] * np.linalg.norm(self.model.opt.gravity)
        if not seated or support < self.parameters.support_fraction * weight:
            raise InvalidCentrifugeOperation(
                "Seat the glass on the holder floor and let it settle before securing"
            )
        rotor_rotation = self.data.xmat[self.rotor_body].reshape(3, 3)
        site = self.model.site(f"{CentrifugeGeometry.ROTOR_SITE_PREFIX}{key}").id
        self.model.site_pos[site] = rotor_rotation.T @ (
            self.data.xpos[body] - self.data.xpos[self.rotor_body]
        )
        self.model.site_quat[site] = Rotation.from_matrix(
            rotor_rotation.T @ rotation
        ).as_quat(scalar_first=True)
        # A measured anchor needs its complete local frame evaluated after movement.
        self.model.site_sameframe[site] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        self.data.eq_active[
            self.model.equality(f"{CentrifugeGeometry.CLAMP_PREFIX}{key}").id
        ] = True
        slot.key = key
        mujoco.mj_forward(self.model, self.data)

    def command(
        self,
        action: str,
        slot: str | None = None,
        key: str | None = None,
        rpm: float | None = None,
        duration: float | None = None,
    ) -> None:
        """Validate and apply one deliberate lid, fixture or timed-drive operation."""
        if not self.available:
            raise InvalidCentrifugeOperation("This laboratory has no centrifuge")
        if action == CentrifugeAction.SECURE:
            self._secure(slot, key)
        elif action == CentrifugeAction.RELEASE:
            self._require_loading()
            if slot not in self.slots or self.slots[slot].key is None:
                raise InvalidCentrifugeOperation("This holder has no secured glass")
            secured = self.slots[slot]
            equality = self.model.equality(
                f"{CentrifugeGeometry.CLAMP_PREFIX}{secured.key}"
            ).id
            self.data.eq_active[equality] = False
            secured.key = None
        elif action in (CentrifugeAction.OPEN, CentrifugeAction.CLOSE):
            if not self.stopped:
                raise InvalidCentrifugeOperation(
                    "Wait until the rotor has completely stopped"
                )
            self.lid_target = (
                self.open_angle if action == CentrifugeAction.OPEN else 0.0
            )
        elif action == CentrifugeAction.START:
            self._require_manual_load()
            requested = self.parameters.default_rpm if rpm is None else rpm
            seconds = self.parameters.default_duration if duration is None else duration
            if (
                isinstance(requested, bool)
                or not isinstance(requested, (int, float))
                or not math.isfinite(requested)
                or not 1 <= requested <= self.maximum_rpm
            ):
                raise InvalidCentrifugeOperation(
                    "Requested rotor speed is outside its operating limits"
                )
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or not 1 <= seconds <= self.parameters.maximum_duration
            ):
                raise InvalidCentrifugeOperation(
                    "Cycle duration must be between one and 300 simulation seconds"
                )
            if not self.lid_closed or not self.balanced or not self.stopped:
                raise InvalidCentrifugeOperation(
                    "Close the lid and secure a balanced opposing pair before starting"
                )
            self.target_rpm = float(requested)
            self.remaining_seconds = float(seconds)
            self.timer_started = False
        elif action == CentrifugeAction.STOP:
            self.target_rpm = self.remaining_seconds = 0.0
            self.timer_started = False
        else:
            raise InvalidCentrifugeOperation("Unknown centrifuge operation")

    def advance(self, elapsed: float) -> None:
        """Ramp bounded actuators and stop the timed cycle from measured speed."""
        if not self.available:
            return
        if self.target_rpm > 0:
            if not self.lid_closed or not self.balanced:
                self.target_rpm = self.remaining_seconds = 0.0
            elif (
                abs(self.rpm - self.target_rpm)
                <= self.parameters.speed_tolerance * self.target_rpm
            ):
                self.timer_started = True
            if self.timer_started:
                self.remaining_seconds = max(0.0, self.remaining_seconds - elapsed)
                if self.remaining_seconds == 0:
                    self.target_rpm = 0.0
        self.commanded_rpm += float(
            np.clip(
                self.target_rpm - self.commanded_rpm,
                -self.parameters.acceleration_rpm * elapsed,
                self.parameters.acceleration_rpm * elapsed,
            )
        )
        if abs(self.commanded_rpm) < 1e-9:
            self.commanded_rpm = 0.0
        self.data.ctrl[self.rotor_actuator] = self.commanded_rpm * math.tau / 60
        self.data.ctrl[self.lid_actuator] += float(
            np.clip(
                self.lid_target - self.data.ctrl[self.lid_actuator],
                -self.parameters.lid_speed * elapsed,
                self.parameters.lid_speed * elapsed,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        """Return measured motor state, slot positions, balance and renderer joints."""
        radius = max(
            (np.linalg.norm(slot.position[:2]) for slot in self.slots.values()),
            default=0.0,
        )
        gravity = float(np.linalg.norm(self.model.opt.gravity))
        relative_force = (
            (self.rpm * math.tau / 60) ** 2 * radius / gravity if gravity else 0.0
        )
        state = {
            CentrifugeField.AVAILABLE: self.available,
            CentrifugeField.RPM: round(self.rpm, 2),
            CentrifugeField.TARGET_RPM: self.target_rpm,
            CentrifugeField.RELATIVE_FORCE: round(relative_force, 3),
            CentrifugeField.LID_ANGLE: self.lid_angle,
            CentrifugeField.LID_OPEN: self.lid_open,
            CentrifugeField.LID_CLOSED: self.lid_closed,
            CentrifugeField.LOCKED: not self.stopped,
            CentrifugeField.RUNNING: not self.stopped,
            CentrifugeField.BALANCED: self.balanced,
            CentrifugeField.MAXIMUM_RPM: self.maximum_rpm,
            CentrifugeField.REMAINING_SECONDS: round(self.remaining_seconds, 2),
            CentrifugeField.JOINTS: (
                {
                    CentrifugeGeometry.LID_JOINT: self.lid_angle,
                    CentrifugeGeometry.ROTOR_JOINT: float(
                        self.data.qpos[self.model.jnt_qposadr[self.rotor_joint]]
                    ),
                }
                if self.available
                else {}
            ),
            CentrifugeField.SLOTS: [
                {
                    CentrifugeField.ID: slot.identifier,
                    CentrifugeField.OPPOSITE: slot.opposite,
                    CentrifugeField.POSITION: self.slot_position(
                        slot.identifier
                    ).tolist(),
                    CentrifugeField.KEY: slot.key,
                    CentrifugeField.GRAMS: round(self.slot_mass(slot), 3),
                    CentrifugeField.SECURED: slot.key is not None,
                }
                for slot in self.slots.values()
            ],
        }
        state[CentrifugeField.KNOWLEDGE] = self.knowledge.snapshot(state)
        return state
