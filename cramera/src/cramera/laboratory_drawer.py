"""Physical drawer rails and bounded endpoint holding for hand manipulation."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np
from typing_extensions import Any


# %% authored geometry and measured state
class DrawerGeometry(StrEnum):
    """Authored names of the drawer, rail and physical grasp bar."""

    BODY = "laboratory_drawer"
    """Sliding tray and front panel."""
    JOINT = "laboratory_drawer_joint"
    """Bounded outward rail translation."""
    MOTOR = "laboratory_drawer_motor"
    """Force-limited endpoint holding actuator."""
    HANDLE = "laboratory_drawer_drawer_handle"
    """Contact geometry for the front grasp bar."""


class DrawerField(StrEnum):
    """Drawer observations published with the laboratory physics state."""

    DRAWER = "drawer"
    """Top-level drawer observation."""
    AVAILABLE = "available"
    """Whether a physical rail and holding actuator were compiled."""
    POSITION = "position"
    """Measured outward translation in metres."""
    TARGET_POSITION = "targetPosition"
    """Requested final translation in metres."""
    OPEN = "open"
    """Whether the rail has reached its outer endpoint."""
    CLOSED = "closed"
    """Whether the rail has reached its inner endpoint."""
    HAND_CONTROLLED = "handControlled"
    """Whether the endpoint motor is disabled for manipulation."""
    JOINTS = "joints"
    """Measured articulation keyed by authored joint name."""


@dataclass(frozen=True)
class DrawerParameters:
    """Bounded endpoint holding and measured rail completion tolerances."""

    maximum_force: float = 40.0
    """Maximum endpoint motor force in Newtons."""
    stiffness: float = 400.0
    """Endpoint holding gain in Newtons per metre."""
    damping: float = 50.0
    """Motor damping in Newton seconds per metre."""
    speed: float = 0.10
    """Maximum actuator reference speed in metres per second."""
    position_tolerance: float = 0.005
    """Measured endpoint tolerance in metres."""


@dataclass
class InvalidDrawerOperation(ValueError):
    """A drawer target is unavailable or outside its authored rail limits."""

    reason: str
    """Description of the invalid drawer request."""

    def __str__(self) -> str:
        """Present the physical operation restriction."""
        return self.reason


# %% collision model
@dataclass
class DrawerCollisionModel:
    """Preserve the authored rail and add force-limited endpoint holding."""

    root: ElementTree.Element
    """MJCF tree containing the drawer body."""
    parameters: DrawerParameters = field(default_factory=DrawerParameters)
    """Holding actuator gains and force limit."""

    @staticmethod
    def add_joint(
        body: ElementTree.Element,
        link: ElementTree.Element,
        joint: ElementTree.Element | None,
    ) -> None:
        """Import the drawer inertial frame and authored prismatic dynamics."""
        if joint is None or joint.attrib["name"] != DrawerGeometry.JOINT:
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
        dynamics = joint.find("dynamics").attrib
        ElementTree.SubElement(
            body,
            "joint",
            name=DrawerGeometry.JOINT,
            type="slide",
            axis=joint.find("axis").attrib["xyz"],
            limited="true",
            range=f"{limits['lower']} {limits['upper']}",
            damping=dynamics["damping"],
            frictionloss=dynamics["friction"],
        )

    def complete(self) -> None:
        """Add a bounded motor when the authored drawer has a physical rail."""
        if self.root.find(f".//joint[@name='{DrawerGeometry.JOINT}']") is None:
            return
        actuators = self.root.find("actuator")
        if actuators is None:
            actuators = ElementTree.SubElement(self.root, "actuator")
        ElementTree.SubElement(
            actuators,
            "position",
            name=DrawerGeometry.MOTOR,
            joint=DrawerGeometry.JOINT,
            kp=str(self.parameters.stiffness),
            kv=str(self.parameters.damping),
            forcelimited="true",
            forcerange=f"{-self.parameters.maximum_force} {self.parameters.maximum_force}",
        )


# %% physical endpoint control
@dataclass
class LaboratoryDrawer:
    """Hold a measured rail endpoint or release its motor to the physical hand."""

    model: mujoco.MjModel
    """Compiled drawer collision and actuator model."""
    data: mujoco.MjData
    """Authoritative positions, velocities and controls."""
    parameters: DrawerParameters = field(default_factory=DrawerParameters)
    """Reference speed and motor force limits."""
    joint: int = field(init=False)
    """Compiled prismatic joint index, or minus one when absent."""
    actuator: int = field(init=False)
    """Compiled endpoint motor index, or minus one when absent."""
    target_position: float = field(init=False, default=0.0)
    """Requested final outward translation in metres."""
    reference_position: float = field(init=False, default=0.0)
    """Rate-limited motor reference in metres."""
    hand_controlled: bool = field(init=False, default=False)
    """Whether the endpoint motor is disabled during manipulation."""

    def __post_init__(self) -> None:
        """Resolve optional rail and motor indices without changing physical state."""
        self.joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, DrawerGeometry.JOINT
        )
        self.actuator = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, DrawerGeometry.MOTOR
        )

    @property
    def available(self) -> bool:
        """Whether the authored drawer has a bounded physical rail motor."""
        return min(self.joint, self.actuator) >= 0

    @property
    def position(self) -> float:
        """Measured outward rail displacement in metres."""
        return (
            float(self.data.qpos[self.model.jnt_qposadr[self.joint]])
            if self.available
            else 0.0
        )

    @property
    def open_position(self) -> float:
        """Outer mechanical stop imported from the authored rail limit."""
        return float(self.model.jnt_range[self.joint, 1]) if self.available else 0.0

    @property
    def open(self) -> bool:
        """Whether measured travel reaches the outer rail endpoint."""
        return self.available and abs(self.position - self.open_position) <= (
            self.parameters.position_tolerance
        )

    @property
    def closed(self) -> bool:
        """Whether measured travel reaches the inner rail endpoint."""
        return (
            self.available and abs(self.position) <= self.parameters.position_tolerance
        )

    def hold(self, position: float) -> None:
        """Request a finite bounded motor target without changing object poses."""
        if (
            not self.available
            or isinstance(position, bool)
            or not isinstance(position, (int, float))
            or not math.isfinite(position)
            or not 0.0 <= position <= self.open_position
        ):
            raise InvalidDrawerOperation("Drawer target is outside its physical rail")
        self.target_position = float(position)

    def release_to_hand(self) -> None:
        """Disable the endpoint motor while contact forces drive the rail."""
        if not self.available:
            raise InvalidDrawerOperation("The drawer has no physical rail")
        self.model.actuator_forcerange[self.actuator] = 0.0
        self.hand_controlled = True

    def restore_hold(self, position: float) -> None:
        """Hold the requested endpoint with a synchronized motor reference."""
        if (
            not isinstance(position, bool)
            and isinstance(position, (int, float))
            and math.isfinite(position)
            and -self.parameters.position_tolerance
            <= position
            <= self.open_position + self.parameters.position_tolerance
        ):
            position = float(np.clip(position, 0.0, self.open_position))
        self.hold(position)
        self.reference_position = self.target_position
        self.data.ctrl[self.actuator] = self.reference_position
        self.model.actuator_forcerange[self.actuator] = (
            -self.parameters.maximum_force,
            self.parameters.maximum_force,
        )
        self.hand_controlled = False

    def reset(self) -> None:
        """Restore bounded holding at the authored closed initial state."""
        if self.available:
            self.restore_hold(0.0)

    def advance(self, timestep: float) -> None:
        """Advance only the bounded motor reference while holding is enabled."""
        if not self.available or self.hand_controlled:
            return
        distance = self.target_position - self.reference_position
        self.reference_position += float(
            np.clip(
                distance,
                -self.parameters.speed * timestep,
                self.parameters.speed * timestep,
            )
        )
        self.data.ctrl[self.actuator] = self.reference_position

    def snapshot(self) -> dict[str, Any]:
        """Publish measured rail articulation and motor ownership."""
        return {
            DrawerField.AVAILABLE: self.available,
            DrawerField.POSITION: self.position,
            DrawerField.TARGET_POSITION: self.target_position,
            DrawerField.OPEN: self.open,
            DrawerField.CLOSED: self.closed,
            DrawerField.HAND_CONTROLLED: self.hand_controlled,
            DrawerField.JOINTS: (
                {DrawerGeometry.JOINT: self.position} if self.available else {}
            ),
        }
