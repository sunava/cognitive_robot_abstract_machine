"""Headless native cable dynamics with an actuated friction gripper."""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import current_thread

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import Any, ClassVar

from cramera.cable import (
    Cable,
    CableConfig,
    CableMujocoBuilder,
    CableMujocoSynchronizer,
)
from cramera.cable_session import CableAction, CableField, CableSession
from physics_simulators.base_simulator import SimulatorState
from physics_simulators.mujoco_simulator import MujocoSimulator
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


# %% apparatus parameters
class CableRigName(StrEnum):
    """Stable names of the laboratory's physical parts."""

    CARRIAGE = "cable_gripper"
    """Cartesian carriage carrying the fingers."""
    LEFT_PAD = "cable_gripper_left_pad"
    """Finger on the positive local y side."""
    RIGHT_PAD = "cable_gripper_right_pad"
    """Finger on the negative local y side."""
    TABLE = "cable_support"
    """Narrow support rail below the cable."""
    FLOOR = "cable_floor"
    """Ground below the laboratory."""
    SCENE = "cable-laboratory.xml"
    """Private initial semantic-world model."""
    WORLD = "world"
    """Root frame of the semantic laboratory."""
    MOUNT = "cable_mount"
    """Elevated parent of the optional endpoint anchor."""


class GraspState(StrEnum):
    """Contact-derived observations of the pinch."""

    OPEN = "open"
    """The fingers have been commanded open."""
    CONTACT = "contact"
    """One or both pads touch the cable."""
    HOLDING = "holding"
    """Both pads contact the cable with little relative motion."""
    SLIPPING = "slipping"
    """Closed fingers have no stable two-sided cable contact."""


class CableObservation(StrEnum):
    """Fields in the laboratory's browser-facing physical observations."""

    SIMULATION_TIME = "simulation_time"
    """Elapsed physical simulation time."""
    CONFIG = "config"
    """Cable configuration used to build the scene."""
    SEGMENTS = "segments"
    """Measured physical capsules."""
    POINTS = "points"
    """Synchronized semantic point references."""
    NAME = "name"
    """Stable body or point name."""
    START = "start"
    """First capsule centerline endpoint."""
    END = "end"
    """Second capsule centerline endpoint."""
    GRIPPER = "gripper"
    """Measured physical gripper pose and pads."""
    OPENING = "opening"
    """Distance between opposing inner pad faces."""
    PADS = "pads"
    """Measured pad boxes."""
    SIZE = "size"
    """Full box dimensions in meters."""
    QUATERNION = "quaternion"
    """Measured XYZW orientation."""
    CONTACTS = "contacts"
    """Solved cable contact forces."""
    NORMAL_FORCE = "normal_force"
    """Compressive contact force in Newtons."""
    FRICTION_FORCE = "friction_force"
    """Tangential contact force magnitude in Newtons."""
    BODY_A = "body_a"
    """First contact body."""
    BODY_B = "body_b"
    """Second contact body."""
    ANCHOR = "anchor"
    """Initial endpoint pose and anchoring selection."""
    ENABLED = "enabled"
    """Whether the optional anchor is present."""
    GRASP = "grasp"
    """Contact-derived pinch observation."""
    CONTACT_COUNT = "contact_count"
    """Number of force-bearing pad-to-cable contacts."""
    PHYSICAL = "physical"
    """Whether native cable installation succeeded."""
    WARNINGS = "warnings"
    """Construction or numerical diagnostics."""
    TABLE = "table"
    """Fixed rail supporting the initial cable."""


@dataclass(frozen=True)
class CableRigParameters:
    """Physical dimensions, actuator limits and motion rates of the test rig."""

    timestep: float = 0.001
    """Integration interval in seconds."""
    lift_distance: float = 0.05
    """Carriage travel per lift or lower command in meters."""
    carriage_speed: float = 0.05
    """Maximum commanded carriage speed in meters per second."""
    pad_speed: float = 0.06
    """Maximum finger target speed in meters per second."""
    open_clearance: float = 0.02
    """Distance from an open pad's inner face to the cable surface."""
    pad_half_size: tuple[float, float, float] = (0.06, 0.004, 0.02)
    """Half extents of each rectangular contact pad in meters."""
    pad_force: float = 2.0
    """Maximum absolute force of each finger actuator in Newtons."""
    pad_stiffness: float = 500.0
    """Finger position-servo gain in Newtons per meter."""
    pad_damping: float = 2.0
    """Finger velocity-servo gain in Newton seconds per meter."""
    carriage_stiffness: float = 2000.0
    """Carriage position-servo gain in Newtons per meter."""
    carriage_damping: float = 60.0
    """Carriage velocity-servo gain in Newton seconds per meter."""
    carriage_force: float = 100.0
    """Maximum Cartesian actuator force in Newtons."""
    support_depth: float = 0.02
    """Thickness of the cable support rail in meters."""
    support_margin: float = 0.05
    """Extra support length beyond each cable endpoint."""
    floor_depth: float = 0.35
    """Distance from the initial cable to the ground in meters."""
    holding_speed: float = 0.01
    """Maximum observed relative speed for reporting a stable pinch."""
    mount_height: float = 0.35
    """Height of the parent cable mount above the ground in meters."""


# %% simulator lifetime
@dataclass(eq=False)
class CableContactSimulator(MujocoSimulator):
    """Serialize stepping with control and snapshot access at real-time pace."""

    def step(self) -> None:
        """Advance and synchronize one locked step, pacing only the background thread."""
        started = time.monotonic()
        with self._model_lock:
            super().step()
        if self.simulation_thread is current_thread():
            time.sleep(max(0.0, self.step_size - (time.monotonic() - started)))


@dataclass(init=False)
class CableLaboratorySimulation(MujocoSim):
    """Build and synchronize a native cable in a private headless scene."""

    builder_class: ClassVar[type[CableMujocoBuilder]] = CableMujocoBuilder
    """Builder omitting the semantic cable placeholders from physics."""
    synchronizer_class: ClassVar[type[CableMujocoSynchronizer]] = (
        CableMujocoSynchronizer
    )
    """Synchronizer reading native cable poses into the semantic world."""
    simulator_class: ClassVar[type[CableContactSimulator]] = CableContactSimulator
    """Physics thread with a model-wide access lock."""

    def __init__(self, world: World, scene_path: Path, timestep: float) -> None:
        """Compile the initial semantic scene into its own temporary file."""
        self.default_file_path = str(scene_path)
        super().__init__(
            world,
            headless=True,
            step_size=timestep,
            integrator=mujoco.mjtIntegrator.mjINT_RK4,
            cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
            impratio=10.0,
        )


# %% physical apparatus and controls
@dataclass
class CablePhysics:
    """Own a semantic cable, its headless simulation, and a contact-only gripper."""

    config: CableConfig = field(default_factory=CableConfig)
    """Native cable material, placement and contact parameters."""
    parameters: CableRigParameters = field(default_factory=CableRigParameters)
    """Gripper geometry and bounded actuator settings."""
    world: World = field(init=False)
    """Semantic world containing independently addressable cable points."""
    cable: Cable = field(init=False)
    """Native composite cable and semantic placeholder references."""
    simulation: CableLaboratorySimulation = field(init=False)
    """Headless framework simulation and its world synchronizer."""
    _directory: TemporaryDirectory = field(init=False, repr=False)
    """Private scene directory retained throughout the simulation lifetime."""
    _origin: np.ndarray = field(init=False, repr=False)
    """Initial carriage position in world coordinates."""
    _target: np.ndarray = field(init=False, repr=False)
    """Requested carriage position in world coordinates."""
    _rotation: np.ndarray = field(init=False, repr=False)
    """Fixed gripper orientation mapping local axes into world coordinates."""
    _stopped: bool = field(init=False, default=False)
    """Whether callbacks have already been disconnected by final shutdown."""
    _pinching: bool = field(init=False, default=False)
    """Whether the finger target requests compressive contact."""
    _actuators: tuple[int, ...] = field(init=False, repr=False)
    """Cartesian and finger actuator indices in that order."""
    _pad_bodies: tuple[int, int] = field(init=False, repr=False)
    """Indices of the two contact-pad bodies."""
    _cable_bodies: tuple[int, ...] = field(init=False, repr=False)
    """Indices of the native cable elements when compilation succeeds."""
    _support_position: np.ndarray = field(init=False, repr=False)
    """World position of the rail's center."""
    _support_size: np.ndarray = field(init=False, repr=False)
    """Full dimensions of the rail."""

    def __post_init__(self) -> None:
        """Build the independent native apparatus and register physics callbacks."""
        self.world = World()
        root = Body(name=PrefixedName(CableRigName.WORLD))
        mount = Body(name=PrefixedName(CableRigName.MOUNT))
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(root)
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(mount)
            self.world.add_connection(
                FixedConnection(
                    parent=root,
                    child=mount,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=self.parameters.mount_height
                    ),
                )
            )
        self.cable = Cable.create(self.world, self.config, mount)
        with ExitStack() as cleanup:
            self._directory = TemporaryDirectory(prefix="cramera-cable-")
            cleanup.callback(self._directory.cleanup)
            self.simulation = CableLaboratorySimulation(
                self.world,
                Path(self._directory.name) / CableRigName.SCENE,
                self.parameters.timestep,
            )
            cleanup.callback(self.simulation.synchronizer.stop)
            cleanup.callback(self.simulation.simulator.stop)
            simulator = self.simulation.simulator
            self._build_apparatus(simulator._mj_spec)
            simulator._mj_model, simulator._mj_data = simulator._mj_spec.recompile(
                simulator._mj_model, simulator._mj_data
            )
            compilation = self.cable.install(
                simulator._mj_spec, simulator._mj_model, simulator._mj_data
            )
            simulator._mj_spec = compilation.spec
            simulator._mj_model = compilation.model
            simulator._mj_data = compilation.data
            self._actuators = tuple(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self._actuator_names()
            )
            self._pad_bodies = tuple(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in (CableRigName.LEFT_PAD, CableRigName.RIGHT_PAD)
            )
            self._cable_bodies = (
                tuple(
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                    for name in self.cable.physical_body_names
                )
                if self.cable.active
                else ()
            )
            simulator.pre_step_callback = self._prepare_step
            mujoco.mj_forward(self.model, simulator._mj_data)
            self.cable.sync(self.model, simulator._mj_data)
            cleanup.pop_all()

    @property
    def model(self) -> mujoco.MjModel:
        """Return the current compiled physical model."""
        return self.simulation.simulator._mj_model

    @property
    def running(self) -> bool:
        """Report whether the framework's background physics thread is active."""
        simulator = self.simulation.simulator
        return (
            simulator.state == SimulatorState.RUNNING
            and simulator.simulation_thread is not None
            and simulator.simulation_thread.is_alive()
        )

    @property
    def segment_positions(self) -> np.ndarray:
        """Copy the measured origins of all physical cable elements."""
        with self.simulation.simulator._model_lock:
            return self.simulation.simulator._mj_data.xpos[
                list(self._cable_bodies)
            ].copy()

    def start(self) -> None:
        """Start one headless framework physics thread."""
        if self._stopped:
            raise ValueError(
                "A stopped cable laboratory must be reconstructed before starting"
            )
        if self.cable.active and not self.running:
            self.simulation.start_simulation()

    def stop(self) -> None:
        """Stop and join the physics thread and release its world callbacks."""
        if self._stopped:
            return
        self._stopped = True
        self.simulation.simulator.stop()
        self.simulation.synchronizer.stop()
        self._directory.cleanup()

    def step(self, count: int = 1) -> None:
        """Advance a stopped laboratory deterministically for experiments."""
        if self.running:
            raise ValueError("Deterministic stepping requires a stopped physics thread")
        if not isinstance(count, int) or count < 1:
            raise ValueError("Step count must be a positive integer")
        simulator = self.simulation.simulator
        with simulator._model_lock:
            for _ in range(count):
                self._prepare_step()
                mujoco.mj_step(self.model, simulator._mj_data)
            mujoco.mj_forward(self.model, simulator._mj_data)
            self.cable.sync(self.model, simulator._mj_data)

    def command(
        self,
        action: str | CableAction,
        position: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        """Set bounded gripper actuator targets without moving cable coordinates."""
        action = CableAction(action)
        with self.simulation.simulator._model_lock:
            if action in (CableAction.PINCH, CableAction.RELEASE):
                self._pinching = action == CableAction.PINCH
                return
            if action in (CableAction.LIFT, CableAction.LOWER):
                direction = 1.0 if action == CableAction.LIFT else -1.0
                self._target[2] = np.clip(
                    self._target[2] + direction * self.parameters.lift_distance,
                    CableSession.TARGET_MINIMUM[2],
                    CableSession.TARGET_MAXIMUM[2],
                )
                return
            target = np.asarray(position, dtype=float)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError(
                    "Move position must contain three finite world coordinates"
                )
            if np.any(target < CableSession.TARGET_MINIMUM) or np.any(
                target > CableSession.TARGET_MAXIMUM
            ):
                raise ValueError("Move position is outside the laboratory workspace")
            self._target = target.copy()

    def _prepare_step(self) -> None:
        """Slew force-limited actuator targets toward the current user command."""
        data = self.simulation.simulator._mj_data
        target = self._rotation.T @ (self._target - self._origin)
        for axis, actuator in enumerate(self._actuators[:3]):
            difference = target[axis] - data.ctrl[actuator]
            maximum = self.parameters.carriage_speed * self.parameters.timestep
            data.ctrl[actuator] += np.clip(difference, -maximum, maximum)
        closing_distance = self.parameters.open_clearance + self.config.radius
        target = closing_distance if self._pinching else 0.0
        for actuator in self._actuators[3:]:
            maximum = self.parameters.pad_speed * self.parameters.timestep
            data.ctrl[actuator] += np.clip(
                target - data.ctrl[actuator], -maximum, maximum
            )

    @staticmethod
    def _actuator_names() -> tuple[str, ...]:
        """Name the carriage and finger transmissions consistently."""
        return tuple(f"{CableRigName.CARRIAGE}_{axis}" for axis in range(3)) + (
            CableRigName.LEFT_PAD,
            CableRigName.RIGHT_PAD,
        )

    def _build_apparatus(self, spec: mujoco.MjSpec) -> None:
        """Add a narrow cable support and a Cartesian parallel-jaw gripper."""
        cable_start = self.cable.root_T_anchor[:3, 3]
        cable_rotation = self.cable.root_T_anchor[:3, :3]
        self._rotation = cable_rotation
        direction = cable_rotation[:, 0]
        self._origin = cable_start + direction * (
            self.config.segment_count * self.config.segment_length / 2.0
        )
        self._target = self._origin.copy()
        self._support_size = np.array(
            [
                self.config.segment_count * self.config.segment_length
                + 2.0 * self.parameters.support_margin,
                self.config.radius,
                self.parameters.support_depth,
            ]
        )
        self._support_position = self._origin.copy()
        self._support_position -= cable_rotation[:, 2] * (
            self.config.radius + self.parameters.support_depth / 2.0
        )
        quaternion = Rotation.from_matrix(cable_rotation).as_quat(scalar_first=True)
        table = spec.worldbody.add_body(
            name=CableRigName.TABLE, pos=self._support_position, quat=quaternion
        )
        table.add_geom(
            name=CableRigName.TABLE,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self._support_size / 2.0,
            friction=self.config.friction,
            rgba=(0.28, 0.36, 0.43, 1.0),
        )
        spec.worldbody.add_geom(
            name=CableRigName.FLOOR,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=(0.0, 0.0, self.parameters.mount_height - self.parameters.floor_depth),
            size=(1.0, 1.0, 0.01),
            rgba=(0.1, 0.12, 0.15, 1.0),
        )
        carriage = spec.worldbody.add_body(
            name=CableRigName.CARRIAGE, pos=self._origin, quat=quaternion
        )
        carriage.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.01, 0.0, 0.0),
            mass=0.2,
            contype=0,
            conaffinity=0,
            rgba=(0.0, 0.0, 0.0, 0.0),
        )
        carriage.gravcomp = 1.0
        for axis, name in enumerate(self._actuator_names()[:3]):
            carriage.add_joint(
                name=name,
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=np.eye(3)[axis],
                damping=0.0,
            )
            self._add_servo(
                spec,
                name,
                self.parameters.carriage_stiffness,
                self.parameters.carriage_damping,
                self.parameters.carriage_force,
            )
        for sign, name in (
            (1.0, CableRigName.LEFT_PAD),
            (-1.0, CableRigName.RIGHT_PAD),
        ):
            center_distance = (
                self.config.radius
                + self.parameters.open_clearance
                + self.parameters.pad_half_size[1]
            )
            pad = carriage.add_body(name=name, pos=(0.0, sign * center_distance, 0.0))
            pad.gravcomp = 1.0
            pad.add_joint(
                name=name,
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=(0.0, -sign, 0.0),
                damping=0.0,
            )
            pad.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=self.parameters.pad_half_size,
                mass=0.03,
                friction=self.config.friction,
                condim=self.config.condim,
                solref=self.config.solref,
                solimp=self.config.solimp,
                rgba=(0.24, 0.67, 0.91, 1.0),
            )
            self._add_servo(
                spec,
                name,
                self.parameters.pad_stiffness,
                self.parameters.pad_damping,
                self.parameters.pad_force,
            )

    @staticmethod
    def _add_servo(
        spec: mujoco.MjSpec,
        name: str,
        stiffness: float,
        damping: float,
        maximum_force: float,
    ) -> None:
        """Create an explicitly force-limited position actuator on a named joint."""
        spec.add_actuator(
            name=name,
            target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=(stiffness, *([0.0] * 9)),
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=(0.0, -stiffness, -damping, *([0.0] * 7)),
            forcelimited=True,
            forcerange=(-maximum_force, maximum_force),
        )

    # %% visual and contact observations
    def snapshot(self) -> dict[str, Any]:
        """Copy measured geometry, contact forces and semantic simulation status."""
        with self.simulation.simulator._model_lock:
            data = self.simulation.simulator._mj_data
            contacts = self._contact_snapshot()
            pads = [self._pad_snapshot(body) for body in self._pad_bodies]
            carriage = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, CableRigName.CARRIAGE
            )
            return {
                CableObservation.SIMULATION_TIME: float(data.time),
                CableObservation.CONFIG: asdict(self.config),
                CableObservation.SEGMENTS: self._segment_snapshot(),
                CableObservation.POINTS: [
                    {
                        CableObservation.NAME: point.name.name,
                        CableField.POSITION: self.world.compute_forward_kinematics_np(
                            self.world.root, point
                        )[:3, 3].tolist(),
                    }
                    for point in self.cable.placeholders
                ],
                CableObservation.GRIPPER: {
                    CableField.POSITION: data.xpos[carriage].tolist(),
                    CableObservation.OPENING: float(
                        np.linalg.norm(
                            data.xpos[self._pad_bodies[0]]
                            - data.xpos[self._pad_bodies[1]]
                        )
                        - 2.0 * self.parameters.pad_half_size[1]
                    ),
                    CableObservation.PADS: pads,
                },
                CableObservation.CONTACTS: contacts,
                CableObservation.ANCHOR: {
                    CableField.POSITION: self.world.compute_forward_kinematics_np(
                        self.world.root, self.cable.placeholders[0]
                    )[:3, 3].tolist(),
                    CableObservation.ENABLED: self.config.anchor_to_parent,
                },
                CableObservation.GRASP: self._grasp_snapshot(contacts),
                CableObservation.PHYSICAL: self.cable.active,
                CableObservation.WARNINGS: self._warnings(),
                CableObservation.TABLE: {
                    CableField.POSITION: self._support_position.tolist(),
                    CableObservation.SIZE: self._support_size.tolist(),
                    CableObservation.QUATERNION: Rotation.from_matrix(self._rotation)
                    .as_quat()
                    .tolist(),
                },
            }

    def _segment_snapshot(self) -> list[dict[str, Any]]:
        """Read the compiled capsule endpoints from geometry poses."""
        data = self.simulation.simulator._mj_data
        segments = []
        for name, body in zip(self.cable.physical_body_names, self._cable_bodies):
            geometry = int(self.model.body_geomadr[body])
            center = data.geom_xpos[geometry]
            direction = data.geom_xmat[geometry].reshape(3, 3)[:, 2]
            half_length = self.model.geom_size[geometry, 1]
            segments.append(
                {
                    CableObservation.NAME: name,
                    CableObservation.START: (center - half_length * direction).tolist(),
                    CableObservation.END: (center + half_length * direction).tolist(),
                }
            )
        return segments

    def _pad_snapshot(self, body: int) -> dict[str, Any]:
        """Expose each measured pad as a box with an XYZW quaternion."""
        data = self.simulation.simulator._mj_data
        return {
            CableField.POSITION: data.xpos[body].tolist(),
            CableObservation.SIZE: (
                2.0 * np.asarray(self.parameters.pad_half_size)
            ).tolist(),
            CableObservation.QUATERNION: np.roll(data.xquat[body], -1).tolist(),
        }

    def _contact_snapshot(self) -> list[dict[str, Any]]:
        """Read solved cable contact forces in each contact's local frame."""
        data = self.simulation.simulator._mj_data
        contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            first = int(self.model.geom_bodyid[contact.geom1])
            second = int(self.model.geom_bodyid[contact.geom2])
            if first not in self._cable_bodies and second not in self._cable_bodies:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, force)
            contacts.append(
                {
                    CableField.POSITION: contact.pos.tolist(),
                    CableObservation.NORMAL_FORCE: float(max(0.0, force[0])),
                    CableObservation.FRICTION_FORCE: float(np.linalg.norm(force[1:3])),
                    CableObservation.BODY_A: mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, first
                    ),
                    CableObservation.BODY_B: mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, second
                    ),
                }
            )
        return contacts

    def _grasp_snapshot(self, contacts: list[dict[str, Any]]) -> dict[str, Any]:
        """Infer pinch state from force-bearing contacts and measured relative motion."""
        pad_names = {CableRigName.LEFT_PAD, CableRigName.RIGHT_PAD}
        pad_contacts = [
            contact
            for contact in contacts
            if {contact[CableObservation.BODY_A], contact[CableObservation.BODY_B]}
            & pad_names
            and contact[CableObservation.NORMAL_FORCE] > 0.0
        ]
        touching = {
            name
            for contact in pad_contacts
            for name in (
                contact[CableObservation.BODY_A],
                contact[CableObservation.BODY_B],
            )
            if name in pad_names
        }
        state = GraspState.OPEN if not self._pinching else GraspState.SLIPPING
        if self._pinching and pad_contacts:
            state = GraspState.CONTACT
            if touching == pad_names:
                data = self.simulation.simulator._mj_data
                middle = self._cable_bodies[len(self._cable_bodies) // 2]
                relative_speed = np.linalg.norm(
                    data.cvel[middle, 3:] - data.cvel[self._pad_bodies[0], 3:]
                )
                state = (
                    GraspState.HOLDING
                    if relative_speed < self.parameters.holding_speed
                    else GraspState.SLIPPING
                )
        return {
            CableField.STATE: state,
            CableObservation.NORMAL_FORCE: sum(
                contact[CableObservation.NORMAL_FORCE] for contact in pad_contacts
            ),
            CableObservation.FRICTION_FORCE: sum(
                contact[CableObservation.FRICTION_FORCE] for contact in pad_contacts
            ),
            CableObservation.CONTACT_COUNT: len(pad_contacts),
        }

    def _warnings(self) -> list[str]:
        """Report native fallback and MuJoCo's numerical recovery warnings."""
        warnings = [self.cable.error] if self.cable.error else []
        for index, count in enumerate(
            self.simulation.simulator._mj_data.warning.number
        ):
            if count:
                warnings.append(
                    f"MuJoCo {mujoco.mjtWarning(index).name}: {int(count)} warning(s); numerical recovery may have reset the simulation"
                )
        return warnings
