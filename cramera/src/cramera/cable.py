"""Native MuJoCo composite cables with semantic point references."""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from semantic_digital_twin.adapters.multi_sim import MujocoBuilder, MujocoSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.mixin import SimulatorAdditionalProperty
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body, Connection


# %% cable configuration
@dataclass(frozen=True)
class CableConfig:
    """Geometry, total mass, elastic moduli, and cable contact parameters."""

    segment_count: int = 10
    """Number of rigid capsule elements."""
    segment_length: float = 0.03
    """Length of each rigid element in metres."""
    radius: float = 0.005
    """Capsule radius in metres."""
    mass: float = 0.1
    """Total cable mass in kilograms."""
    color: Color = field(default_factory=lambda: Color(1, 0, 0, 1))
    """Uniform colour of the generated physical geoms."""
    bend: float = 1e5
    """Bending modulus in pascals."""
    twist: float = 1e5
    """Twisting modulus in pascals."""
    anchor_to_parent: bool = True
    """Constrain the first endpoint through a native ball joint on its parent."""
    anchor_offset: tuple[float, float, float] = (0, 0, 0)
    """Initial endpoint translation in the parent frame."""
    anchor_rpy: tuple[float, float, float] = (0, 0, 0)
    """Initial cable rotation in parent-frame roll, pitch, and yaw radians."""
    friction: tuple[float, float, float] = (1, 0.005, 0.0001)
    """Sliding, torsional, and rolling contact friction coefficients."""
    condim: int = 6
    """MuJoCo contact dimensionality."""
    solref: tuple[float, float] = (0.005, 1)
    """Contact solver time constant and damping ratio."""
    solimp: tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2)
    """Contact solver impedance parameters."""

    def __post_init__(self) -> None:
        """Reject invalid geometry and solver settings before modifying a world."""
        if type(self.segment_count) is not int or self.segment_count < 2:
            raise ValueError("segment_count must be an integer of at least two")
        for value in (self.segment_length, self.radius, self.mass):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Cable dimensions and mass must be finite and positive"
                )
        for value in (self.bend, self.twist):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("Cable stiffness must be finite and nonnegative")
        if type(self.anchor_to_parent) is not bool:
            raise ValueError("anchor_to_parent must be boolean")
        for vector, length in (
            (self.anchor_offset, 3),
            (self.anchor_rpy, 3),
            (self.friction, 3),
            (self.solref, 2),
            (self.solimp, 5),
        ):
            if len(vector) != length or any(
                isinstance(value, bool) or not math.isfinite(value) for value in vector
            ):
                raise ValueError(
                    "Cable contact and pose vectors must have finite values and valid dimensions"
                )
        if type(self.condim) is not int or self.condim not in (1, 3, 4, 6):
            raise ValueError(
                "condim must be one of MuJoCo's supported contact dimensions"
            )
        if min(self.friction) < 0 or min(self.solref) <= 0:
            raise ValueError(
                "Friction must be nonnegative and solver reference positive"
            )
        if not (
            0 < self.solimp[0] <= self.solimp[1] < 1
            and self.solimp[2] > 0
            and 0 < self.solimp[3] < 1
            and self.solimp[4] >= 1
        ):
            raise ValueError("Invalid contact solver impedance")
        if not isinstance(self.color, Color) or any(
            not math.isfinite(value) or not 0 <= value <= 1
            for value in self.color.to_rgba()
        ):
            raise ValueError(
                "color must contain finite RGBA values between zero and one"
            )


class CableName(StrEnum):
    """Names shared by native model generation and semantic references."""

    ROOT = "world"
    """MuJoCo's root body name."""
    PREFIX = "cable_segment_"
    """Default namespace for cable points and native elements."""
    PLUGIN = "mujoco.elasticity.cable"
    """MuJoCo's distributed bending and twisting plugin."""
    HOME = "home"
    """Initial-state keyframe written by the world builder."""


class CableSpecMethod(StrEnum):
    """Native specification operations required for atomic cable construction."""

    COPY = "copy"
    """Create a candidate without changing the live specification."""
    ATTACH = "attach"
    """Insert the expanded native composite into its parent frame."""
    PARSE = "from_string"
    """Expand the composite MJCF macro."""
    RECOMPILE = "recompile"
    """Compile the extended scene."""


@dataclass
class CablePoint(SimulatorAdditionalProperty):
    """Mark a semantic point that must not become a second physical body."""


@dataclass
class CableCompilation:
    """A successfully extended simulation or its unchanged fallback state."""

    spec: mujoco.MjSpec
    """Specification corresponding to the returned model."""
    model: mujoco.MjModel
    """Compiled physical scene."""
    data: mujoco.MjData
    """Simulation state preserved across the model extension."""
    success: bool
    """Whether the native cable is present."""


# %% semantic cable representation
@dataclass
class Cable(SimulatorAdditionalProperty):
    """A native rigid cable and the semantic points mirroring its link frames."""

    world: World
    """Semantic scene containing the cable's point bodies."""
    config: CableConfig
    """Geometry, mechanics, and initial placement."""
    parent_body: Body | None
    """Optional parent supplying the initial pose and anchored endpoint frame."""
    prefix: str
    """Unique namespace for generated model elements."""
    placeholders: tuple[Body, ...]
    """Semantic bodies corresponding to the physical link origins."""
    connections: tuple[Connection6DoF, ...]
    """Root-relative semantic point placements."""
    physical_body_names: tuple[str, ...]
    """Stable names of MuJoCo's generated rigid link bodies."""
    root_T_anchor: np.ndarray
    """Initial anchor pose expressed in the semantic root frame."""
    active: bool = False
    """Whether native physical construction succeeded."""
    error: str | None = None
    """Diagnostic explaining a placeholder-only fallback."""

    @classmethod
    def create(
        cls,
        world: World,
        config: CableConfig | None = None,
        parent_body: Body | None = None,
        prefix: str = CableName.PREFIX,
    ) -> Cable:
        """Add freely referenceable points after establishing a valid world root."""
        config = config or CableConfig()
        if not prefix or any(character.isspace() for character in prefix):
            raise ValueError("Cable prefix must be nonempty and contain no whitespace")
        if world.root is None:
            with world.modify_world():
                world.add_kinematic_structure_entity(
                    Body(name=PrefixedName(CableName.ROOT))
                )
        root = world.root
        if parent_body is not None and parent_body._world is None:
            with world.modify_world():
                world.add_kinematic_structure_entity(parent_body)
                world.add_connection(FixedConnection(parent=root, child=parent_body))
        if parent_body is not None and parent_body._world is not world:
            raise ValueError("Cable parent must belong to the cable world")
        names = {body.name.name for body in world.bodies}
        if any(f"{prefix}{index}" in names for index in range(config.segment_count)):
            raise ValueError("Cable prefix is already present in the world")
        try:
            root_T_parent = (
                np.eye(4)
                if parent_body is None
                else world.compute_forward_kinematics(root, parent_body).to_np()
            )
            parent_T_anchor = HomogeneousTransformationMatrix.from_xyz_rpy(
                *config.anchor_offset, *config.anchor_rpy
            ).to_np()
            root_T_anchor = root_T_parent @ parent_T_anchor
        except (ValueError, RuntimeError, AssertionError) as error:
            logging.getLogger(__name__).warning(
                "Cable anchor pose unavailable; using world origin: %s", error
            )
            root_T_anchor = np.eye(4)
        placeholders = []
        connections = []
        with world.modify_world():
            for index in range(config.segment_count):
                point = Body(
                    name=PrefixedName(f"{prefix}{index}"),
                    simulator_additional_properties=[CablePoint()],
                )
                world.add_kinematic_structure_entity(point)
                connection = Connection6DoF.create_with_dofs(world, root, point)
                world.add_connection(connection)
                placeholders.append(point)
                connections.append(connection)
        cable = cls(
            world,
            config,
            parent_body,
            prefix,
            tuple(placeholders),
            tuple(connections),
            tuple(f"{prefix}B{index}" for index in range(config.segment_count)),
            root_T_anchor,
        )
        with world.batch_state_changes():
            for index, connection in enumerate(cable.connections):
                point_T_anchor = root_T_anchor.copy()
                point_T_anchor[:3, 3] += (
                    root_T_anchor[:3, 0] * index * config.segment_length
                )
                cable._set_point_pose(
                    connection,
                    point_T_anchor[:3, 3],
                    Rotation.from_matrix(point_T_anchor[:3, :3]).as_quat(),
                )
        world.notify_state_change()
        world.simulator_additional_properties.append(cable)
        return cable

    def _set_point_pose(
        self, connection: Connection6DoF, position: np.ndarray, quaternion: np.ndarray
    ) -> None:
        """Write a point's root-relative transform to its semantic degrees of freedom."""
        for degree, value in zip(
            (
                connection.x,
                connection.y,
                connection.z,
                connection.qx,
                connection.qy,
                connection.qz,
                connection.qw,
            ),
            (*position, *quaternion),
        ):
            self.world.state[degree.id].position = float(value)

    def sync(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Mirror physical link transforms into semantic points without moving physics."""
        if not self.active:
            return
        with self.world.batch_state_changes(publish_changes=False):
            for connection, name in zip(self.connections, self.physical_body_names):
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id < 0:
                    raise ValueError(f"Physical cable body is missing: {name}")
                self._set_point_pose(
                    connection, data.xpos[body_id], np.roll(data.xquat[body_id], -1)
                )
        self.world.notify_state_change()

    def _native_spec(self) -> mujoco.MjSpec:
        """Expand MuJoCo's native composite macro into rigid capsule links."""
        root = ElementTree.Element("mujoco")
        extension = ElementTree.SubElement(root, "extension")
        ElementTree.SubElement(extension, "plugin", plugin=CableName.PLUGIN)
        worldbody = ElementTree.SubElement(root, "worldbody")
        composite = ElementTree.SubElement(
            worldbody,
            "composite",
            type="cable",
            prefix=self.prefix,
            count=f"{self.config.segment_count + 1} 1 1",
            curve="s",
            size=str(self.config.segment_count * self.config.segment_length),
            initial="ball" if self.config.anchor_to_parent else "free",
        )
        plugin = ElementTree.SubElement(composite, "plugin", plugin=CableName.PLUGIN)
        for key, value in (
            ("bend", self.config.bend),
            ("twist", self.config.twist),
            ("vmax", 0),
        ):
            ElementTree.SubElement(plugin, "config", key=key, value=str(value))
        ElementTree.SubElement(
            composite,
            "geom",
            type="capsule",
            size=str(self.config.radius),
            mass=str(self.config.mass / self.config.segment_count),
            rgba=" ".join(map(str, self.config.color.to_rgba())),
            friction=" ".join(map(str, self.config.friction)),
            condim=str(self.config.condim),
            solref=" ".join(map(str, self.config.solref)),
            solimp=" ".join(map(str, self.config.solimp)),
        )
        native = mujoco.MjSpec.from_string(
            ElementTree.tostring(root, encoding="unicode")
        )
        renamed = {}
        for body, name in zip(native.bodies[1:], self.physical_body_names):
            renamed[body.name] = name
            body.name = name
        for exclusion in native.excludes:
            exclusion.bodyname1 = renamed[exclusion.bodyname1]
            exclusion.bodyname2 = renamed[exclusion.bodyname2]
        return native

    def install(
        self, spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData
    ) -> CableCompilation:
        """Recompile an isolated candidate and preserve the live scene on failure."""
        if self.active:
            raise ValueError("Cable is already installed")
        if not set(CableSpecMethod).issubset(dir(type(spec))):
            self.error = "Installed MuJoCo lacks the native cable specification API"
            logging.getLogger(__name__).warning(
                "%s; retaining semantic points only", self.error
            )
            return CableCompilation(spec, model, data, False)
        try:
            candidate = spec.copy()
            parent = candidate.worldbody
            root_T_anchor = self.root_T_anchor
            if self.config.anchor_to_parent and self.parent_body is not None:
                parent = candidate.body(self.parent_body.name.name)
                if parent is None:
                    raise ValueError(
                        "Cable anchor parent is absent from the native model"
                    )
                root_T_parent = self.world.compute_forward_kinematics(
                    self.world.root, self.parent_body
                ).to_np()
                root_T_anchor = np.linalg.inv(root_T_parent) @ self.root_T_anchor
            frame = parent.add_frame(
                pos=root_T_anchor[:3, 3],
                quat=Rotation.from_matrix(root_T_anchor[:3, :3]).as_quat(
                    scalar_first=True
                ),
            )
            candidate.attach(self._native_spec(), prefix="", frame=frame)
            compiled_model, compiled_data = candidate.recompile(model, data)
            mujoco.mj_resetData(compiled_model, compiled_data)
            self._restore_scene_state(model, data, compiled_model, compiled_data)
            mujoco.mj_forward(compiled_model, compiled_data)
        except (ValueError, RuntimeError) as error:
            self.error = str(error)
            logging.getLogger(__name__).warning(
                "Native cable unavailable; retaining semantic points only: %s", error
            )
            return CableCompilation(spec, model, data, False)
        self.active = True
        self.error = None
        self.sync(compiled_model, compiled_data)
        return CableCompilation(candidate, compiled_model, compiled_data, True)

    @staticmethod
    def _restore_scene_state(
        previous_model: mujoco.MjModel,
        previous_data: mujoco.MjData,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        """Preserve named world joints and actuators after recompiling a copied spec."""
        for joint_id in range(previous_model.njnt):
            name = previous_model.joint(joint_id).name
            target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if target < 0:
                raise ValueError(
                    f"Cannot preserve unnamed or missing world joint: {name}"
                )
            position_start = previous_model.jnt_qposadr[joint_id]
            position_end = (
                previous_model.jnt_qposadr[joint_id + 1]
                if joint_id + 1 < previous_model.njnt
                else previous_model.nq
            )
            velocity_start = previous_model.jnt_dofadr[joint_id]
            velocity_end = (
                previous_model.jnt_dofadr[joint_id + 1]
                if joint_id + 1 < previous_model.njnt
                else previous_model.nv
            )
            target_position = model.jnt_qposadr[target]
            target_velocity = model.jnt_dofadr[target]
            data.qpos[
                target_position : target_position + position_end - position_start
            ] = previous_data.qpos[position_start:position_end]
            data.qvel[
                target_velocity : target_velocity + velocity_end - velocity_start
            ] = previous_data.qvel[velocity_start:velocity_end]
            data.qfrc_applied[
                target_velocity : target_velocity + velocity_end - velocity_start
            ] = previous_data.qfrc_applied[velocity_start:velocity_end]
        for equality_id in range(previous_model.neq):
            name = previous_model.equality(equality_id).name
            target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
            if target < 0:
                raise ValueError(
                    f"Cannot preserve unnamed or missing world equality: {name}"
                )
            data.eq_active[target] = previous_data.eq_active[equality_id]
        for actuator_id in range(previous_model.nu):
            name = previous_model.actuator(actuator_id).name
            target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if target < 0:
                raise ValueError(
                    f"Cannot preserve unnamed or missing world actuator: {name}"
                )
            data.ctrl[target] = previous_data.ctrl[actuator_id]
            count = previous_model.actuator_actnum[actuator_id]
            if count:
                start = previous_model.actuator_actadr[actuator_id]
                target_start = model.actuator_actadr[target]
                data.act[target_start : target_start + count] = previous_data.act[
                    start : start + count
                ]
        for body_id in range(previous_model.nbody):
            name = previous_model.body(body_id).name
            target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if target >= 0:
                data.xfrc_applied[target] = previous_data.xfrc_applied[body_id]
                mocap_id = previous_model.body_mocapid[body_id]
                if mocap_id >= 0:
                    target_mocap = model.body_mocapid[target]
                    data.mocap_pos[target_mocap] = previous_data.mocap_pos[mocap_id]
                    data.mocap_quat[target_mocap] = previous_data.mocap_quat[mocap_id]
        data.time = previous_data.time


# %% world builder and synchronization hooks
@dataclass
class CableMujocoBuilder(MujocoBuilder):
    """Build ordinary world bodies while keeping cable points purely semantic."""

    @staticmethod
    def _is_point(body: Body) -> bool:
        """Identify semantic point markers independently of their naming prefix."""
        return any(
            isinstance(prop, CablePoint)
            for prop in body.simulator_additional_properties
        )

    def build_body(self, body: Body) -> None:
        """Exclude semantic cable bodies from physical mass and geometry generation."""
        if not self._is_point(body):
            super().build_body(body)

    def _build_connection(self, connection: Connection) -> None:
        """Exclude semantic cable joints from the physical coordinate vector."""
        if not self._is_point(connection.child):
            super()._build_connection(connection)

    def _end_build(self, file_path: str) -> None:
        """Keep the home keyframe aligned with the filtered physical joint set."""
        super()._end_build(file_path)
        tree = ElementTree.parse(file_path)
        model = self.spec.compile()
        positions = model.qpos0.copy()
        for body in self.world.bodies:
            connection = body.parent_connection
            if (
                self._is_point(body)
                or connection is None
                or isinstance(connection, self._ignore_connection_types)
            ):
                continue
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, connection.name.name
            )
            if joint_id < 0:
                continue
            address = model.jnt_qposadr[joint_id]
            if isinstance(connection, Connection6DoF):
                root_T_body = self.world.compute_forward_kinematics(
                    self.world.root, body
                ).to_np()
                positions[address : address + 3] = root_T_body[:3, 3]
                positions[address + 3 : address + 7] = Rotation.from_matrix(
                    root_T_body[:3, :3]
                ).as_quat(scalar_first=True)
            else:
                coordinates = [
                    self.world.state[degree.id].position
                    for degree in connection.active_dofs + connection.passive_dofs
                ]
                positions[address : address + len(coordinates)] = coordinates
        key = tree.find("keyframe/key")
        key.set("qpos", " ".join(map(str, positions)))
        tree.write(file_path, encoding="utf-8", xml_declaration=True)
        for prop in self.world.simulator_additional_properties:
            if isinstance(prop, Cable):
                prop.connections = tuple(
                    point.parent_connection for point in prop.placeholders
                )


@dataclass(eq=False)
class CableMujocoSynchronizer(MujocoSynchronizer):
    """Synchronize native cable poses after ordinary world degrees of freedom."""

    def _sim_to_world(self) -> None:
        """Update points at the same throttled rate while suppressing feedback."""
        previous_time = self._last_sync_time
        super()._sim_to_world()
        if self._last_sync_time == previous_time:
            return
        self._state_callback.pause()
        for prop in self._world.simulator_additional_properties:
            if isinstance(prop, Cable):
                prop.sync(self.simulator._mj_model, self.simulator._mj_data)
        self._state_callback.update_previous_world_state()
        self._state_callback.resume()
