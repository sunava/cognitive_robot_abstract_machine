"""One-way contact dynamics for loose screws during native box retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from typing_extensions import TYPE_CHECKING

from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import ActionNode, PlanNode
from cramera.workshop_physics import WorkshopPhysics
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

if TYPE_CHECKING:
    from coraplex.robot_plans.actions.base import ActionDescription
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart

# %% coupling contract and measurements


class InvalidContentsCoupling(ValueError):
    """The native bodies cannot receive independent measured contents poses."""


class UnpreparedContents(InvalidContentsCoupling):
    """The carrier hierarchy was not prepared before constructing its native plan."""


class ContentsModel(StrEnum):
    """Physical scope recorded alongside native controller acceptance metrics."""

    RELATIVE_CONTACT = "one-way-relative-contact-contents"
    """Independent MuJoCo contents expressed in a separately controlled native box."""


@dataclass
class ContentsMeasurements:
    """Measured contact integration and mismatch between the two carrier poses."""

    model: ContentsModel = ContentsModel.RELATIVE_CONTACT
    """Explicit one-way coupling model used for this recording."""
    ticks: int = 0
    """Number of native control periods integrated by the contact world."""
    simulated_seconds: float = 0.0
    """MuJoCo time integrated during retrieval, including initial settling."""
    maximum_box_position_lag: float = 0.0
    """Largest carrier position mismatch before relative projection, in metres."""
    maximum_box_angular_lag: float = 0.0
    """Largest carrier orientation mismatch before projection, in radians."""
    maximum_relative_displacement: float = 0.0
    """Largest independent screw translation within the box frame, in metres."""
    maximum_projection_correction: float = 0.0
    """Largest difference between projected and absolute measured screw positions."""
    retained_contents: bool = False
    """Whether every measured screw remained inside the cavity at retrieval end."""
    escaped_contents: tuple[str, ...] = ()
    """Screws whose collision bounds exceeded the carrier cavity at retrieval end."""
    contents_count: int = 0
    """Number of independent screws included in the contact integration."""


@dataclass(frozen=True)
class CollisionBounds:
    """Axis-aligned collision limits expressed in a common frame."""

    lower: np.ndarray
    """Minimum coordinates along each frame axis."""
    upper: np.ndarray
    """Maximum coordinates along each frame axis."""

    def contains(self, other: CollisionBounds, tolerance: float) -> bool:
        """Allow only the configured resting-contact penetration at the boundary."""
        return bool(
            np.all(other.lower >= self.lower - tolerance)
            and np.all(other.upper <= self.upper + tolerance)
        )


# %% independent contact contents in a native carrier
@dataclass
class WorkshopContents(PlanCallback):
    """Project independently simulated contents into the native carrier frame.

    The robot and carrier retain their controller poses. The compliant MuJoCo
    carrier follows that measured trajectory, while contact-resolved screw poses
    are transferred relative to it. Contact forces do not feed back to the robot.
    """

    world: World = field(kw_only=True)
    """Native planning world receiving only free screw pose updates."""
    physics: WorkshopPhysics = field(kw_only=True)
    """Separate contact world containing the matching carrier and loose screws."""
    box: Body = field(kw_only=True)
    """Native carrier whose pose remains entirely controlled by the robot plan."""
    contents: tuple[Body, ...] = field(kw_only=True)
    """Independent native screws receiving measured relative contact motion."""
    control_dt: float = field(kw_only=True)
    """Simulated seconds per native executor tick, supplied by its configuration."""
    action: ActionDescription | None = field(default=None, kw_only=True)
    """Exact retrieval action whose start and completion bound the coupling."""
    initial_settle_seconds: float = field(default=0.25, kw_only=True)
    """Initial gravity settling time before beginning native carrier tracking."""
    final_settle_seconds: float = field(default=0.35, kw_only=True)
    """Contact settling time at the stationary destination before freezing contents."""
    retention_tolerance: float = field(default=0.0005, kw_only=True)
    """Allowed resting-contact penetration into the cavity boundary, in metres."""
    active: bool = field(default=False, init=False)
    """Whether native motion ticks currently advance and publish contact states."""
    measurements: ContentsMeasurements = field(
        default_factory=ContentsMeasurements, init=False
    )
    """Observed integration time, slip and frame-projection mismatch."""
    initial_relative_positions: dict[str, np.ndarray] = field(
        default_factory=dict, init=False
    )
    """Measured per-screw positions in the carrier frame after initial settling."""

    @property
    def box_key(self) -> str:
        """Return the common native and MuJoCo carrier identifier."""
        return str(self.box.name)

    @staticmethod
    def matrix(pose: np.ndarray) -> np.ndarray:
        """Convert the contact world's position and xyzw quaternion into a matrix."""
        result = np.eye(4)
        result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        result[:3, 3] = pose[:3]
        return result

    def validate_contents(self) -> None:
        """Reject attached or unrelated bodies before changing any native pose."""
        if self.box_key not in self.physics.objects or not self.contents:
            raise InvalidContentsCoupling(
                "A matching carrier and loose screw contents are required."
            )
        if len({str(body.name) for body in self.contents}) != len(self.contents):
            raise InvalidContentsCoupling(
                "Contents must contain distinct independent screws."
            )
        for body in self.contents:
            if str(body.name) not in self.physics.geometry.screws:
                raise InvalidContentsCoupling(
                    "Only declared workshop screws may receive contact updates."
                )
            parent = self.box if self.active else self.world.root
            if (
                not isinstance(body.parent_connection, Connection6DoF)
                or body.parent_connection.parent is not parent
            ):
                raise InvalidContentsCoupling(
                    "Contents must remain independent free bodies in their carrier frame."
                )

    def reparent_contents(self, parent: KinematicStructureEntity) -> None:
        """Preserve independent free connections and world poses in one model change."""
        with self.world.batch_state_changes():
            with self.world.modify_world():
                for body in self.contents:
                    self.world.move_branch(
                        body, parent, enable_unsafe_inside_world_block=True
                    )

    def start(self) -> None:
        """Prepare the carrier hierarchy before constructing or compiling native plans.

        Reparenting changes degree-of-freedom storage, so this explicit preparation
        must precede goal binding. Repeated calls while active preserve that model.
        """
        if self.active:
            return
        self.validate_contents()
        self.measurements.contents_count = len(self.contents)
        before = float(self.physics.data.time)
        self.physics.step(self.initial_settle_seconds)
        self.measurements.simulated_seconds += float(self.physics.data.time) - before
        self.physics.grab(self.box_key)
        self.reparent_contents(self.box)
        self.active = True
        measured_world_T_box = self.matrix(self.physics.pose(self.box_key))
        box_T_measured_world = np.linalg.inv(measured_world_T_box)
        self.initial_relative_positions = {
            str(body.name): (
                box_T_measured_world @ self.matrix(self.physics.pose(str(body.name)))
            )[:3, 3].copy()
            for body in self.contents
        }
        self.publish_contents()

    def advance(self, duration: float) -> None:
        """Integrate one carrier-following interval and publish independent screw poses."""
        if not self.active:
            return
        if not math.isfinite(duration) or not 0 < duration <= 30:
            raise InvalidContentsCoupling(
                "A contact interval must be positive, finite and no longer than 30 seconds."
            )
        self.validate_contents()
        self.physics.move(self.box.global_pose.to_position_quaternion_list())
        before = float(self.physics.data.time)
        self.physics.step(duration)
        self.measurements.simulated_seconds += float(self.physics.data.time) - before
        self.measurements.ticks += 1
        self.publish_contents()

    def publish_contents(self) -> None:
        """Transfer measured box-relative poses without moving the native carrier."""
        self.validate_contents()
        native_world_T_box = self.box.global_pose.to_np()
        measured_world_T_box = self.matrix(self.physics.pose(self.box_key))
        box_T_measured_world = np.linalg.inv(measured_world_T_box)
        self.measurements.maximum_box_position_lag = max(
            self.measurements.maximum_box_position_lag,
            float(
                np.linalg.norm(native_world_T_box[:3, 3] - measured_world_T_box[:3, 3])
            ),
        )
        self.measurements.maximum_box_angular_lag = max(
            self.measurements.maximum_box_angular_lag,
            float(
                Rotation.from_matrix(
                    native_world_T_box[:3, :3].T @ measured_world_T_box[:3, :3]
                ).magnitude()
            ),
        )
        projected: list[tuple[Body, np.ndarray]] = []
        for body in self.contents:
            measured_world_T_screw = self.matrix(self.physics.pose(str(body.name)))
            box_T_screw = box_T_measured_world @ measured_world_T_screw
            native_world_T_screw = native_world_T_box @ box_T_screw
            self.measurements.maximum_relative_displacement = max(
                self.measurements.maximum_relative_displacement,
                float(
                    np.linalg.norm(
                        box_T_screw[:3, 3]
                        - self.initial_relative_positions[str(body.name)]
                    )
                ),
            )
            self.measurements.maximum_projection_correction = max(
                self.measurements.maximum_projection_correction,
                float(
                    np.linalg.norm(
                        native_world_T_screw[:3, 3] - measured_world_T_screw[:3, 3]
                    )
                ),
            )
            parent_T_screw = (
                box_T_screw
                if body.parent_connection.parent is self.box
                else native_world_T_screw
            )
            projected.append((body, parent_T_screw))
        with self.world.batch_state_changes():
            for body, parent_T_screw in projected:
                body.parent_connection.origin = HomogeneousTransformationMatrix(
                    data=parent_T_screw,
                    reference_frame=body.parent_connection.parent,
                    child_frame=body,
                )

    def geometry_indices(self, key: str) -> range:
        """Return collision geometries attached directly to an independent body."""
        body = self.physics.model.body(key)
        start = int(body.geomadr[0])
        return range(start, start + int(body.geomnum[0]))

    def cavity_bounds(self) -> CollisionBounds:
        """Derive the open cavity from the compiled floor and four wall boxes."""
        model = self.physics.model
        indices = list(self.geometry_indices(self.box_key))
        if len(indices) != 5 or any(
            model.geom_type[index] != mujoco.mjtGeom.mjGEOM_BOX
            or not np.allclose(model.geom_quat[index], [1, 0, 0, 0])
            for index in indices
        ):
            raise InvalidContentsCoupling(
                "The carrier requires an axis-aligned floor and four wall boxes."
            )
        floor = min(indices, key=lambda index: model.geom_size[index, 2])
        lower = model.geom_pos[floor] - model.geom_size[floor]
        upper = model.geom_pos[floor] + model.geom_size[floor]
        lower[2] = upper[2]
        walls = [index for index in indices if index != floor]
        upper[2] = min(
            model.geom_pos[index, 2] + model.geom_size[index, 2] for index in walls
        )
        for index in walls:
            axis = int(np.argmin(model.geom_size[index, :2]))
            center = model.geom_pos[index, axis]
            extent = model.geom_size[index, axis]
            if center < model.geom_pos[floor, axis]:
                lower[axis] = max(lower[axis], center + extent)
            else:
                upper[axis] = min(upper[axis], center - extent)
        return CollisionBounds(lower=lower, upper=upper)

    def collision_bounds(self, key: str) -> CollisionBounds:
        """Bound measured primitive collisions in the measured carrier frame."""
        model, data = self.physics.model, self.physics.data
        world_T_box = self.matrix(self.physics.pose(self.box_key))
        box_R_world = world_T_box[:3, :3].T
        lower, upper = [], []
        for index in self.geometry_indices(key):
            center = box_R_world @ (data.geom_xpos[index] - world_T_box[:3, 3])
            rotation = box_R_world @ data.geom_xmat[index].reshape(3, 3)
            size = model.geom_size[index]
            match model.geom_type[index]:
                case mujoco.mjtGeom.mjGEOM_BOX:
                    extent = np.abs(rotation) @ size
                case mujoco.mjtGeom.mjGEOM_CYLINDER:
                    extent = size[0] * np.linalg.norm(rotation[:, :2], axis=1) + size[
                        1
                    ] * np.abs(rotation[:, 2])
                case mujoco.mjtGeom.mjGEOM_SPHERE:
                    extent = np.full(3, size[0])
                case _:
                    raise InvalidContentsCoupling(
                        "Unsupported loose-part collision primitive."
                    )
            lower.append(center - extent)
            upper.append(center + extent)
        return CollisionBounds(lower=np.min(lower, axis=0), upper=np.max(upper, axis=0))

    def measure_retention(self) -> None:
        """Record whether every independent collision shape fits inside the cavity."""
        cavity = self.cavity_bounds()
        self.measurements.escaped_contents = tuple(
            str(body.name)
            for body in self.contents
            if not cavity.contains(
                self.collision_bounds(str(body.name)), self.retention_tolerance
            )
        )
        self.measurements.retained_contents = not self.measurements.escaped_contents

    def stop(self, settle_seconds: float | None = None) -> None:
        """Settle at the destination and leave free poses for subsequent native actions."""
        if not self.active:
            return
        duration = (
            self.final_settle_seconds if settle_seconds is None else settle_seconds
        )
        if duration:
            self.advance(duration)
        self.measure_retention()
        self.reparent_contents(self.world.root)
        self.physics.release()
        self.active = False
        if not self.measurements.retained_contents:
            raise InvalidContentsCoupling(
                "Screws escaped the carrier: "
                + ", ".join(self.measurements.escaped_contents)
            )

    def on_start(self, node: PlanNode) -> None:
        """Require explicit preparation without changing a compiled plan's model."""
        if (
            isinstance(node, ActionNode)
            and node.designator is self.action
            and not self.active
        ):
            raise UnpreparedContents(
                "Call WorkshopContents.start() before constructing the native plan."
            )

    def on_end(self, node: PlanNode) -> None:
        """Freeze contact contents before subsequent screw or tool pickup begins."""
        if isinstance(node, ActionNode) and node.designator is self.action:
            self.stop()

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        """Match contact integration to the native simulated control period."""
        if self.active:
            self.advance(self.control_dt)
