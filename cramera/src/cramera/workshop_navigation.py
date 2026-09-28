"""Short fixed-heading workshop translations checked against native collision shapes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import giskardpy_bullet_bindings as bullet
import numpy as np
from typing_extensions import ClassVar

from coraplex.plans.failures import PlanFailure
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.base import BaseMotion
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianVelocityLimit,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from krrood.symbolic_math.symbolic_math import trinary_logic_and
from semantic_digital_twin.collision_checking.collision_matrix import CollisionMatrix
from semantic_digital_twin.collision_checking.pybullet_collision_detector import (
    BulletCollisionDetector,
    create_object,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Body


# %% independent collision preview
class TranslationFailureReason(StrEnum):
    """Conditions outside the fixed-posture corridor contract."""

    DRIVE = "A planar omnidirectional base is required."
    ORIENTATION = "The straight corridor requires unchanged height and orientation."
    LIMITS = "The translation or collision sampling limits are invalid."
    POLICY = "No external collision checks cover the moving robot."
    GEOMETRY = "Native Bullet collision geometry is required."
    OBSTRUCTION = "The corridor intersects native collision geometry."


@dataclass
class WorkshopTranslationBlocked(PlanFailure):
    """A short translation cannot be certified with the current posture and policy."""

    reason: TranslationFailureReason
    """Condition preventing the translation."""
    pair: tuple[str, str] | None = None
    """Colliding body names when an obstruction was measured."""

    def error_message(self) -> str:
        """Describe a failed corridor check without changing the execution world."""
        return str(self.reason) if self.pair is None else f"{self.reason} {self.pair}"


@dataclass(frozen=True)
class TranslationClearance:
    """Finite certificate for a sampled straight translation."""

    samples: int
    """Number of checked base positions, including both endpoints."""
    sample_spacing: float
    """Actual separation of adjacent sampled positions in metres."""
    minimum_clearance: float
    """Measured clearance lower bound, limited by the collision query horizon."""


@dataclass
class WorkshopTranslation:
    """Check a short base translation in an isolated copy of collision geometry.

    Native world state, model structure and callbacks remain untouched. Collision
    objects are moved only inside the temporary Bullet world. An additional half
    sample-spacing margin covers obstructions between sampled positions.
    """

    world: World
    """Native world supplying shapes, attachments and current collision policy."""
    robot: AbstractRobot
    """Omnidirectional robot translated with its current attachments and posture."""
    target: Pose
    """Requested world or relative base pose with unchanged heading."""
    maximum_sample_spacing: float = 0.005
    """Largest permitted interval between collision samples in metres."""
    clearance: float = 0.0005
    """Extra clearance beyond the between-sample coverage margin in metres."""
    maximum_distance: float = 2.5
    """Upper bound on the supported straight segment length in metres."""

    def collision_matrix(self) -> CollisionMatrix:
        """Apply existing ordered collision rules to a private query matrix."""
        manager = self.world.collision_manager
        matrix = CollisionMatrix()
        for rule in manager.rules:
            rule.update(self.world)
        for rule in (
            manager.default_rules
            + manager.temporary_rules
            + manager.ignore_collision_rules
        ):
            rule.apply_to_collision_matrix(matrix)
        return matrix

    def world_target(self) -> Pose:
        """Resolve and validate a bounded translation without a heading change."""
        if not isinstance(self.robot.drive, OmniDrive):
            raise WorkshopTranslationBlocked(TranslationFailureReason.DRIVE)
        start = self.robot.root.global_pose.to_np()
        target = self.world.transform(self.target, self.world.root)
        value = target.to_np()
        distance = float(np.linalg.norm(value[:3, 3] - start[:3, 3]))
        if not (
            np.isfinite(value).all()
            and 0 < self.maximum_sample_spacing <= 0.005
            and math.isfinite(self.clearance)
            and self.clearance >= 0
            and 0 < self.maximum_distance <= 2.5
            and distance <= self.maximum_distance
        ):
            raise WorkshopTranslationBlocked(TranslationFailureReason.LIMITS)
        if abs(value[2, 3] - start[2, 3]) > 1e-8 or not np.allclose(
            value[:3, :3], start[:3, :3], atol=1e-8, rtol=0
        ):
            raise WorkshopTranslationBlocked(TranslationFailureReason.ORIENTATION)
        return target

    def validate(self) -> TranslationClearance:
        """Certify a fixed-heading corridor while retaining every original native pose."""
        with self.world.state.world_lock:
            target = self.world_target()
            displacement = (
                target.to_np()[:3, 3] - self.robot.root.global_pose.to_np()[:3, 3]
            )
            distance = float(np.linalg.norm(displacement))
            intervals = max(1, math.ceil(distance / self.maximum_sample_spacing))
            spacing = distance / intervals
            guard = spacing / 2 + self.clearance
            query_distance = guard + self.maximum_sample_spacing
            moving = set(self.robot.bodies_with_collision)
            checks = [
                check
                for check in self.collision_matrix().collision_checks
                if (check.body_a in moving) != (check.body_b in moving)
            ]
            if not checks:
                raise WorkshopTranslationBlocked(TranslationFailureReason.POLICY)
            detector = self.world.collision_manager.collision_detector
            if not isinstance(detector, BulletCollisionDetector):
                raise WorkshopTranslationBlocked(TranslationFailureReason.GEOMETRY)
            bodies = {body for check in checks for body in check.bodies()}
            geometry = bullet.KineverseWorld()
            objects: dict[Body, bullet.CollisionObject] = {}
            poses = {body: body.global_pose.to_np().copy() for body in bodies}
            try:
                for body in bodies:
                    item = create_object(
                        body.id, detector.body_to_bullet_object[body].collision_shape
                    )
                    geometry.add_collision_object(item)
                    objects[body] = item
                ordered = list(bodies)
                query = {
                    (objects[check.body_a], objects[check.body_b]): query_distance
                    for check in checks
                }
                minimum = query_distance
                for index in range(intervals + 1):
                    fraction = index / intervals
                    transforms = []
                    for body in ordered:
                        pose = poses[body].copy()
                        if body in moving:
                            pose[:3, 3] += displacement * fraction
                        transforms.append(pose)
                    bullet.batch_set_transforms(
                        [objects[body] for body in ordered], np.vstack(transforms)
                    )
                    for contact in geometry.get_closest_filtered_map_batch(query):
                        separation = float(contact.contact_distance)
                        if not math.isfinite(separation) or separation < guard:
                            first = self.world.get_kinematic_structure_entity_by_id(
                                contact.obj_a.name
                            )
                            second = self.world.get_kinematic_structure_entity_by_id(
                                contact.obj_b.name
                            )
                            raise WorkshopTranslationBlocked(
                                TranslationFailureReason.OBSTRUCTION,
                                (str(first.name), str(second.name)),
                            )
                        minimum = min(minimum, separation)
                return TranslationClearance(intervals + 1, spacing, minimum)
            finally:
                for item in list(geometry.collision_objects):
                    geometry.remove_collision_object(item)


# %% native execution after clearance validation
@dataclass(eq=False, repr=False)
class WorkshopTranslationGoal(Sequence):
    """Execute a validated straight corridor with native goals and collision avoidance."""

    translation: WorkshopTranslation = field(kw_only=True)
    """Geometry certificate to recompute from the posture at controller expansion."""
    _base_goal: CartesianPose = field(init=False)
    """Continuous base controller whose observation completes the translation."""
    _joint_hold: JointPositionList | None = field(default=None, init=False)
    """Parallel goal retaining the arm and gripper posture during transport."""

    def expand(self, context: MotionStatechartContext) -> None:
        """Validate before constructing continuous base and posture-holding goals."""
        start = self.translation.robot.root.global_pose
        destination = self.translation.world.transform(
            self.translation.target, self.translation.world.root
        )
        self.translation.target = Pose(
            Point3(destination.x, destination.y, start.z),
            start.to_quaternion(),
            reference_frame=self.translation.world.root,
        )
        self.translation.validate()
        self._base_goal = CartesianPose(
            root_link=self.translation.world.root,
            tip_link=self.translation.robot.root,
            goal_pose=self.translation.world_target(),
            translation_threshold=0.001,
            orientation_threshold=0.002,
            reference_linear_velocity=0.08,
            reference_angular_velocity=0.1,
        )
        self.nodes = [self._base_goal]
        super().expand(context)
        self.add_node(
            CartesianVelocityLimit(
                root_link=self.translation.world.root,
                tip_link=self.translation.robot.root,
                max_linear_velocity=0.08,
                max_angular_velocity=0.1,
            )
        )
        joints = JointState.from_mapping(
            {
                connection: float(connection.position)
                for connection in self.translation.robot.controlled_connections
                if isinstance(connection, ActiveConnection1DOF)
            }
        )
        if len(joints):
            self._joint_hold = JointPositionList(goal_state=joints)
            self.add_node(self._joint_hold)
        if not any(
            node.parent_node is None
            for node in self.motion_statechart.get_nodes_by_type(
                ExternalCollisionAvoidance
            )
        ):
            self.add_node(ExternalCollisionAvoidance(robot=self.translation.robot))

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """Require the base destination while retaining the carried posture."""
        observed = self._base_goal.observation_variable
        if self._joint_hold is not None:
            observed = trinary_logic_and(
                observed, self._joint_hold.observation_variable
            )
        return NodeArtifacts(observation=observed)


@dataclass
class WorkshopTranslateMotion(BaseMotion):
    """Translate to an XY destination while retaining the measured height and heading."""

    target: Pose
    """Destination supplying X and Y; measured base height and orientation are retained."""
    requires_individual_execution: ClassVar[bool] = True
    """Defer corridor checks until preceding manipulation actions have finished."""

    @property
    def _motion_chart(self) -> MotionStatechartNode:
        """Build a native controller around the current posture's geometry certificate."""
        return WorkshopTranslationGoal(
            translation=WorkshopTranslation(self.world, self.robot, self.target)
        )


@dataclass
class WorkshopTranslateAction(ActionDescription):
    """Translate the workshop robot without turning its loaded posture."""

    target: Pose
    """XY destination reached without changing the measured base height or heading."""

    @property
    def _action_plan(self) -> PlanNode:
        """Execute one modular native base motion with a deferred geometry check."""
        return execute_single(
            WorkshopTranslateMotion(self.target), context=self.context
        )
