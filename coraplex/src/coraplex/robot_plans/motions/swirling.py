"""
Continuous, constrained Giskard motion for swirling a held container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing_extensions import ClassVar

import numpy as np
from scipy.spatial.transform import Rotation

from coraplex.datastructures.enums import Arms
from coraplex.datastructures.swirling import SwirlProfile, SwirlTrajectory
from coraplex.robot_plans.motions.base import BaseMotion
from coraplex.view_manager import ViewManager
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
)
from giskardpy.motion_statechart.tasks.feature_functions import AngleGoal
import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% Held container validation


@dataclass
class ContainerNotHeld(ValueError):
    """
    The selected container is not rigidly attached to the requested gripper.
    """

    container: Body
    """Container whose connection to the gripper is missing."""


class SwirlTaskName(StrEnum):
    """
    Stable names of the tasks and variables in a swirling motion.
    """

    MOTION = "Swirl container"
    POSE = "Swirl pose"
    TILT = "Swirl tilt limit"
    GOAL = "swirl_goal"
    ELAPSED = "swirl_elapsed"


# %% Continuous pose tracking


@dataclass(eq=False, repr=False)
class SwirlPoseTask(Task):
    """
    Track a timed conical container pose with Cartesian position and rotation
    constraints.
    """

    container: Body = field(kw_only=True)
    """
    Held container whose local positive Z axis follows the swirl cone.
    """

    tip_link: Body = field(kw_only=True)
    """
    Gripper tool frame whose measured container transform is preserved.
    """

    root_link: Body = field(kw_only=True)
    """
    Stationary reference frame for the motion.
    """

    container_P_pivot: Point3 = field(kw_only=True)
    """
    Stationary pivot in the container frame.
    """

    profile: SwirlProfile = field(kw_only=True)
    """
    Duration, cone angle and cycle count.
    """

    position_threshold: float = field(default=0.003, kw_only=True)
    """
    Final position tolerance in meters.
    """

    orientation_threshold: float = field(default=0.025, kw_only=True)
    """
    Final orientation tolerance in radians.
    """

    maximum_position_error: float = field(default=0.01, kw_only=True)
    """
    Position lag in meters beyond which trajectory progress waits.
    """

    maximum_orientation_error: float = field(default=0.05, kw_only=True)
    """
    Orientation lag in radians beyond which trajectory progress waits.
    """

    reference_linear_velocity: float = field(default=0.08, kw_only=True)
    """
    Normalization speed for the position constraints in meters per second.
    """

    reference_angular_velocity: float = field(default=0.8, kw_only=True)
    """
    Normalization speed for the rotation constraints in radians per second.
    """

    root_T_goal: HomogeneousTransformationMatrix = field(init=False, repr=False)
    """
    Symbolic tool pose updated continuously along the container trajectory.
    """

    elapsed: sm.FloatVariable = field(init=False, repr=False)
    """
    Elapsed active trajectory time, excluding pauses.
    """

    _trajectory: SwirlTrajectory = field(init=False, repr=False)
    """
    Container trajectory bound to the pose at motion start.
    """

    _container_T_tool: np.ndarray = field(init=False, repr=False)
    """
    Measured grasp transform captured at motion start.
    """

    _last_cycle: float = field(default=0.0, init=False)
    """
    Last active control cycle used to advance the trajectory.
    """

    _pause_cycle: float = field(default=0.0, init=False)
    """
    Control cycle at which the current pause began.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Create position and rotation constraints with a duration-gated finish condition.
        """
        self.root_T_goal = HomogeneousTransformationMatrix.create_with_variables(
            f"{self.name}/{SwirlTaskName.GOAL}"
        )
        self.root_T_goal.reference_frame = self.root_link
        self.root_T_goal.child_frame = self.tip_link
        self.elapsed = sm.FloatVariable(f"{self.name}/{SwirlTaskName.ELAPSED}")
        context.float_variable_data.register_expression(self.root_T_goal)
        context.float_variable_data.register_expression(self.elapsed)
        self.on_start(context)
        root_T_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        artifacts = NodeArtifacts()
        artifacts.geometry.add_point_goal_constraints(
            frame_P_current=root_T_current.to_position(),
            frame_P_goal=self.root_T_goal.to_position(),
            reference_velocity=self.reference_linear_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.geometry.add_rotation_goal_constraints(
            frame_R_current=root_T_current.to_rotation_matrix(),
            frame_R_goal=self.root_T_goal.to_rotation_matrix(),
            reference_velocity=self.reference_angular_velocity,
            quadratic_weight=self.weight,
        )
        position_error = root_T_current.to_position().euclidean_distance(
            self.root_T_goal.to_position()
        )
        orientation_error = sm.abs(
            root_T_current.to_rotation_matrix().rotational_error(
                self.root_T_goal.to_rotation_matrix()
            )
        )
        artifacts.observation = sm.trinary_logic_and(
            self.elapsed >= self.profile.duration,
            position_error <= self.position_threshold,
            orientation_error <= self.orientation_threshold,
        )
        return artifacts

    def on_start(self, context: MotionStatechartContext) -> None:
        """
        Bind the container pose and grasp transform when execution starts.
        """
        self._trajectory = SwirlTrajectory(
            context.world.compute_forward_kinematics_np(self.root_link, self.container),
            self.container_P_pivot.to_np()[:3],
            self.profile,
        )
        self._container_T_tool = context.world.compute_forward_kinematics_np(
            self.container, self.tip_link
        )
        self._last_cycle = context.float_variable_data.get_value(
            context.control_cycle_variable
        )
        self._write_goal(context, 0.0)

    def _write_goal(self, context: MotionStatechartContext, elapsed: float) -> None:
        """
        Publish the container trajectory as its corresponding gripper tool pose.
        """
        root_T_tool = self._trajectory.pose_at(elapsed) @ self._container_T_tool
        context.float_variable_data.set_value(
            self.root_T_goal, root_T_tool[:3, :4].T.flatten()
        )
        context.float_variable_data.set_value(self.elapsed, elapsed)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> ObservationStateValues | None:
        """
        Advance the target using active control time and retain the final upright pose.
        """
        cycle = context.float_variable_data.get_value(context.control_cycle_variable)
        elapsed = context.float_variable_data.get_value(self.elapsed)
        if self._is_tracking(context, elapsed):
            elapsed += (
                cycle - self._last_cycle
            ) * context.qp_controller_config.control_dt
        self._last_cycle = cycle
        self._write_goal(context, elapsed)
        return None

    def _is_tracking(self, context: MotionStatechartContext, elapsed: float) -> bool:
        """
        Advance only while the gripper follows the current target within tolerance.
        """
        root_T_current = context.world.compute_forward_kinematics_np(
            self.root_link, self.tip_link
        )
        root_T_goal = self._trajectory.pose_at(elapsed) @ self._container_T_tool
        position_error = np.linalg.norm(root_T_goal[:3, 3] - root_T_current[:3, 3])
        orientation_error = Rotation.from_matrix(
            root_T_current[:3, :3].T @ root_T_goal[:3, :3]
        ).magnitude()
        return (
            position_error <= self.maximum_position_error
            and orientation_error <= self.maximum_orientation_error
        )

    def on_pause(self, context: MotionStatechartContext) -> None:
        """
        Remember the cycle at which the motion was paused.
        """
        self._pause_cycle = context.float_variable_data.get_value(
            context.control_cycle_variable
        )

    def on_unpause(self, context: MotionStatechartContext) -> None:
        """
        Exclude paused control cycles from the trajectory clock.
        """
        cycle = context.float_variable_data.get_value(context.control_cycle_variable)
        self._last_cycle += cycle - self._pause_cycle


# %% Motion designator


@dataclass(kw_only=True)
class SwirlContainerMotion(BaseMotion):
    """
    Swirl a held container with a continuous pose goal and a bounded tilt cone.
    """

    requires_individual_execution: ClassVar[bool] = True
    """
    Bind the grasp after preceding motions have finished.
    """

    container: Body
    """
    Rigidly held container; its local positive Z axis points toward its mouth.
    """

    arm: Arms
    """
    Arm that already holds the container.
    """

    profile: SwirlProfile = field(default_factory=SwirlProfile)
    """
    Smooth conical motion to execute.
    """

    pivot: Point3 | None = None
    """
    Pivot in the container frame; defaults to the center of its upper rim.
    """

    tilt_margin: float = 0.03
    """
    Allowed orientation tracking margin beyond the desired cone, in radians.
    """

    @property
    def maximum_tilt(self) -> float:
        """
        Return the Giskard tilt bound relative to the initial container axis.
        """
        return self.profile.tilt_angle + self.tilt_margin

    def validate_grasp(self) -> None:
        """
        Require an existing rigid grasp in the selected arm's kinematic chain.
        """
        tool = ViewManager.get_end_effector_view(self.arm, self.robot).tool_frame
        current = self.container
        if current == tool:
            raise ContainerNotHeld(self.container)
        while current != tool:
            connection = self.world.compute_parent_connection(current)
            if not isinstance(connection, FixedConnection):
                raise ContainerNotHeld(self.container)
            current = connection.parent

    def _pivot(self) -> Point3:
        """
        Resolve the requested pivot or the upper-rim center from container geometry.
        """
        if self.pivot is not None:
            return self.world.transform(self.pivot, self.container)
        bounds = self.container.visual.as_bounding_box_collection_in_frame(
            self.container
        ).bounding_box()
        return Point3(
            x=0.5 * (bounds.min_x + bounds.max_x),
            y=0.5 * (bounds.min_y + bounds.max_y),
            z=bounds.max_z,
            reference_frame=self.container,
        )

    @property
    def _motion_chart(self) -> Parallel:
        """
        Build simultaneous continuous pose tracking, tilt and speed constraints.
        """
        self.validate_grasp()
        tool = ViewManager.get_end_effector_view(self.arm, self.robot).tool_frame
        root = self.world.root
        tool_T_container = self.world.compute_forward_kinematics_np(
            tool, self.container
        )
        root_T_container = self.world.compute_forward_kinematics_np(
            root, self.container
        )
        tracking = SwirlPoseTask(
            name=SwirlTaskName.POSE,
            root_link=root,
            tip_link=tool,
            container=self.container,
            container_P_pivot=self._pivot(),
            profile=self.profile,
        )
        return Parallel(
            name=SwirlTaskName.MOTION,
            nodes=[
                tracking,
                AngleGoal(
                    name=SwirlTaskName.TILT,
                    root_link=root,
                    tip_link=tool,
                    tip_vector=Vector3(*tool_T_container[:3, 2], reference_frame=tool),
                    reference_vector=Vector3(
                        *root_T_container[:3, 2], reference_frame=root
                    ),
                    lower_angle=0.0,
                    upper_angle=self.maximum_tilt,
                    weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
                ),
                CartesianPositionVelocityLimit(
                    root_link=root,
                    tip_link=tool,
                    max_linear_velocity=tracking.reference_linear_velocity,
                ),
                CartesianRotationVelocityLimit(
                    root_link=root,
                    tip_link=tool,
                    max_angular_velocity=tracking.reference_angular_velocity,
                ),
            ],
        )
