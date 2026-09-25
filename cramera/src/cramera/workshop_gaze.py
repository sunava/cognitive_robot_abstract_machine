"""Native PR2 head tracking during workshop tool motions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import ClassVar

from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from coraplex.view_manager import ViewManager
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.error_signals import SampledErrorSignal
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import NodeArtifacts
from giskardpy.motion_statechart.tasks.pointing import Pointing
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import Point3, Vector3


# %% continuously observed manipulation target
class GazeTask(StrEnum):
    """Names identifying the accompanying head task in native recordings."""

    FOLLOW_TOOL = "Follow active gripper"
    """Optical camera tracking the observed manipulation frame."""
    MOVE_WITH_GAZE = "Move tool with gaze"
    """Native tool movement accompanied by head tracking."""


@dataclass(eq=False, repr=False)
class FollowingPointing(Pointing):
    """Refresh the target-frame transform each tick without controlling that frame."""

    def on_tick(self, context: MotionStatechartContext) -> None:
        """Observe the moving gripper before computing the next head command."""
        self._forward_kinematics_binding.bind(context.world)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """Account for the changing observed target in the pointing error rate."""
        artifacts = super().build_artifacts(context)
        artifacts.error = SampledErrorSignal(artifacts.error.expression)
        return artifacts


@dataclass(eq=False, repr=False)
class ManipulationWithGaze(Parallel):
    """Keep tracking active until the accompanying tool movement completes."""

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """Complete on the tool goal even when head limits prevent perfect alignment."""
        return NodeArtifacts(observation=self.nodes[0].observation_variable)


# %% PR2 alternative motion mapping
@dataclass
class WorkshopMoveToolCenterPoint(MoveToolCenterPointMotion, AlternativeMotion[PR2]):
    """Add optical gripper tracking to the unchanged native Cartesian tool task."""

    execution_type: ClassVar[ExecutionType] = ExecutionType.SIMULATED
    """Simulation execution selects this workshop-specific mapping."""
    maximum_head_velocity: ClassVar[float] = 0.5
    """Conservative angular speed of the pointing constraint, in radians per second."""
    pointing_threshold: ClassVar[float] = 0.025
    """Head alignment observation tolerance, in radians."""

    @property
    def _motion_chart(self) -> ManipulationWithGaze:
        """Combine tool motion with a head-only task following its current position."""
        tool = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        camera = self.robot.get_default_camera().root
        gaze = FollowingPointing(
            name=GazeTask.FOLLOW_TOOL,
            root_link=self.robot.get_torso().root,
            tip_link=camera,
            goal_point=Point3(reference_frame=tool),
            pointing_axis=Vector3.Z(reference_frame=camera),
            max_velocity=self.maximum_head_velocity,
            threshold=self.pointing_threshold,
            weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        )
        return ManipulationWithGaze(
            [super()._motion_chart, gaze], name=GazeTask.MOVE_WITH_GAZE
        )
