from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from typing_extensions import Union, Optional, Any, Iterable

from demos.thesis_new.thesis_math.metrics import (
    points_world_to_body,
    cutting_depth_metrics,
    mixing_bowl_metrics,
)
from demos.thesis_new.thesis_math.motion_models import MotionSegment, MotionSequence
from demos.thesis_new.thesis_math.motion_presets import (
    build_container_sequence,
    build_cutting_sequence,
    build_surface_sequence,
)
from demos.thesis_new.thesis_math.motion_profiles import planar_spiral_xy
from demos.thesis_new.thesis_math.world_utils import (
    body_local_aabb,
)
from demos.thesis_new.utils.rviz import publish_points_sequence
from semantic_digital_twin.semantic_annotations.semantic_annotations import Tool
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world_description.world_entity import Body
from ...motions.gripper import MoveTCPWaypointsAlignedMotion
from ....datastructures.enums import (
    Arms,
)
from ....datastructures.partial_designator import PartialDesignator
from ....datastructures.pose import PoseStamped
from ....language import SequentialPlan
from ....robot_plans.actions.base import ActionDescription
from ....view_manager import ViewManager

logger = logging.getLogger(__name__)
DEFAULT_SAMPLE_DT = 0.01


def _logging_helper_apply_fields(action: Any, fields: dict) -> None:
    for key, value in fields.items():
        setattr(action, key, value)


def _logging_helper_collect_identity_fields(action: Any) -> dict:
    tool_root_name = None
    if action.tool is not None and getattr(action.tool, "root", None) is not None:
        tool_root_name = str(getattr(action.tool.root, "name", action.tool.root))

    container_obj = getattr(action, "container", None)
    container_name = (
        "no container given"
        if container_obj is None
        else str(getattr(container_obj, "name", container_obj))
    )

    return {
        "logged_tool_root_name": tool_root_name,
        "logged_container_name": container_name,
    }


def _logging_helper_collect_target_intersection_fields(
        action: Any, points_world: np.ndarray
) -> dict:
    container = getattr(action, "container", None)
    if container is None:
        return {}

    points_body = points_world_to_body(points_world, action.world, container)
    mins, maxs = body_local_aabb(container, use_visual=False, apply_shape_scale=True)
    inside = (
            (points_body[:, 0] >= mins[0])
            & (points_body[:, 0] <= maxs[0])
            & (points_body[:, 1] >= mins[1])
            & (points_body[:, 1] <= maxs[1])
            & (points_body[:, 2] >= mins[2])
            & (points_body[:, 2] <= maxs[2])
    )
    inside_ratio = float(np.mean(inside)) if len(inside) > 0 else 0.0
    return {"logged_target_intersection_success": bool(inside_ratio >= 0.5)}


def _logging_helper_collect_cutting_fields(action: Any, points_world: np.ndarray) -> dict:
    if action.__class__.__name__ != "CuttingAction" or getattr(action, "container", None) is None:
        return {}

    metrics = cutting_depth_metrics(
        points_world=points_world,
        world=action.world,
        bread_body=action.container,
        apply_shape_scale=True,
    )
    return {
        "has_entry_from_above": metrics.get("has_entry_from_above"),
        "reaches_cut_depth": metrics.get("reaches_cut_depth"),
        "cutting_success": metrics.get("cutting_success"),
    }


def _logging_helper_collect_mixing_fields(action: Any, points_world: np.ndarray) -> dict:
    if action.__class__.__name__ != "MixingAction" or getattr(action, "container", None) is None:
        return {}

    metrics = mixing_bowl_metrics(
        points_world=points_world,
        world=action.world,
        bowl_body=action.container,
        apply_shape_scale=True,
    )
    return {"mixing_bowl_metrics": metrics}


def _logging_helper_collect_container_fields(action: Any, points_world: np.ndarray) -> dict:
    fields = {}
    fields.update(_logging_helper_collect_target_intersection_fields(action, points_world))
    fields.update(_logging_helper_collect_cutting_fields(action, points_world))
    fields.update(_logging_helper_collect_mixing_fields(action, points_world))
    return fields


@dataclass
class GeneralizedActionPlan(ActionDescription):
    """
    Base class for tool-based motion sequences over a container.
    """

    arm: Arms
    """
    Arm used for the motion.
    """

    tool: Tool
    """
    Tool body used to estimate the tip offset.
    """

    clear_viz: bool = False
    """
    If viz should be cleared
    """

    pointer_stride: int = 1
    """
    Keep every Nth waypoint for execution (testing downsampling).
    """

    logged_target_intersection_success: Optional[bool] = None
    """
    Optional DB-logged boolean derived from target_intersection_success.
    """

    logged_tool_root_name: Optional[str] = None
    """
    Optional DB-logged tool root name captured during execute().
    """

    logged_container_name: Optional[str] = None
    """
    Optional DB-logged container name captured during execute().
    Uses "no container given" if no container is present.
    """

    logged_action_name: Optional[str] = None
    """
    Optional DB-logged concrete action class name captured during execute().
    """

    logged_technique: Optional[str] = None
    """
    Optional DB-logged technique value captured during execute() when available.
    """

    def execute(self) -> None:
        _, points, ids = self._sample_points()
        # points_world = self._points_with_tip(points)
        P = np.asarray(points, dtype=float)
        self.logged_action_name = self.__class__.__name__
        self.logged_technique = getattr(self, "technique", None)
        _logging_helper_apply_fields(self, _logging_helper_collect_identity_fields(self))

        publish_points_sequence(
            node=self.context.ros_node,
            points=P,
            frame_id="apartment/apartment_root",
            topic="/point_sequence",
            phase_id=ids,
            republish_hz=2.0,
            clear_existing=self.clear_viz,
        )

        self.robot_view.full_body_controlled = True
        stride = max(1, int(self.pointer_stride))
        pointery = self._to_waypoints(points, stride)
        if self.__class__.__name__ == "CuttingAction":
            self.db_debug_waypoint_count = float(len(pointery))

        _logging_helper_apply_fields(self, _logging_helper_collect_container_fields(self, P))

        alignment_target = self._resolve_alignment_target()

        alignment_pairs = (
            self.tool.tool_alignment(alignment_target)
            if (self.tool is not None and alignment_target is not None)
            else []
        )
        try:
            tip = self.tool.get_tool_frame()
        except Exception:
            tip = ViewManager().get_end_effector_view(self.arm, self.robot_view).tool_frame
        try:
            SequentialPlan(
                self.context,
                MoveTCPWaypointsAlignedMotion(
                    pointery,
                    self.arm,
                    allow_gripper_collision=True,
                    alignment_pairs=alignment_pairs,
                    tip=tip
                ),
            ).perform()
        except Exception:
            raise

    def _sample_points(selfs):
        raise NotImplementedError

    def _to_waypoints(self, points: np.ndarray, stride: int) -> list[Point3]:
        waypoints = [
            Point3(x=p[0], y=p[1], z=p[2], reference_frame=self.world.root) for p in points
        ][::stride]
        if not waypoints:
            raise ValueError("No waypoints left after applying pointer_stride.")
        return waypoints
    def _resolve_alignment_target(self):
        if hasattr(self, "surface_body") and self.surface_body is not None:
            return self.surface_body
        if hasattr(self, "container") and self.container is not None:
            return self.container
        if hasattr(self, "target_pose") and self.target_pose is not None:
            return self.target_pose
        return None


@dataclass
class MixingAction(GeneralizedActionPlan):
    """
    Execute a mixing motion sequence around a container.
    """

    container: Body = None
    """
    The container (e.g., bowl) to operate in.
    """

    mix_duration_s: float = 0.0
    """
    Total mixing time in seconds for a continuous connected stir loop.
    If <= 0, the default short pattern is used.
    """

    mixing_bowl_metrics: Optional[dict] = None
    """
    Optional DB-logged metrics dict populated during execute().
    """

    def _sample_points(self):
        pattern = "stir" if float(self.mix_duration_s) > 0.0 else "spiral"
        seq = build_container_sequence(
            self.container,
            use_visual_aabb=True,
            apply_shape_scale=True,
            pattern=pattern,
            mix_duration_s=self.mix_duration_s if float(self.mix_duration_s) > 0.0 else None,
        )
        return seq.sample(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)

    def validate(
            self,
            result: Optional[Any] = None,
            max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass

    @classmethod
    def description(
            cls,
            container: Union[Iterable[Body], Body],
            arm: Union[Iterable[Arms], Arms],
            tool: Union[Iterable[Tool], Tool],
            mix_duration_s: Union[Iterable[float], float] = 0.0,
            clear_viz: Union[Iterable[bool], bool] = False,
            pointer_stride: Union[Iterable[int], int] = 1,
    ) -> PartialDesignator[MixingAction]:
        return PartialDesignator(
            cls,
            container=container,
            arm=arm,
            tool=tool,
            mix_duration_s=mix_duration_s,
            clear_viz=clear_viz,
            pointer_stride=pointer_stride,
        )


@dataclass(kw_only=True)
class WipingAction(GeneralizedActionPlan):
    """
    Execute a planar wiping motion around a target pose.
    """
    container: Optional[Body] = None
    """
    Optional alias for surface_body (backward compatibility).
    """
    target_pose: Optional[PoseStamped] = None
    """
    Center pose for the wiping patch.
    """
    length: float = 0.20
    """
    Sweep length for the wiping motion.
    """

    cycles: float = 2.0
    """
    Number of sweep cycles.
    """
    _resolved_surface_body: Optional[Body] = field(
        default=None, init=False, repr=False
    )

    def _target_pose_to_spatial(self):
        if self.target_pose is None:
            return None
        if getattr(self.target_pose, "frame_id", None) is None:
            logger.warning(
                "WipingAction target_pose has no frame_id; defaulting to world root."
            )
            self.target_pose.frame_id = self.world.root
        return self.target_pose.to_spatial_type()

    def _resolve_surface_body(self) -> Body:
        if self._resolved_surface_body is not None:
            return self._resolved_surface_body

        if self.container is None:
            # todo make logic for if i get a pose i need to calculate the normal OR the body next to it.. rotation is just off for the gripper currently
            # todo idea: maybe i just rotate how sponge is rotated and then look at the one function from cram back then
            resolved = ...
        else:
            resolved = self.container
        self._resolved_surface_body = resolved
        return resolved

    def _resolve_alignment_target(self):
        return self._resolve_surface_body()

    def _sample_points(self):
        if self.container is not None:
            seq = build_surface_sequence(
                self.container,
                use_visual_aabb=True,
                apply_shape_scale=True,
                pattern="raster",
            )
            return seq.sample(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)

        if self.target_pose is None:
            raise ValueError(
                "WipingAction requires either container or target_pose for sampling."
            )

        t_pose = self._target_pose_to_spatial()
        segment = MotionSegment(
            name="planar_spiral",
            duration_s=2.0,
            local_curve=lambda tau: planar_spiral_xy(
                tau, r0=0.00, r1=0.12, cycles=2.5
            ),
        )
        seq = MotionSequence([segment])
        return seq.sample(frame=t_pose, dt=DEFAULT_SAMPLE_DT)

    @classmethod
    def description(
            cls,
            arm: Union[Iterable[Arms], Arms],
            tool: Union[Iterable[Tool], Tool],
            length: Union[Iterable[float], float] = 0.20,
            cycles: Union[Iterable[float], float] = 2.0,
            container: Union[Iterable[Body], Body] = None,
            target_pose: Union[Iterable[PoseStamped], PoseStamped] = None,
            clear_viz: Union[Iterable[bool], bool] = False,
            pointer_stride: Union[Iterable[int], int] = 1,
    ) -> PartialDesignator[WipingAction]:
        return PartialDesignator(
            cls,

            arm=arm,
            container=container,
            target_pose=target_pose,
            tool=tool,
            length=length,
            cycles=cycles,
            clear_viz=clear_viz,
            pointer_stride=pointer_stride,
        )


@dataclass
class CuttingAction(GeneralizedActionPlan):
    """
    Execute a cutting motion sequence on a food object.
    """

    container: Body = None
    """
    The object to cut.
    """
    # todo : all my annoations should be within semantic annotation and then add technique there
    technique: str = "saw"
    """
    Cutting trajectory variant.
    """

    slice_thickness: float = 0.03
    """
    Target slice thickness used to place the cut anchor.
    """

    num_cuts_x: int = 1
    """
    Number of repeated cut passes distributed across local X.
    """

    db_debug_waypoint_count: Optional[float] = None
    """
    Optional DB-logged test metric set during execute().
    """

    has_entry_from_above: Optional[bool] = None
    """
    Optional DB-logged cutting flag populated during execute().
    """

    reaches_cut_depth: Optional[bool] = None
    """
    Optional DB-logged cutting flag populated during execute().
    """

    cutting_success: Optional[bool] = None
    """
    Optional DB-logged cutting flag populated during execute().
    """

    def _sample_points(self):
        seq = build_cutting_sequence(
            self.container,
            use_visual_aabb=True,
            apply_shape_scale=True,
            technique=self.technique,
            slice_thickness=self.slice_thickness,
            num_cuts_x=self.num_cuts_x,
        )
        return seq.sample(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)

    def validate(
            self,
            result: Optional[Any] = None,
            max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass

    @classmethod
    def description(
            cls,
            container: Union[Iterable[Body], Body],
            arm: Union[Iterable[Arms], Arms],
            tool: Union[Iterable[Tool], Tool],
            technique: Union[Iterable[str], str] = "saw",
            slice_thickness: Union[Iterable[float], float] = 0.03,
            num_cuts_x: Union[Iterable[int], int] = 1,
            db_debug_waypoint_count: Union[Iterable[Optional[float]], Optional[float]] = None,
            has_entry_from_above: Union[Iterable[Optional[bool]], Optional[bool]] = None,
            reaches_cut_depth: Union[Iterable[Optional[bool]], Optional[bool]] = None,
            cutting_success: Union[Iterable[Optional[bool]], Optional[bool]] = None,
            clear_viz: Union[Iterable[bool], bool] = False,
            pointer_stride: Union[Iterable[int], int] = 1,

    ) -> PartialDesignator[CuttingAction]:
        return PartialDesignator(
            cls,
            container=container,
            arm=arm,
            tool=tool,
            technique=technique,
            slice_thickness=slice_thickness,
            num_cuts_x=num_cuts_x,
            db_debug_waypoint_count=db_debug_waypoint_count,
            has_entry_from_above=has_entry_from_above,
            reaches_cut_depth=reaches_cut_depth,
            cutting_success=cutting_success,
            clear_viz=clear_viz,
            pointer_stride=pointer_stride,
        )


MixingActionDescription = MixingAction.description
WipingActionDescription = WipingAction.description
CuttingActionDescription = CuttingAction.description
