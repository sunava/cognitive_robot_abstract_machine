from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional, Any

from pycram.robot_plans.actions.composite.thesis_math.metrics import (
    points_world_to_body,
    cutting_depth_metrics,
    mixing_bowl_metrics,
)
from pycram.robot_plans.actions.composite.thesis_math.motion_models import (
    MotionSegment,
    MotionSequence,
)


from pycram.robot_plans.actions.composite.utils.rviz import publish_points_sequence
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from .thesis_math.motion_presets import (
    build_container_sequence,
    build_surface_sequence,
    build_cutting_sequence,
    build_pouring_sequence,
)
from .thesis_math.motion_profiles import planar_spiral_xy, planar_sweep_x
from .thesis_math.world_utils import body_local_aabb
from ... import (
    MoveTCPWaypointsAlignedMotion,
    MoveTCPWaypointsAlignedMotionw,
    MoveTCPWaypointsMotion,
)

from ....datastructures.enums import (
    Arms,
    MovementType,
)
from ....plans.factories import sequential
from ....robot_plans.actions.base import ActionDescription
from ....robot_plans.motions.gripper import MoveToolCenterPointMotion
from ....tf_transformations import quaternion_from_matrix
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


def _logging_helper_collect_cutting_fields(
    action: Any, points_world: np.ndarray
) -> dict:
    if (
        action.__class__.__name__ != "CuttingAction"
        or getattr(action, "container", None) is None
    ):
        return {}

    metrics = cutting_depth_metrics(
        points_world=points_world,
        world=action.world,
        bread_body=action.container,
        apply_shape_scale=True,
    )
    return {
        "has_entry_from_above": metrics.get("has_entry_from_above"),
    }


def _logging_helper_collect_mixing_fields(
    action: Any, points_world: np.ndarray
) -> dict:
    if (
        action.__class__.__name__ != "MixingAction"
        or getattr(action, "container", None) is None
    ):
        return {}

    metrics = mixing_bowl_metrics(
        points_world=points_world,
        world=action.world,
        bowl_body=action.container,
        apply_shape_scale=True,
    )
    return {"mixing_bowl_metrics": metrics}


def _logging_helper_collect_container_fields(
    action: Any, points_world: np.ndarray
) -> dict:
    fields = {}
    fields.update(
        _logging_helper_collect_target_intersection_fields(action, points_world)
    )
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

    tool: HasRootBody
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
        sampled = self._sample_motion()
        P = self._extract_points_for_logging(sampled)
        self.logged_action_name = self.__class__.__name__
        self.logged_technique = self._logged_technique_value()
        _logging_helper_apply_fields(
            self, _logging_helper_collect_identity_fields(self)
        )

        publish_points_sequence(
            node=self.plan.context.ros_node,
            points=P,
            frame_id="map",
            topic="/point_sequence",
            phase_id=self._extract_phase_ids_for_logging(sampled),
            republish_hz=2.0,
            clear_existing=self.clear_viz,
        )
        self.robot.full_body_controlled = True
        waypoints = self._build_waypoints(sampled)
        self._last_waypoints = waypoints
        self._postprocess_waypoints_for_logging(waypoints)

        _logging_helper_apply_fields(
            self, _logging_helper_collect_container_fields(self, P)
        )

        try:
            self.add_subplan(sequential([self._build_motion(waypoints)])).perform()
        except Exception as exc:
            if self._accept_execution_failure_as_success(waypoints, exc):
                return

            collision_contacts = None
            try:
                collision_contacts = len(
                    self.world.collision_manager.compute_collisions().contacts
                )
            except Exception:
                collision_contacts = None

            msg = str(exc)
            if (
                "No waypoints provided to MoveTCPWaypointsAlignedMotion" in msg
                or "No aligned waypoint tasks generated" in msg
                or "No waypoints left after applying pointer_stride" in msg
                or "No pose waypoints left after applying pointer_stride" in msg
            ):
                raise ValueError(
                    "Aligned motion failed: no waypoint sequence to execute "
                    f"(waypoints={len(waypoints)}, collisions_now={collision_contacts})."
                ) from exc

            raise RuntimeError(
                "Aligned motion failed during execution "
                f"(waypoints={len(waypoints)}, collisions_now={collision_contacts}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _accept_execution_failure_as_success(self, waypoints, exc: Exception) -> bool:
        return False

    def _sample_motion(self):
        return self._sample_points()

    def _extract_points_for_logging(self, sampled) -> np.ndarray:
        _, points, _ = sampled
        return np.asarray(points, dtype=float)

    def _extract_phase_ids_for_logging(self, sampled) -> np.ndarray:
        _, _, ids = sampled
        return ids

    def _logged_technique_value(self):
        return getattr(self, "technique", None)

    def _build_waypoints(self, sampled):
        _, points, _ = sampled
        stride = max(1, int(self.pointer_stride))
        return self._to_waypoints(points, stride)

    def _postprocess_waypoints_for_logging(self, waypoints) -> None:
        if self.__class__.__name__ == "CuttingAction":
            self.db_debug_waypoint_count = float(len(waypoints))

    def _build_motion(self, waypoints):
        alignment_target = self._resolve_alignment_target()

        alignment_pairs = (
            self.tool.tool_alignment(alignment_target)
            if (self.tool is not None and alignment_target is not None)
            else []
        )
        try:
            tip = self.tool.get_tool_frame()
        except Exception:
            tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame

        return MoveTCPWaypointsAlignedMotion(
            waypoints,
            self.arm,
            allow_gripper_collision=True,
            alignment_pairs=alignment_pairs,
            tip=tip,
        )

    def _sample_points(selfs):
        raise NotImplementedError

    def _to_waypoints(self, points: np.ndarray, stride: int) -> list[Point3]:
        waypoints = [
            Point3(x=p[0], y=p[1], z=p[2], reference_frame=self.world.root)
            for p in points
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

    motion_timeout_ticks = 300

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
            mix_duration_s=(
                self.mix_duration_s if float(self.mix_duration_s) > 0.0 else None
            ),
        )
        return seq.sample(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)


@dataclass(kw_only=True)
class WipingAction(GeneralizedActionPlan):
    """
    Execute a planar wiping motion around a target pose.
    """

    motion_timeout_ticks = 300

    final_waypoint_success_tolerance_m: float = 0.08
    """
    Accept a timeout as successful if the tool is already this close to the final wipe waypoint.
    """

    container: Optional[Body] = None
    """
    Optional alias for surface_body (backward compatibility).
    """
    target_pose: Optional[Pose] = None
    """
    Center pose for the wiping patch.
    """
    technique: str = "wipe"
    """
    Surface-contact technique variant.
    """
    length: float = 0.20
    """
    Sweep length for the wiping motion.
    """

    cycles: float = 1.0
    """
    Number of sweep cycles.
    """
    _resolved_surface_body: Optional[Body] = field(default=None, init=False, repr=False)
    _resolved_alignment_target: Optional[Body | Pose] = field(
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
        return self.target_pose

    def _resolved_technique(self) -> str:
        return str(self.technique).lower().strip()

    def _resolve_surface_body(self) -> Body:
        if self._resolved_surface_body is not None:
            return self._resolved_surface_body

        if self.container is None:
            if self.target_pose is None:
                raise ValueError(
                    "WipingAction requires either container or target_pose."
                )
            resolved = getattr(self.target_pose, "frame_id", None)
        else:
            resolved = self.container
        self._resolved_surface_body = resolved
        return resolved

    def _resolve_alignment_target(self):
        if self._resolved_alignment_target is not None:
            return self._resolved_alignment_target
        if self.container is None and self.target_pose is not None:
            self._resolved_alignment_target = self._resolve_pose_alignment_target()
            return self._resolved_alignment_target
        return self._resolve_surface_body()

    def _resolve_pose_alignment_target(self):
        return self.target_pose

    def _sample_points(self):
        if self.container is not None:
            seq = build_surface_sequence(
                self.container,
                use_visual_aabb=True,
                apply_shape_scale=True,
                technique=self._resolved_technique(),
            )
            return seq.sample(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)

        if self.target_pose is None:
            raise ValueError(
                "WipingAction requires either container or target_pose for sampling."
            )

        t_pose = self._target_pose_to_spatial()
        technique = self._resolved_technique()
        if technique == "spread":
            segment = MotionSegment(
                name="spread_patch",
                duration_s=2.0,
                local_curve=lambda tau: planar_sweep_x(
                    tau, length=float(self.length), cycles=max(1.0, float(self.cycles))
                ),
            )
        else:
            segment = MotionSegment(
                name="wipe_patch",
                duration_s=2.0,
                local_curve=lambda tau: planar_spiral_xy(
                    tau, r0=0.00, r1=0.12, cycles=2.5
                ),
            )
        seq = MotionSequence([segment])
        return seq.sample(frame=t_pose, dt=DEFAULT_SAMPLE_DT)

    def _accept_execution_failure_as_success(self, waypoints, exc: Exception) -> bool:
        if not self._is_timeout_failure(exc) or not waypoints:
            return False

        sponge = getattr(getattr(self, "tool", None), "root", None)
        if sponge is None:
            print("[wipe timeout check] no sponge root on WipingAction.tool")
            return False

        try:
            raw_last_xyz = np.asarray(waypoints[-1].to_np(), dtype=float).reshape(-1)[
                :3
            ]
            raw_sponge_xyz = np.asarray(
                sponge.global_pose.to_position().to_np(), dtype=float
            ).reshape(-1)[:3]
            print(
                "[wipe timeout check] raw last_point_xyz="
                f"{np.round(raw_last_xyz, 4).tolist()} "
                f"sponge_global_xyz={np.round(raw_sponge_xyz, 4).tolist()}"
            )
            tip_point = self.world.transform(
                sponge.global_pose.to_position(), self.world.root
            )
            tip_xyz = np.asarray(tip_point.to_np(), dtype=float).reshape(-1)[:3]
            goal_point = self.world.transform(waypoints[-1], self.world.root)
            goal_xyz = np.asarray(goal_point.to_np(), dtype=float).reshape(-1)[:3]
            distance = float(np.linalg.norm(tip_xyz - goal_xyz))
        except Exception as transform_exc:
            print(
                "[wipe timeout check] failed to compare sponge and last point: "
                f"{type(transform_exc).__name__}: {transform_exc}"
            )
            return False

        print(
            "[wipe timeout check] transformed last_point_xyz="
            f"{np.round(goal_xyz, 4).tolist()} "
            f"sponge_xyz={np.round(tip_xyz, 4).tolist()} "
            f"distance={distance:.3f}m "
            f"tolerance={float(self.final_waypoint_success_tolerance_m):.3f}m"
        )
        logger.warning(
            "WipingAction timeout check: last_point_xyz=%s sponge_xyz=%s distance=%.3fm",
            np.round(goal_xyz, 4).tolist(),
            np.round(tip_xyz, 4).tolist(),
            distance,
        )

        if distance > float(self.final_waypoint_success_tolerance_m):
            return False

        logger.warning(
            "Accepting WipingAction timeout as success because sponge reached final waypoint "
            "(distance=%.3fm, tolerance=%.3fm).",
            distance,
            self.final_waypoint_success_tolerance_m,
        )
        return True

    def _accept_motion_timeout_as_success(self, exc: Exception) -> bool:
        waypoints = getattr(self, "_last_waypoints", [])
        return self._accept_execution_failure_as_success(waypoints, exc)

    @staticmethod
    def _is_timeout_failure(exc: Exception) -> bool:
        current = exc
        while current is not None:
            if isinstance(current, TimeoutError):
                return True
            msg = str(current)
            if (
                "Timeout reached while waiting for end of motion" in msg
                or "Motion stalled while waiting for end of motion" in msg
                or "Hard timeout reached while waiting for end of motion" in msg
            ):
                return True
            current = current.__cause__ or current.__context__
        return False


@dataclass
class CuttingAction(GeneralizedActionPlan):
    """
    Execute a cutting motion sequence on a food object.
    """

    motion_timeout_ticks = 100

    container: Body = None
    """
    The object to cut.
    """
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


@dataclass
class PouringAction(GeneralizedActionPlan):
    """
    Execute a simple pose-aware pouring motion around a target container or anchor.
    """

    container: Body = None
    """
    Source container that is being tilted.
    """

    target_container: Optional[Body] = None
    """
    Optional target container/body over which the pour should happen.
    """

    pour_height: float = 0.10
    """
    Vertical offset above the target top surface during pouring.
    """

    pour_offset_xy: tuple[float, float] = (0.0, 0.0)
    """
    Additional local XY offset for the pour anchor.
    """

    approach_distance: float = 0.08
    """
    Approach distance before the pour anchor.
    """

    retreat_distance: float = 0.08
    """
    Retreat distance after the pour anchor.
    """

    max_tilt: float = float(np.deg2rad(65.0))
    """
    Maximum tilt angle during the pouring phase.
    """

    tilt_axis: str = "y"
    """
    Local tilt axis for the pouring rotation profile.
    """

    def _sample_pose_sequence(self):
        seq = build_pouring_sequence(
            self.container,
            target_body=self.target_container,
            use_visual_aabb=True,
            apply_shape_scale=True,
            pour_height=self.pour_height,
            pour_offset_xy=self.pour_offset_xy,
            approach_distance=self.approach_distance,
            retreat_distance=self.retreat_distance,
            max_tilt=self.max_tilt,
            tilt_axis=self.tilt_axis,
        )
        return seq.sample_poses(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)

    def _to_pose_waypoints(
        self, positions: np.ndarray, rotations: np.ndarray, stride: int
    ) -> list[Pose]:
        poses = []
        for p, r in zip(positions[::stride], rotations[::stride]):
            rotation_matrix = np.asarray(r, dtype=float)
            if rotation_matrix.shape == (3, 3):
                homogeneous_rotation = np.eye(4, dtype=float)
                homogeneous_rotation[:3, :3] = rotation_matrix
            elif rotation_matrix.shape == (4, 4):
                homogeneous_rotation = rotation_matrix
            else:
                raise ValueError(
                    "Expected pouring waypoint rotation to have shape (3, 3) or (4, 4), "
                    f"got {rotation_matrix.shape}."
                )

            quat = quaternion_from_matrix(homogeneous_rotation)
            poses.append(
                Pose.from_xyz_quaternion(
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                    reference_frame=self.world.root,
                )
            )
        if not poses:
            raise ValueError("No pose waypoints left after applying pointer_stride.")
        return poses

    def _sample_motion(self):
        return self._sample_pose_sequence()

    def _extract_points_for_logging(self, sampled) -> np.ndarray:
        return np.asarray(sampled.positions, dtype=float)

    def _extract_phase_ids_for_logging(self, sampled) -> np.ndarray:
        return sampled.phase_ids

    def _logged_technique_value(self):
        return "pour"

    def _build_waypoints(self, sampled):
        stride = max(1, int(self.pointer_stride))
        return self._to_pose_waypoints(sampled.positions, sampled.rotations, stride)

    def _build_motion(self, waypoints):
        return MoveTCPWaypointsMotion(
            waypoints,
            self.arm,
            allow_gripper_collision=True,
        )


@dataclass
class SimplePouringAction(ActionDescription):
    """
    Execute a simple Cartesian pour over a target container.
    """

    object_designator: Body
    """
    Target container over which the held object should be poured.
    """

    arm: Arms
    """
    Arm that is holding the pouring object.
    """

    offset_x: float = 0.009
    """
    World-space X offset from the target container center.
    """

    offset_y: float = -0.125
    """
    World-space Y offset from the target container center.
    """

    offset_z: float = 0.17
    """
    World-space Z offset above the target container.
    """

    tilt_degrees: float = -110.0
    """
    Local roll applied after reaching the pre-pour pose.
    """

    def _approach_pose(self) -> Pose:
        target_pose = self.object_designator.global_pose
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        tip_quat = tip.global_pose.to_quaternion().to_np()

        return Pose.from_xyz_quaternion(
            float(target_pose.x) + float(self.offset_x),
            float(target_pose.y) + float(self.offset_y),
            float(target_pose.z) + float(self.offset_z),
            float(tip_quat[0]),
            float(tip_quat[1]),
            float(tip_quat[2]),
            float(tip_quat[3]),
            reference_frame=self.world.root,
        )

    def _tilted_pose(self, approach_pose: Pose) -> Pose:
        tilt = Pose.from_xyz_rpy(
            roll=float(np.deg2rad(self.tilt_degrees)),
            reference_frame=self.world.root,
        ).to_homogeneous_matrix()
        return (approach_pose.to_homogeneous_matrix() @ tilt).to_pose()

    def execute(self) -> None:
        approach_pose = self._approach_pose()
        tilted_pose = self._tilted_pose(approach_pose)

        self.add_subplan(
            sequential(
                [
                    MoveToolCenterPointMotion(
                        approach_pose,
                        self.arm,
                        allow_gripper_collision=True,
                        movement_type=MovementType.CARTESIAN,
                    ),
                    MoveToolCenterPointMotion(
                        tilted_pose,
                        self.arm,
                        allow_gripper_collision=True,
                        movement_type=MovementType.CARTESIAN,
                    ),
                ]
            )
        ).perform()
