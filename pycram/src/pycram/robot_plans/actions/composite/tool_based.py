from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from trimesh.proximity import closest_point
from typing_extensions import Union, Optional, Type, Any, Iterable

import numpy as np

from demos.thesis_new.thesis_math.motion_models import MotionSegment, MotionSequence
from demos.thesis_new.thesis_math.motion_profiles import planar_sweep_x, planar_spiral_xy
from demos.thesis_new.thesis_math.motion_presets import (
    build_container_sequence,
    build_cutting_sequence,
    build_surface_sequence,
)
from demos.thesis_new.thesis_math.metrics import (
    points_world_to_body,
    distance_to_mesh_metrics,
    cutting_depth_metrics,
    mixing_bowl_metrics,
)
from demos.thesis_new.thesis_math.world_utils import (
    body_local_aabb,
    make_identity_pose_stamped,
    Rp_from_spatial,
)
from demos.thesis_new.utils.rviz import MotionSequenceRviz, publish_points_sequence
from pycram.datastructures.partial_designator import PartialDesignator
from semantic_digital_twin.adapters.ros.pose_publisher import PosePublisher
from semantic_digital_twin.semantic_annotations.semantic_annotations import Tool
from semantic_digital_twin.spatial_types import Vector3, Point3
from semantic_digital_twin.world_description.world_entity import Body
from .experiment_record import ExperimentRecord
from ...motions.gripper import MoveTCPMotion, MoveTCPWaypointsMotion, AlignmentPair, MoveTCPWaypointsAlignedMotion
from .... import utils

from ....datastructures.enums import (
    Arms,
    VerticalAlignment,
    ApproachDirection,
    MovementType,
)
from ....datastructures.grasp import GraspDescription
from ....datastructures.partial_designator import PartialDesignator
from ....datastructures.pose import PoseStamped
from ....language import SequentialPlan
from ....robot_plans.actions.base import ActionDescription
from ....view_manager import ViewManager

logger = logging.getLogger(__name__)


def tip_offset_from_body(body):
    """Estimate the tool tip along +X using the local AABB."""
    mins, maxs = body_local_aabb(body, use_visual=True, apply_shape_scale=True)
    center_y = 0.5 * (mins[1] + maxs[1])
    center_z = 0.5 * (mins[2] + maxs[2])
    return np.array([maxs[0], center_y, center_z], dtype=float)


@dataclass
class GeneralizedActionPlan(ActionDescription):
    """
    Base class for tool-based motion sequences over a container.
    """

    arm: Arms
    """
    Arm used for the motion.
    """

    tool_name: Optional[str] = None
    """
    Tool configuration name (e.g., 'whisk').
    """

    tool: Optional[Tool] = None
    """
    Tool body used to estimate the tip offset.
    """

    tool_tip_offset: Optional[Iterable[float]] = None
    """
    Explicit tool tip offset in the tool frame.
    """

    dt: float = 0.01
    """
    Sampling time step for the motion sequence.
    """

    use_visual_aabb: bool = True
    """
    Use the visual AABB for sizing the motion sequence.
    """

    apply_shape_scale: bool = True
    """
    Apply shape scales when computing the AABB.
    """

    clear_viz: bool = False
    """
    If viz should be cleared
    """

    pointer_stride: int = 1
    """
    Keep every Nth waypoint for execution (testing downsampling).
    """

    def _sample_points(selfs):
        raise NotImplementedError

    def _tool_tip_in_world(self):
        """Return the tool tip pose in the world frame if available."""
        if self.tool is None or not hasattr(self.tool, "tip"):
            return None
        tip_pose = self.tool.tip()
        if tip_pose is None:
            return None
        return self.world.transform(tip_pose, self.world.root)

    def _pose_orientation(self) -> list[float]:
        """Return default waypoint orientation in world frame."""
        if hasattr(self, "container") and self.container is not None:
            return self.container.global_pose.to_quaternion().to_list()
        if hasattr(self, "target_pose") and self.target_pose is not None:
            return self.target_pose.orientation.to_list()
        return [0.0, 0.0, 0.0, 1.0]

    # def _points_with_tip(self, points: np.ndarray) -> np.ndarray:
    #     """
    #     Apply tool tip offset in tool frame:
    #     world.root point -> tool.root -> add tip offset -> world.root.
    #     """
    #     P = np.asarray(points, dtype=float).reshape(-1, 3)
    #
    #     if self.tool is None or not hasattr(self.tool, "tip"):
    #         return P
    #     tip_pose = self.tool.tip()
    #     if tip_pose is None:
    #         return P
    #
    #     tip_offset_tool = np.asarray(tip_pose.to_position().to_np()[:3], dtype=float)
    #     points_world = []
    #     for p in P:
    #         p_world = Point3(
    #             x=float(p[0]),
    #             y=float(p[1]),
    #             z=float(p[2]),
    #             reference_frame=self.world.root,
    #         )
    #         p_tool = self.world.transform(p_world, self.tool.root)
    #         p_tool_offset = Point3(
    #             x=float(p_tool.x) + float(tip_offset_tool[0]),
    #             y=float(p_tool.y) + float(tip_offset_tool[1]),
    #             z=float(p_tool.z) + float(tip_offset_tool[2]),
    #             reference_frame=self.tool.root,
    #         )
    #         p_world_offset = self.world.transform(p_tool_offset, self.world.root)
    #         points_world.append(
    #             [
    #                 float(p_world_offset.x),
    #                 float(p_world_offset.y),
    #                 float(p_world_offset.z),
    #             ]
    #         )
    #     return np.asarray(points_world, dtype=float)



    def closest_face_with_normal(self, mesh, query_point):
        if isinstance(query_point, Point3):
            q_local = self.world.transform(query_point, self.container)
            q= np.array([float(q_local.x), float(q_local.y), float(q_local.z)], dtype=float)
        else:
            q = np.asarray(query_point, dtype=float).reshape(3)


        cp, dist, tri_id = closest_point(mesh, q.reshape(1, 3))
        face_id = int(tri_id[0])

        normal = mesh.face_normals[face_id]  # Normal in mesh/body frame
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        return {
            "face_id": face_id,
            "closest_point": cp[0],
            "distance": float(dist[0]),
            "normal_body": normal,
        }

    def _log_experiment_record(self, record: dict) -> None:
        """Store experiment records on the ROS node for later evaluation."""
        node = getattr(self.context, "ros_node", None)
        if node is None:
            return
        if not hasattr(node, "_experiment_metrics"):
            node._experiment_metrics = []
        node._experiment_metrics.append(record)

    def _target_intersection_metrics(self, points_world: np.ndarray) -> dict:
        if not hasattr(self, "container") or self.container is None:
            return {}
        points_body = points_world_to_body(points_world, self.world, self.container)
        mins, maxs = body_local_aabb(
            self.container, use_visual=False, apply_shape_scale=self.apply_shape_scale
        )
        inside = (
            (points_body[:, 0] >= mins[0])
            & (points_body[:, 0] <= maxs[0])
            & (points_body[:, 1] >= mins[1])
            & (points_body[:, 1] <= maxs[1])
            & (points_body[:, 2] >= mins[2])
            & (points_body[:, 2] <= maxs[2])
        )
        inside_ratio = float(np.mean(inside)) if len(inside) > 0 else 0.0
        return {
            "intersects_target_volume": bool(np.any(inside)),
            "inside_target_volume_ratio": inside_ratio,
            "target_intersection_success": inside_ratio >= 0.5,
        }

    def execute(self) -> None:
        _, points, ids = self._sample_points()
        # points_world = self._points_with_tip(points)
        P = np.asarray(points, dtype=float)
       
        publish_points_sequence(
            node=self.context.ros_node,
            points=P,
            frame_id="apartment/apartment_root",
            topic="/point_sequence",
            phase_id=ids,  # same length as points
            republish_hz=2.0,
            clear_existing=self.clear_viz,
        )


        self.robot_view.full_body_controlled = True
        pointery = []
        for p in points:
            pointery.append(Point3(x=p[0], y=p[1], z=p[2], reference_frame=self.world.root))
        stride = max(1, int(self.pointer_stride))
        pointery = pointery[::stride]
        if len(pointery) == 0:
            raise ValueError("No waypoints left after applying pointer_stride.")

        record = ExperimentRecord.from_action(
            action=self,
            num_points_sampled=len(points),
            num_points_executed=len(pointery),
            pointer_stride=stride,
        )

        if hasattr(self, "container") and self.container is not None:
            closest = self.closest_face_with_normal(
                self.container.collision.shapes[0].mesh, pointery[0]
            )
            record.set("closest_face", closest)
            dist_threshold = (
                float(self.tool.debug_distance_threshold())
                if self.tool is not None and hasattr(self.tool, "debug_distance_threshold")
                else 0.005
            )
            record.update(self._target_intersection_metrics(P))
            distance_metrics = distance_to_mesh_metrics(
                points_world=P,
                world=self.world,
                body=self.container,
                threshold_m=dist_threshold,
            )
            record.update(
                {
                    "tool_distance_threshold_m_for_success": dist_threshold,
                    "distance_within_threshold_ratio": distance_metrics.get(
                        "below_threshold_ratio"
                    ),
                    "distance_within_threshold_percent": (
                        100.0
                        * float(distance_metrics.get("below_threshold_ratio", 0.0))
                    ),
                }
            )
            record.update(distance_metrics)
            if "mean_distance" in distance_metrics:
                record.set(
                    "distance_mean_minus_threshold_m",
                    float(distance_metrics["mean_distance"]) - dist_threshold,
                )
            if "min_distance" in distance_metrics:
                record.set(
                    "distance_min_minus_threshold_m",
                    float(distance_metrics["min_distance"]) - dist_threshold,
                )
            if self.__class__.__name__ == "CuttingAction":
                record.update(
                    cutting_depth_metrics(
                        points_world=P,
                        world=self.world,
                        bread_body=self.container,
                        apply_shape_scale=self.apply_shape_scale,
                    )
                )
            if self.__class__.__name__ == "MixingAction":
                record.update(
                    mixing_bowl_metrics(
                        points_world=P,
                        world=self.world,
                        bowl_body=self.container,
                        apply_shape_scale=self.apply_shape_scale,
                    )
                )

        alignment_target = None
        if hasattr(self, "container") and self.container is not None:
            alignment_target = self.container
        elif hasattr(self, "target_pose") and self.target_pose is not None:
            alignment_target = self.target_pose

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
            record.mark_action_success(True)
        except Exception as exc:
            record.mark_action_success(False).mark_exception(exc)
            self._log_experiment_record(record.to_dict())
            raise

        record.finalize_geometric()
        self._log_experiment_record(record.to_dict())
        # poses = self._poses_from_points(points)

        # node = self.context.ros_node
        # if not hasattr(node, "_temporary_pose_publishers"):
        #     node._temporary_pose_publishers = []
        # for i, pose in enumerate(poses):
        #     pub = PosePublisher(
        #         pose=pose.to_spatial_type(),
        #         node=node,
        #         lifetime=0,
        #         text=str(i),
        #         topic_name="/pose_sequence",
        #         world=self.world,
        #     )
        #     node._temporary_pose_publishers.append(pub)
        #

        # self.robot_view.full_body_controlled = True
        # SequentialPlan(
        #     self.context,
        #     MoveTCPWaypointsMotion(
        #         poses,
        #         self.arm,
        #         allow_gripper_collision=True,
        #     ),
        # ).perform()
        # print("Pose was published from Designator")
        # poses = self._poses_from_points(points)
        # P = self._points_with_tool_tip_offset(points)

    @classmethod
    def _normalize_tip_offset(cls, tool_tip_offset):
        normalized_tip_offset = tool_tip_offset
        if tool_tip_offset is not None and isinstance(
            tool_tip_offset, (list, tuple, np.ndarray)
        ):
            try:
                if len(tool_tip_offset) == 3 and all(
                    isinstance(v, (int, float, np.floating)) for v in tool_tip_offset
                ):
                    normalized_tip_offset = [list(tool_tip_offset)]
            except TypeError:
                pass
        return normalized_tip_offset


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

    def _sample_points(self):
        pattern = "stir" if float(self.mix_duration_s) > 0.0 else "spiral"
        seq = build_container_sequence(
            self.container,
            use_visual_aabb=self.use_visual_aabb,
            apply_shape_scale=self.apply_shape_scale,
            pattern=pattern,
            mix_duration_s=self.mix_duration_s if float(self.mix_duration_s) > 0.0 else None,
        )
        return seq.sample(frame=self.container.global_pose, dt=self.dt)

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
        tool_name: Union[Iterable[Optional[str]], Optional[str]] = None,
        tool: Union[Iterable[Tool], Tool] = None,
        tool_tip_offset: Union[Iterable[Iterable[float]], Iterable[float]] = None,
        dt: Union[Iterable[float], float] = 0.01,
        use_visual_aabb: Union[Iterable[bool], bool] = True,
        apply_shape_scale: Union[Iterable[bool], bool] = True,
        mix_duration_s: Union[Iterable[float], float] = 0.0,
        clear_viz: Union[Iterable[bool], bool] = False,
        pointer_stride: Union[Iterable[int], int] = 1,
    ) -> PartialDesignator[MixingAction]:
        normalized_tip_offset = cls._normalize_tip_offset(tool_tip_offset)
        return PartialDesignator(
            cls,
            container=container,
            arm=arm,
            tool_name=tool_name,
            tool=tool,
            tool_tip_offset=normalized_tip_offset,
            dt=dt,
            use_visual_aabb=use_visual_aabb,
            apply_shape_scale=apply_shape_scale,
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
    The container (e.g., bowl) to operate in.
    """
    target_pose: Optional[PoseStamped]
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

    def _sample_points(self):

        if self.container is not None:
            print("use container in wiping")
            seq = build_surface_sequence(
                self.container,
                use_visual_aabb=self.use_visual_aabb,
                apply_shape_scale=self.apply_shape_scale,
                pattern="raster",
            )
            return seq.sample(frame=self.container.global_pose, dt=self.dt)
        else :
            tPose = self.target_pose.to_spatial_type()
            segment = MotionSegment(
                name="planar_spiral",
                duration_s=2.0,
                local_curve=lambda tau: planar_spiral_xy(tau, r0=0.00, r1=0.12, cycles=2.5),
            )

            seq = MotionSequence([segment])
            return seq.sample(frame=tPose, dt=self.dt)

    @classmethod
    def description(
        cls,
        arm: Union[Iterable[Arms], Arms],
        tool_name: Union[Iterable[Optional[str]], Optional[str]] = None,
        tool: Union[Iterable[Tool], Tool] = None,
        tool_tip_offset: Union[Iterable[Iterable[float]], Iterable[float]] = None,
        dt: Union[Iterable[float], float] = 0.01,
        length: Union[Iterable[float], float] = 0.20,
        cycles: Union[Iterable[float], float] = 2.0,
        container: Union[Iterable[Body], Body] = None,
        target_pose: Union[Iterable[PoseStamped], PoseStamped] = None,
        clear_viz: Union[Iterable[bool], bool] = False,
        pointer_stride: Union[Iterable[int], int] = 1,
    ) -> PartialDesignator[WipingAction]:
        normalized_tip_offset = cls._normalize_tip_offset(tool_tip_offset)
        return PartialDesignator(
            cls,

            arm=arm,
            container=container,
            target_pose=target_pose,
            tool_name=tool_name,
            tool=tool,
            tool_tip_offset=normalized_tip_offset,
            dt=dt,
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


    # def _pose_orientation(self) -> list[float]:
    #     """
    #     Rotate knife heading by +90 deg around local Z relative to the
    #     container/food orientation so it is perpendicular to the long side.
    #     """
    #     base_orientation = np.asarray(super()._pose_orientation(), dtype=float)
    #     yaw_90 = quaternion_from_euler(0.0, 0.0, np.pi / 2.0, axes="sxyz")
    #     return list(quaternion_multiply(yaw_90, base_orientation))

    def _sample_points(self):
        seq = build_cutting_sequence(
            self.container,
            use_visual_aabb=self.use_visual_aabb,
            apply_shape_scale=self.apply_shape_scale,
            technique=self.technique,
            slice_thickness=self.slice_thickness,
            num_cuts_x=self.num_cuts_x,
        )
        return seq.sample(frame=self.container.global_pose, dt=self.dt)

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
        tool_name: Union[Iterable[Optional[str]], Optional[str]] = None,
        tool: Union[Iterable[Tool], Tool] = None,
        tool_tip_offset: Union[Iterable[Iterable[float]], Iterable[float]] = None,
        dt: Union[Iterable[float], float] = 0.01,
        use_visual_aabb: Union[Iterable[bool], bool] = True,
        apply_shape_scale: Union[Iterable[bool], bool] = True,
        technique: Union[Iterable[str], str] = "saw",
        slice_thickness: Union[Iterable[float], float] = 0.03,
        num_cuts_x: Union[Iterable[int], int] = 1,
        clear_viz : Union[Iterable[bool], bool] = False,
        pointer_stride: Union[Iterable[int], int] = 1,

    ) -> PartialDesignator[CuttingAction]:
        normalized_tip_offset = cls._normalize_tip_offset(tool_tip_offset)
        return PartialDesignator(
            cls,
            container=container,
            arm=arm,
            tool_name=tool_name,
            tool=tool,
            tool_tip_offset=normalized_tip_offset,
            dt=dt,
            use_visual_aabb=use_visual_aabb,
            apply_shape_scale=apply_shape_scale,
            technique=technique,
            slice_thickness=slice_thickness,
            num_cuts_x=num_cuts_x,
            clear_viz=clear_viz,
            pointer_stride=pointer_stride,
        )


MixingActionDescription = MixingAction.description
WipingActionDescription = WipingAction.description
CuttingActionDescription = CuttingAction.description
