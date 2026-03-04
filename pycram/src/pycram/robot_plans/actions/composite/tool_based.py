from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Union, Optional, Type, Any, Iterable

import numpy as np

from demos.thesis_new.thesis_math.motion_models import MotionSegment, MotionSequence
from demos.thesis_new.thesis_math.motion_profiles import planar_sweep_x, planar_spiral_xy
from demos.thesis_new.thesis_math.motion_presets import (
    build_container_sequence,
    build_cutting_sequence,
    build_surface_sequence,
)
from demos.thesis_new.thesis_math.world_utils import (
    body_local_aabb,
    make_identity_pose_stamped,
    Rp_from_spatial,
)
from demos.thesis_new.utils.rviz import MotionSequenceRviz, publish_points_sequence
from pycram.datastructures.partial_designator import PartialDesignator
from semantic_digital_twin.world_description.world_entity import Body
from ...motions.gripper import MoveTCPMotion, MoveTCPWaypointsMotion
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

    tool_body: Optional[Body] = None
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


    def _sample_points(selfs):
        raise NotImplementedError


    def execute(self) -> None:
        _, points, ids = self._sample_points()
        temporary_poses = []
        P= np.asarray(points)
        for p in P:
            pose_t = PoseStamped.from_list(
                position=[float(p[0]), float(p[1]), float(p[2])],
                orientation=[0.0, 0.0, 0.0, 1.0],  # Identity
                frame=self.world.root,  # wichtig: points sind hier Weltkoordinaten
            )
            temporary_poses.append(pose_t)


        P = np.array([[p[0], p[1], p[2]] for p in points], dtype=float)
        # publish_points_sequence(
        #     node=self.context.ros_node,
        #     points=P,
        #     frame_id="apartment/apartment_root",
        #     topic="temporary_pose_seq",
        #     line_width=0.01,
        #     color=(0.2, 0.8, 1.0),
        #     alpha=0.9,
        # )

        # color by phase segments (same id => same color)
        # Example: 6 colors repeated across the sequence
        print(P)
        publish_points_sequence(
            node=self.context.ros_node,
            points=P,
            frame_id="apartment/apartment_root",
            topic="temporary_pose_seq",
            phase_id=ids,  # same length as points
            republish_hz=2.0,
        )
        print("rviz node:", self.context.ros_node)
        print("published pose?")
        #
        # SequentialPlan(
        #     self.context,
        #     MoveTCPWaypointsMotion(
        #         temporary_poses,
        #         self.arm,
        #         allow_gripper_collision=True,
        #     ),
        # ).perform()

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

    def _sample_points(self):
        seq = build_container_sequence(
            self.container,
            use_visual_aabb=self.use_visual_aabb,
            apply_shape_scale=self.apply_shape_scale,
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
        tool_body: Union[Iterable[Body], Body] = None,
        tool_tip_offset: Union[Iterable[Iterable[float]], Iterable[float]] = None,
        dt: Union[Iterable[float], float] = 0.01,
        use_visual_aabb: Union[Iterable[bool], bool] = True,
        apply_shape_scale: Union[Iterable[bool], bool] = True,
    ) -> PartialDesignator[MixingAction]:
        normalized_tip_offset = cls._normalize_tip_offset(tool_tip_offset)
        return PartialDesignator(
            cls,
            container=container,
            arm=arm,
            tool_name=tool_name,
            tool_body=tool_body,
            tool_tip_offset=normalized_tip_offset,
            dt=dt,
            use_visual_aabb=use_visual_aabb,
            apply_shape_scale=apply_shape_scale,
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
        tool_body: Union[Iterable[Body], Body] = None,
        tool_tip_offset: Union[Iterable[Iterable[float]], Iterable[float]] = None,
        dt: Union[Iterable[float], float] = 0.01,
        length: Union[Iterable[float], float] = 0.20,
        cycles: Union[Iterable[float], float] = 2.0,
        container: Union[Iterable[Body], Body] = None,
        target_pose: Union[Iterable[PoseStamped], PoseStamped] = None,
    ) -> PartialDesignator[WipingAction]:
        normalized_tip_offset = cls._normalize_tip_offset(tool_tip_offset)
        return PartialDesignator(
            cls,

            arm=arm,
            container=container,
            target_pose=target_pose,
            tool_name=tool_name,
            tool_body=tool_body,
            tool_tip_offset=normalized_tip_offset,
            dt=dt,
            length=length,
            cycles=cycles,
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

    def _sample_points(self):
        seq = build_cutting_sequence(
            self.container,
            use_visual_aabb=self.use_visual_aabb,
            apply_shape_scale=self.apply_shape_scale,
            technique=self.technique,
            slice_thickness=self.slice_thickness,
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
        tool_body: Union[Iterable[Body], Body] = None,
        tool_tip_offset: Union[Iterable[Iterable[float]], Iterable[float]] = None,
        dt: Union[Iterable[float], float] = 0.01,
        use_visual_aabb: Union[Iterable[bool], bool] = True,
        apply_shape_scale: Union[Iterable[bool], bool] = True,
        technique: Union[Iterable[str], str] = "saw",
        slice_thickness: Union[Iterable[float], float] = 0.03,
    ) -> PartialDesignator[CuttingAction]:
        normalized_tip_offset = cls._normalize_tip_offset(tool_tip_offset)
        return PartialDesignator(
            cls,
            container=container,
            arm=arm,
            tool_name=tool_name,
            tool_body=tool_body,
            tool_tip_offset=normalized_tip_offset,
            dt=dt,
            use_visual_aabb=use_visual_aabb,
            apply_shape_scale=apply_shape_scale,
            technique=technique,
            slice_thickness=slice_thickness,
        )


MixingActionDescription = MixingAction.description
WipingActionDescription = WipingAction.description
CuttingActionDescription = CuttingAction.description
