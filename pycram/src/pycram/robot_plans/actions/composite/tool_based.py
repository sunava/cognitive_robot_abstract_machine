from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Union, Optional, Type, Any, Iterable

import numpy as np

from demos.thesis_new.thesis_math.motion_models import MotionSegment, MotionSequence
from demos.thesis_new.thesis_math.motion_profiles import planar_sweep_x
from demos.thesis_new.thesis_math.motion_presets import build_container_sequence
from demos.thesis_new.thesis_math.world_utils import (
    body_local_aabb,
    make_identity_pose_stamped,
    Rp_from_spatial,
)
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



@dataclass
class ToolConfig:
    name: str
    use_rotation: bool
    apply_tip_in_world_z: bool
    gripper_x_offset_scale: float = 0.0
    gripper_y_offset_scale: float = 0.0
    gripper_z_offset_scale: float = 0.0
    rotation_axis: Optional[Iterable[float]] = None
    rotation_deg: float = 0.0
    tip_z_offset: float = 0.0


TOOLS = {
    "whisk": ToolConfig(
        name="whisk",
        use_rotation=False,
        apply_tip_in_world_z=True,
        gripper_z_offset_scale=0.5,
    ),
    "knife": ToolConfig(
        name="knife",
        use_rotation=True,
        apply_tip_in_world_z=False,
        gripper_x_offset_scale=-1.0,
    ),
    "sponge": ToolConfig(
        name="sponge",
        use_rotation=True,
        apply_tip_in_world_z=False,
        gripper_z_offset_scale=0,
        rotation_axis=[0.0, 1.0, 0],
        rotation_deg=90.0,
    ),
}


def get_tool_config(name: Optional[str]) -> Optional[ToolConfig]:
    if not name:
        return None
    return TOOLS.get(str(name))


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

    container: Optional[Body] = None
    """
    The container (e.g., bowl) to operate in.
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

    def _build_sequence(self):
        raise NotImplementedError


    def _resolve_tip_offset(self):
        if self.tool_tip_offset is not None:
            return np.asarray(self.tool_tip_offset, dtype=float).reshape(3)
        if self.tool_body is not None:
            return tip_offset_from_body(self.tool_body)
        return np.zeros(3, dtype=float)

    @staticmethod
    def _identity_rotation():
        return np.eye(3, dtype=float)

    @staticmethod
    def _tangents_from_points(points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        tangents = np.zeros_like(pts)
        if len(pts) < 2:
            return tangents
        diffs = np.diff(pts, axis=0)
        tangents[:-1] = diffs
        tangents[-1] = diffs[-1]
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return tangents / norms

    @staticmethod
    def _rotation_from_tangent(tangent):
        x_axis = np.asarray(tangent, dtype=float).reshape(3)
        x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-8)
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(x_axis, up)) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=float)
        y_axis = np.cross(up, x_axis)
        y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-8)
        z_axis = np.cross(x_axis, y_axis)
        return np.column_stack([x_axis, y_axis, z_axis])

    @staticmethod
    def _rotation_from_axis_angle(axis, angle_deg):
        axis = np.asarray(axis, dtype=float).reshape(3)
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            return np.eye(3, dtype=float)
        axis = axis / norm
        angle = np.deg2rad(angle_deg)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1.0 - c
        return np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ],
            dtype=float,
        )

    @classmethod
    def make_tool_wrist_poses_for_world(
        cls, points, world, tip_offset, tool_cfg: Optional[ToolConfig] = None
    ):
        """Convert tip path points into wrist poses.

        Default behavior is neutral and tool-agnostic: identity orientation and
        wrist placement from the provided tip offset. Tool config is optional.
        """
        wrist_to_tip = np.asarray(tip_offset, dtype=float).reshape(3)
        tangents = cls._tangents_from_points(points) if tool_cfg is not None else None

        poses = []
        for i, p in enumerate(points):
            R_world = cls._identity_rotation()
            if tool_cfg is not None and tool_cfg.use_rotation:
                R_world = cls._rotation_from_tangent(tangents[i])
            if (
                tool_cfg is not None
                and tool_cfg.rotation_axis is not None
                and tool_cfg.rotation_deg
            ):
                R_world = R_world @ cls._rotation_from_axis_angle(
                    tool_cfg.rotation_axis, tool_cfg.rotation_deg
                )

            p_world = np.asarray(p, dtype=float).reshape(3).copy()
            p_world -= R_world @ wrist_to_tip
            if tool_cfg is not None and tool_cfg.apply_tip_in_world_z:
                p_world[2] += tool_cfg.tip_z_offset
            if tool_cfg is not None and (
                tool_cfg.gripper_x_offset_scale
                or tool_cfg.gripper_y_offset_scale
                or tool_cfg.gripper_z_offset_scale
            ):
                tip_x = wrist_to_tip[0]
                p_world += R_world @ np.array(
                    [
                        tool_cfg.gripper_x_offset_scale * tip_x,
                        tool_cfg.gripper_y_offset_scale * tip_x,
                        tool_cfg.gripper_z_offset_scale * tip_x,
                    ],
                    dtype=float,
                )
            T_world_wrist = np.eye(4, dtype=float)
            T_world_wrist[:3, :3] = R_world
            T_world_wrist[:3, 3] = p_world
            poses.append(PoseStamped.from_matrix(T_world_wrist, frame=world.root))
        return poses


    def execute(self) -> None:
        seq = self._build_sequence()
        _, points, _ = seq.sample(frame=self.container.global_pose.to_np(), dt=self.dt)

        tool_cfg = get_tool_config(self.tool_name)
        tip_offset = self._resolve_tip_offset()

        poses = self.make_tool_wrist_poses_for_world(
            points, self.world, tip_offset, tool_cfg
        )


        SequentialPlan(
            self.context,
            MoveTCPWaypointsMotion(
                poses,
                self.arm,
                allow_gripper_collision=True,
            ),
        ).perform()

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

    def _build_sequence(self):
        return build_container_sequence(
            self.container,
            use_visual_aabb=self.use_visual_aabb,
            apply_shape_scale=self.apply_shape_scale,
        )

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
    ) -> PartialDesignator[Type[MixingAction]]:
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

    target_pose: PoseStamped
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

    def _build_sequence(self):
        segment = MotionSegment(
            name="planar_sweep",
            duration_s=1.5,
            local_curve=lambda tau: planar_sweep_x(
                tau, length=self.length, cycles=self.cycles
            ),
        )
        return MotionSequence([segment])

    @classmethod
    def description(
        cls,
        target_pose: Union[Iterable[PoseStamped], PoseStamped],
        arm: Union[Iterable[Arms], Arms],
        tool_name: Union[Iterable[Optional[str]], Optional[str]] = None,
        tool_body: Union[Iterable[Body], Body] = None,
        tool_tip_offset: Union[Iterable[Iterable[float]], Iterable[float]] = None,
        dt: Union[Iterable[float], float] = 0.01,
        length: Union[Iterable[float], float] = 0.20,
        cycles: Union[Iterable[float], float] = 2.0,
    ) -> PartialDesignator[Type[WipingAction]]:
        normalized_tip_offset = cls._normalize_tip_offset(tool_tip_offset)
        return PartialDesignator(
            cls,
            target_pose=target_pose,
            arm=arm,
            tool_name=tool_name,
            tool_body=tool_body,
            tool_tip_offset=normalized_tip_offset,
            dt=dt,
            length=length,
            cycles=cycles,
        )



MixingActionDescription = MixingAction.description
WipingActionDescription = WipingAction.description
