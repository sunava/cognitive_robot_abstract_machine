from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Union, Optional, Type, Any, Iterable

import numpy as np

from demos.thesis_new.frame_provider import WorldTransformFrameProvider
from demos.thesis_new.motion_models import MotionSegment, MotionSequence, Pose, FixedFrameProvider
from demos.thesis_new.motion_profiles import planar_sweep_x
from demos.thesis_new.motion_presets import build_container_sequence
from demos.thesis_new.world_utils import body_local_aabb, make_identity_pose_stamped
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

WHISK_TIP_Z_OFFSET = 0.0


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


def get_tool_config(name: str) -> ToolConfig:
    return TOOLS.get(str(name), TOOLS["whisk"])


def tip_offset_from_body(body):
    """Estimate the tool tip along +X using the local AABB."""
    mins, maxs = body_local_aabb(body, use_visual=True, apply_shape_scale=True)
    center_y = 0.5 * (mins[1] + maxs[1])
    center_z = 0.5 * (mins[2] + maxs[2])
    return np.array([maxs[0], center_y, center_z], dtype=float)


@dataclass
class SimplePouringAction(ActionDescription):
    """
    Park the arms of the robot.
    """

    object_designator: Body
    """
    The object to pick up
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    def execute(self) -> None:
        for arm_chain in self.robot_view.manipulator_chains:
            grasp = GraspDescription(
                ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
            ).calculate_grasp_orientation(
                arm_chain.manipulator.front_facing_orientation.to_np()
            )

        object_pose = self.object_designator.global_pose
        object_pose.x += 0.009
        object_pose.y -= 0.125
        object_pose.z += 0.17

        def approach_or_rotate(rotate: bool) -> PoseStamped:
            ros_pose = PoseStamped.from_spatial_type(object_pose)

            if rotate:
                q = utils.axis_angle_to_quaternion([1, 0, 0], -110)
                ros_pose.rotate_by_quaternion(utils.quat_np_list(q))

            man = next(iter(self.robot_view.manipulators))
            tool_frame = man.tool_frame

            poseTg = PoseStamped.from_spatial_type(
                self.world.transform(ros_pose.to_spatial_type(), tool_frame)
            )
            poseTg.rotate_by_quaternion(grasp)

            return PoseStamped.from_spatial_type(
                self.world.transform(poseTg.to_spatial_type(), self.world.root)
            )

        pose = approach_or_rotate(False)
        pose_rot = approach_or_rotate(True)

        SequentialPlan(
            self.context,
            MoveTCPMotion(
                pose,
                self.arm,
                allow_gripper_collision=True,
                movement_type=MovementType.CARTESIAN,
            ),
            MoveTCPMotion(
                pose_rot,
                self.arm,
                allow_gripper_collision=True,
                movement_type=MovementType.CARTESIAN,
            ),
        ).perform()

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass

    @classmethod
    def description(
        cls,
        object_designator: Union[Iterable[PoseStamped], PoseStamped],
        arm: Union[Iterable[Arms], Arms],
    ) -> PartialDesignator[Type[SimplePouringAction]]:
        return PartialDesignator(cls, object_designator=object_designator, arm=arm)


@dataclass
class SimpleMoveTCPAction(ActionDescription):
    """
    Represents an action to move a robotic arm's TCP (Tool Center Point) to a target
    location or through a series of waypoints.
    """

    target_location: Union[Iterable[PoseStamped], PoseStamped]
    """
    Target location(s) for the TCP motion. Can be a single PoseStamped object or an iterable of PoseStamped objects.
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    def execute(self) -> None:
        if isinstance(self.target_location, PoseStamped):
            motion = MoveTCPMotion(
                self.target_location,
                self.arm,
                allow_gripper_collision=True,
                movement_type=MovementType.CARTESIAN,
            )
        else:
            motion = MoveTCPWaypointsMotion(
                list(self.target_location),
                self.arm,
                allow_gripper_collision=True,
            )

        SequentialPlan(self.context, motion).perform()

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass

    @classmethod
    def description(
        cls,
        target_location: Union[Iterable[PoseStamped], PoseStamped] = None,
        arm: Union[Iterable[Arms], Arms] = None,
        target_locations: Union[Iterable[PoseStamped], PoseStamped] = None,
    ) -> PartialDesignator[Type[SimpleMoveTCPAction]]:
        resolved_target = (
            target_location if target_location is not None else target_locations
        )
        if resolved_target is None:
            raise ValueError(
                "Provide either target_location or target_locations."
            )
        if arm is None:
            raise ValueError("Provide arm.")

        if isinstance(resolved_target, PoseStamped):
            target_for_designator = resolved_target
        else:
            resolved_target_list = list(resolved_target)
            if all(isinstance(pose, PoseStamped) for pose in resolved_target_list):
                target_for_designator = [resolved_target_list]
            else:
                target_for_designator = resolved_target

        return PartialDesignator(
            cls, target_location=target_for_designator, arm=arm
        )


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

    tool_name: str = "whisk"
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

    def _sample_points(self, seq):
        prov = WorldTransformFrameProvider(
            world=self.world,
            source_frame=self.container,
            root_frame=self.world.root,
            make_identity_spatial=make_identity_pose_stamped,
        )
        _, points, _ = seq.sample(prov, dt=self.dt)
        return points

    def _resolve_tip_offset(self):
        if self.tool_tip_offset is not None:
            return np.asarray(self.tool_tip_offset, dtype=float).reshape(3)
        if self.tool_body is not None:
            return tip_offset_from_body(self.tool_body)
        return np.zeros(3, dtype=float)

    @staticmethod
    def _rotation_tool_x_to_world_z():
        """Return identity rotation (keep current tool orientation)."""
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
    def make_tool_wrist_poses_for_world(cls, points, world, tip_offset, tool_cfg):
        """Convert path points into wrist poses using the selected tool config."""
        T_wrist_tip = np.eye(4, dtype=float)
        wrist_to_tip = np.asarray(tip_offset, dtype=float).reshape(3)
        T_wrist_tip[:3, 3] = 0.0
        T_tip_wrist = np.linalg.inv(T_wrist_tip)

        tangents = cls._tangents_from_points(points)

        poses = []
        for i, p in enumerate(points):
            T_world_tip = np.eye(4, dtype=float)
            if tool_cfg.use_rotation:
                T_world_tip[:3, :3] = cls._rotation_from_tangent(tangents[i])
            else:
                T_world_tip[:3, :3] = cls._rotation_tool_x_to_world_z()
            if tool_cfg.rotation_axis is not None and tool_cfg.rotation_deg:
                T_world_tip[:3, :3] = T_world_tip[:3, :3] @ cls._rotation_from_axis_angle(
                    tool_cfg.rotation_axis, tool_cfg.rotation_deg
                )
            T_world_tip[:3, 3] = np.asarray(p, dtype=float).reshape(3)
            if tool_cfg.apply_tip_in_world_z:
                T_world_tip[2, 3] += tool_cfg.tip_z_offset + wrist_to_tip[0]
            T_world_wrist = T_world_tip @ T_tip_wrist
            if (
                tool_cfg.gripper_x_offset_scale
                or tool_cfg.gripper_y_offset_scale
                or tool_cfg.gripper_z_offset_scale
            ):
                T_world_wrist[:3, 3] += T_world_wrist[:3, :3] @ np.array(
                    [
                        tool_cfg.gripper_x_offset_scale * wrist_to_tip[0],
                        tool_cfg.gripper_y_offset_scale * wrist_to_tip[0],
                        tool_cfg.gripper_z_offset_scale * wrist_to_tip[0],
                    ],
                    dtype=float,
                )
            poses.append(PoseStamped.from_matrix(T_world_wrist, frame=world.root))
        return poses

    def _make_tool_wrist_poses(self, points, tip_offset, tool_cfg):
        print(f"tool_cfg={tool_cfg}, tip_offset={tip_offset}")
        return self.make_tool_wrist_poses_for_world(
            points, self.world, tip_offset, tool_cfg
        )

    def execute(self) -> None:
        seq = self._build_sequence()
        points = self._sample_points(seq)
        tool_cfg = get_tool_config(self.tool_name)
        tip_offset = self._resolve_tip_offset()

        poses = self._make_tool_wrist_poses(points, tip_offset, tool_cfg)
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
        tool_name: Union[Iterable[str], str] = "whisk",
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

    def _sample_points(self, seq):
        pos = self.target_pose.pose.position
        base_pose = Pose(R=np.eye(3, dtype=float), p=[pos.x, pos.y, pos.z])
        prov = FixedFrameProvider(base_pose)
        _, points, _ = seq.sample(prov, dt=self.dt)
        return points

    @classmethod
    def description(
        cls,
        target_pose: Union[Iterable[PoseStamped], PoseStamped],
        arm: Union[Iterable[Arms], Arms],
        tool_name: Union[Iterable[str], str] = "whisk",
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


@dataclass
class SimpleMoveTCPsAction(ActionDescription):
    """
    Park the arms of the robot.
    """

    target_locations: list[PoseStamped]

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    def execute(self) -> None:
        SequentialPlan(
                self.context,
                MoveTCPWaypointsMotion(
                    self.target_locations,
                    self.arm,
                    allow_gripper_collision=True,
                )
            ).perform()


    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass

    @classmethod
    def description(
        cls,
        target_locations: Union[Iterable[list[PoseStamped]], list[PoseStamped]],
        arm: Union[Iterable[Arms], Arms],
    ) -> PartialDesignator[Type[SimpleMoveTCPsAction]]:
        if isinstance(target_locations, list) and (
            len(target_locations) == 0
            or all(isinstance(pose, PoseStamped) for pose in target_locations)
        ):
            normalized_target_locations = [target_locations]
        else:
            normalized_target_locations = target_locations
        return PartialDesignator(
            cls, target_locations=normalized_target_locations, arm=arm
        )


SimplePouringActionDescription = SimplePouringAction.description
SimpleMoveTCPActionDescription = SimpleMoveTCPAction.description
SimpleMoveTCPsActionDescription = SimpleMoveTCPsAction.description

MixingActionDescription = MixingAction.description
WipingActionDescription = WipingAction.description
