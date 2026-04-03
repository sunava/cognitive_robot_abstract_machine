from dataclasses import dataclass, field
from typing import Optional, List

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.collision_avoidance import UpdateTemporaryCollisionRules
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
    CartesianPositionTrajectory,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from pycram.datastructures.dataclasses import AlignmentPair
from semantic_digital_twin.collision_checking.collision_rules import AllowCollisionBetweenGroups
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Vector3
from semantic_digital_twin.world_description.world_entity import Body
from pycram.robot_plans.motions.base import BaseMotion
from pycram.datastructures.enums import (
    Arms,
    MovementType,
    WaypointsMovementType,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.view_manager import ViewManager
from pycram.utils import translate_pose_along_local_axis


@dataclass
class ReachMotion(BaseMotion):
    """ """

    object_designator: Body
    """
    Object designator_description describing the object that should be picked up
    """
    arm: Arms
    """
    The arm that should be used for pick up
    """
    grasp_description: GraspDescription
    """
    The grasp description that should be used for picking up the object
    """
    movement_type: MovementType = MovementType.CARTESIAN
    """
    The type of movement that should be performed.
    """
    reverse_pose_sequence: bool = False
    """
    Reverses the sequence of poses, i.e., moves away from the object instead of towards it. Used for placing objects.
    """

    def _calculate_pose_sequence(self) -> List[Pose]:
        end_effector = ViewManager.get_end_effector_view(self.arm, self.robot)

        target_pose = GraspDescription.get_grasp_pose(
            self.grasp_description, end_effector, self.object_designator
        )
        target_pose.rotate_by_quaternion(
            GraspDescription.calculate_grasp_orientation(
                self.grasp_description,
                end_effector.front_facing_orientation.to_np(),
            )
        )
        target_pre_pose = translate_pose_along_local_axis(
            target_pose,
            end_effector.front_facing_axis.to_np()[:3],
            -0.05,  # TODO: Maybe put these values in the semantic annotates
        )

        pose = self.world.transform(target_pre_pose, self.world.root)

        sequence = [target_pre_pose, pose]
        return sequence.reverse() if self.reverse_pose_sequence else sequence

    def perform(self):
        pass

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        nodes = [
            CartesianPose(
                root_link=self.robot.root,
                tip_link=tip,
                goal_pose=pose,
                threshold=0.005,
                name="Reach",
            )
            for pose in self._calculate_pose_sequence()
        ]
        return Sequence(nodes=nodes)


@dataclass
class MoveGripperMotion(BaseMotion):
    """
    Opens or closes the gripper
    """

    motion: GripperState
    """
    Motion that should be performed, either 'open' or 'close'
    """
    gripper: Arms
    """
    Name of the gripper that should be moved
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper is allowed to collide with something
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        arm = ViewManager().get_end_effector_view(self.gripper, self.robot)

        return JointPositionList(
            goal_state=arm.get_joint_state_by_type(self.motion),
            name=(
                "OpenGripper" if self.motion == GripperState.OPEN else "CloseGripper"
            ),
        )


@dataclass
class MoveToolCenterPointMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    target: Pose
    """
    Target pose to which the TCP should be moved
    """
    arm: Arms
    """
    Arm with the TCP that should be moved to the target
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something
    """
    movement_type: Optional[MovementType] = MovementType.CARTESIAN
    """
    The type of movement that should be performed.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        root = self.world.root if self.robot.full_body_controlled else self.robot.root
        task = None
        if self.movement_type == MovementType.TRANSLATION:
            task = CartesianPosition(
                root_link=root,
                tip_link=tip,
                goal_point=self.target.to_position(),
                name="MoveTCP",
            )
        else:
            task = CartesianPose(
                root_link=root,
                tip_link=tip,
                goal_pose=self.target,
                name="MoveTCP",
            )
        return task


@dataclass
class MoveTCPWaypointsMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    waypoints: List[Pose]
    """
    Waypoints the TCP should move along 
    """
    arm: Arms
    """
    Arm with the TCP that should be moved to the target
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something
    """
    movement_type: WaypointsMovementType = (
        WaypointsMovementType.ENFORCE_ORIENTATION_FINAL_POINT
    )
    """
    The type of movement that should be performed.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        root = self.world.root if self.robot.full_body_controlled else self.robot.root
        nodes = [
            CartesianPose(
                root_link=root,
                tip_link=tip,
                goal_pose=pose,
                # threshold=0.005,
            )
            for pose in self.waypoints
        ]
        return Sequence(nodes=nodes)


@dataclass
class MoveTCPWaypointsAlignedMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    waypoints: List[Point3]
    """
    Waypoints the TCP should move along 
    """
    arm: Arms
    """
    Arm with the TCP that should be moved to the target
    """
    alignment_pairs: List[AlignmentPair] = field(default_factory=list)
    """
    List of alignment pairs for AlignPlanes constraints.
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something
    """

    movement_type: WaypointsMovementType = (
        WaypointsMovementType.ENFORCE_ORIENTATION_FINAL_POINT
    )
    """
    The type of movement that should be performed.
    """
    tip: Optional[Body] = None
    """
    The end effector that should be used to perform the movement.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        if not self.waypoints:
            raise ValueError("No waypoints provided to MoveTCPWaypointsAlignedMotion.")

        if self.tip is None:
            tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
            if tip is None:
                raise ValueError(f"No tool frame available for arm {self.arm}.")

            tip_children = getattr(tip, "child_kinematic_structure_entities", []) or []
            tip_link = next((child for child in tip_children if child is not None), tip)
        else:
            tip_link = self.tip

        root_link = (
            self.world.root if self.robot.full_body_controlled else self.robot.root
        )
        if root_link is None:
            root_link = self.world.root

        motion_state_chart_nodes = self._only_allow_gripper_collision_rules(self.arm)
        tasks = [
            CartesianPositionTrajectory(
                root_link=root_link,
                tip_link=tip_link,
                goal_points=self.waypoints,
                weight=DefaultWeights.WEIGHT_BELOW_CA,
                name="MoveTCPWaypointsAligned",
            )
        ]
        tasks.extend(
            AlignPlanes(
                tip_link=tip_link,
                root_link=root_link,
                tip_normal=pair.tip_normal,
                goal_normal=pair.goal_normal,
                weight=DefaultWeights.WEIGHT_BELOW_CA,
            )
            for pair in self.alignment_pairs
        )
        if self.robot.name.name == "rollin_justin":
            tasks.append(
                AlignPlanes(
                    tip_link=self.world.get_body_by_name("torso4"),
                    root_link=self.world.get_body_by_name("torso1"),
                    tip_normal=Vector3.X(self.world.get_body_by_name("torso4")),
                    goal_normal=Vector3.Z(self.world.get_body_by_name("torso1")),
                    weight=DefaultWeights.WEIGHT_ABOVE_CA,
                )
            )
        motion_state_chart_nodes.append(Parallel(tasks))
        return Parallel(motion_state_chart_nodes)


    def _only_allow_gripper_collision_rules(
        self, arm: Arms
    ) -> list[MotionStatechartNode]:
        """
        Returns collision rules that only allow collisions between the manipulator of an arm and the environment.

        :param arm: The arm for which to get the collision rules
        """
        manipulator_bodies = (
            ViewManager()
            .get_end_effector_view(arm, self.robot)
            .bodies_with_collision
        )
        rules = [
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AllowCollisionBetweenGroups(
                        self.world.bodies_with_collision, manipulator_bodies
                    )
                ]
            )
        ]
        rules.extend(self.robot.special_constraints)
        return rules