from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple, List

from typing_extensions import Optional, Dict, Any

from pycram.datastructures.enums import AxisIdentifier, Arms

from pycram.datastructures.trajectory import PoseTrajectory
from pycram.plans.factories import execute_single, sequential
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.motions.gripper import MoveGripperMotion, MoveTCPWaypointsMotion
from pycram.robot_plans.motions.robot_body import MoveJointsMotion
from pycram.validation.goal_validator import create_multiple_joint_goal_validator
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import (
    TorsoState,
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.spatial_types import Vector3

logger = logging.getLogger(__name__)


@dataclass
class MoveTorsoAction(ActionDescription):
    """
    Move the torso of the robot up and down.
    """

    torso_state: TorsoState
    """
    The state of the torso that should be set
    """

    def execute(self) -> None:
        joint_state = self.robot.torso.get_joint_state_by_type(self.torso_state)
        if len(joint_state.connections) == 0:
            logger.debug(
                "Skipping torso motion '%s' because robot '%s' exposes no torso joints.",
                self.torso_state,
                self.robot.root.name,
            )
            return
        self.add_subplan(
            execute_single(
                MoveJointsMotion(
                    [c.name.name for c in joint_state.connections],
                    joint_state.target_values,
                ),
            )
        ).perform()

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        """
        Create a goal validator for the joint positions and wait until the goal is achieved or the timeout is reached.
        """

        joint_positions: dict = (
            RobotDescription.current_robot_description.get_static_joint_chain(
                "torso", self.torso_state
            )
        )
        validator = create_multiple_joint_goal_validator(
            World.current_world.robot, joint_positions
        )
        validator.wait_until_goal_is_achieved(
            max_wait_time=max_wait_time, time_per_read=timedelta(milliseconds=20)
        )
        if not validator.goal_achieved:
            raise TorsoGoalNotReached(validator)


@dataclass
class SetGripperAction(ActionDescription):
    """
    Set the gripper state of the robot.
    """

    gripper: Arms
    """
    The gripper that should be set 
    """
    motion: GripperState
    """
    The motion that should be set on the gripper
    """

    def execute(self) -> None:
        arms = [Arms.LEFT, Arms.RIGHT] if self.gripper == Arms.BOTH else [self.gripper]
        motions = []
        skipped_arms = []
        for arm in arms:
            end_effector = ViewManager().get_end_effector_view(arm, self.robot)
            joint_state = end_effector.get_joint_state_by_type(self.motion)
            if len(joint_state.connections) == 0:
                skipped_arms.append(arm.name)
                continue
            motions.append(MoveGripperMotion(gripper=arm, motion=self.motion))

        if not motions:
            logger.debug(
                "Skipping gripper motion '%s' because no addressed gripper exposes controllable joints (%s).",
                self.motion,
                ", ".join(skipped_arms) if skipped_arms else self.gripper.name,
            )
            return

        if skipped_arms:
            logger.debug(
                "Skipping gripper motion '%s' for unsupported grippers: %s",
                self.motion,
                ", ".join(skipped_arms),
            )

        self.add_subplan(sequential(motions)).perform()

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        """
        Needs gripper state to be read or perceived.
        """
        pass


@dataclass
class ParkArmsAction(ActionDescription):
    """
    Park the arms of the robot.
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    def execute(self) -> None:
        joint_names, joint_poses = self.get_joint_poses()
        if len(joint_names) == 0:
            logger.debug(
                "Skipping park-arms action because robot '%s' exposes no park joint targets for arm selection '%s'.",
                self.robot.root.name,
                self.arm,
            )
            return

        self.add_subplan(
            execute_single(MoveJointsMotion(names=joint_names, positions=joint_poses))
        ).perform()

    def get_joint_poses(self) -> Tuple[List[str], List[float]]:
        """
        :return: The joint positions that should be set for the arm to be in the park position.
        """
        arm_chain = ViewManager().get_all_arm_views(self.arm, self.robot)
        names = []
        values = []
        for arm in arm_chain:
            joint_state = arm.get_joint_state_by_type(StaticJointState.PARK)
            names.extend([c.name.name for c in joint_state.connections])
            values.extend(joint_state.target_values)
        return names, values


@dataclass
class CarryAction(ActionDescription):
    """
    Parks the robot's arms. And align the arm with the given Axis of a frame.
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    align: Optional[bool] = False
    """
    If True, aligns the end-effector with a specified axis.
    """

    tip_link: Optional[str] = None
    """
    Name of the tip link to align with, e.g the object.
    """

    tip_axis: Optional[AxisIdentifier] = None
    """
    Tip axis of the tip link, that should be aligned.
    """

    root_link: Optional[str] = None
    """
    Base link of the robot; typically set to the torso.
    """

    root_axis: Optional[AxisIdentifier] = None
    """
    Goal axis of the root link, that should be used to align with.
    """

    def execute(self) -> None:
        joint_names, joint_poses = self.get_joint_poses()
        tip_normal = self.axis_to_vector3_stamped(self.tip_axis, link=self.tip_link)
        root_normal = self.axis_to_vector3_stamped(self.root_axis, link=self.root_link)
        self.add_subplan(
            execute_single(
                MoveJointsMotion(
                    names=joint_names,
                    positions=joint_poses,
                    align=self.align,
                    tip_link=self.tip_link,
                    tip_normal=tip_normal,
                    root_link=self.root_link,
                    root_normal=root_normal,
                )
            )
        ).perform()

    def get_joint_poses(self) -> Tuple[List[str], List[float]]:
        """
        :return: The joint positions that should be set for the arm to be in the park position.
        """
        arm_chain = ViewManager().get_all_arm_views(self.arm, self.robot)
        names = []
        values = []
        for arm in arm_chain:
            joint_state = arm.get_joint_state_by_type(StaticJointState.PARK)
            names.extend([c.name.name for c in joint_state.connections])
            values.extend(joint_state.target_values)
        return names, values

    def axis_to_vector3_stamped(
        self, axis: AxisIdentifier, link: str = "base_link"
    ) -> Vector3:
        v = {
            AxisIdentifier.X: Vector3(x=1.0, y=0.0, z=0.0),
            AxisIdentifier.Y: Vector3(x=0.0, y=1.0, z=0.0),
            AxisIdentifier.Z: Vector3(x=0.0, y=0.0, z=1.0),
        }[axis]
        v.frame_id = link
        return v


@dataclass
class FollowToolCenterPointPathAction(ActionDescription):
    """
    Represents an action to move a robotic arm's TCP (Tool Center Point) along a
    path of poses.
    """

    target_locations: PoseTrajectory
    """
    Path poses for the TCP motion.
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    def execute(self) -> None:
        target_locations = list(self.target_locations.poses)

        motion = MoveTCPWaypointsMotion(
            target_locations,
            self.arm,
            allow_gripper_collision=True,
        )

        self.add_subplan(execute_single(motion)).perform()

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass
