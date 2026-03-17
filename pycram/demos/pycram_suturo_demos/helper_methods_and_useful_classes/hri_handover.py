from __future__ import annotations

import logging

import rclpy

import nlp_human_robot_interaction as hri
from time import sleep

from demos.pycram_suturo_demos.pycram_basic_hsr_demos.gripper_open_close_demo import (
    GripperActionClient,
)
from giskardpy.motion_statechart.graph_node import Task
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from pycram.language import SequentialPlan
from pycram.motion_executor import ExecutionEnvironment, simulated_robot, real_robot
from pycram.robot_plans import BaseMotion
from semantic_digital_twin.datastructures.joint_state import JointState
from test.conftest import hsr_world_setup


class HandoverMotion(BaseMotion):
    def perform(self):
        return

    def motion_chart(self) -> Task:
        prehandover_goal = JointPositionList(
            goal_state=JointState.from_str_dict({
                "arm_lift_joint": 0.30,  # arm raised a bit
                "arm_flex_joint": -1.0,  # flex arm forward to roughly horizontal
                "arm_roll_joint": 0.0,  # neutral roll, arm faces forward
                "wrist_flex_joint": -0.5,  # tilt gripper opening slightly upward
                "wrist_roll_joint": 0.0,  # neutral wrist roll
                # Bonus:
                "head_pan_joint": 0.0,  # head facing forward
                "head_tilt_joint": -0.3,  # head looking slightly downward at hands
                "torso_lift_joint": 0.25,  # raise torso to ~mid-high
            },
                world=hsr_world_setup,
            ),
        )

        return prehandover_goal




from demos.pycram_suturo_demos.helper_methods_and_useful_classes.robot_setup import (
    robot_setup,
)


def initialization(simulation: bool = True):
    logger = logging.getLogger(__name__)

    result = robot_setup(simulation=simulation)
    rclpy_node, world, robot_view, context = (
        result.node,
        result.world,
        result.robot_view,
        result.context,
    )
    return rclpy_node, world, robot_view, context



def countdown(n, node: hri.TalkingNode):
    while n > 0:
        node.pub(str(n))
        sleep(1.5)
        n -= 1


def take_object_from_human():
    talk = hri.TalkingNode()
    gripper_open = 0.8
    gripper_close = -0.8

    grippy = GripperActionClient()
    grippy.send_goal(effort=gripper_open)

    talk.pub("Please put the object into my gripper. Closing gripper in ")
    countdown(5, talk)
    grippy.send_goal(effort=gripper_close)
    sleep(5)


def give_object_to_human():
    talk = hri.TalkingNode()
    gripper_open = 0.8

    grippy = GripperActionClient()

    talk.pub("Please take the object from my gripper. Opening gripper in ")
    countdown(5, talk)
    grippy.send_goal(effort=gripper_open)
    sleep(5)


def main():
    with_giskard = True
    rclpy.init()

    if with_giskard:
        SIMULATED = True
        robot_type: ExecutionEnvironment = simulated_robot if SIMULATED else real_robot

        rclpy_node, world, robot_view, context = initialization(simulation=SIMULATED)
        with robot_type:

            SequentialPlan(
                context,
                HandoverMotion(),
            ).perform()

    take_object_from_human()
    # give_object_to_human()
    sleep(5)


if __name__ == "__main__":
    main()
