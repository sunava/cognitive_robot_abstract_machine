import os
import numpy as np

import rclpy

from demos.thesis.simulation_setup import add_box, BoxSpec
from demos.thesis_new.thesis_math.world_utils import try_get_body
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import MoveTorsoActionDescription, MixingActionDescription, \
    ParkArmsActionDescription
from pycram.testing import setup_world
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color

def _setup_world():
    world = setup_world()

    bowl = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
        )
    ).parse()

    whisk = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "pycram_object_gap_demo", "whisk.stl"
        )
    ).parse()

    sponge = add_box(
        world, BoxSpec(name="sponge", scale_xyz=(0.05, 0.05, 0.05)), tf_frame="/map", color=Color(R=1, G=1, B=0)
    )

    with world.modify_world():
        l_robot_tip = world.get_body_by_name("l_gripper_tool_frame")
        r_robot_tip = world.get_body_by_name("r_gripper_tool_frame")
        world.add_kinematic_structure_entity(sponge)
        connection_sponge = FixedConnection(
            parent=l_robot_tip, child=sponge,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_axis_angle(axis=(0, 1, 0),
                                                                                               angle=np.pi / 2,
                                                                                               reference_frame=l_robot_tip
                                                                                               ))
        connection_whisk = FixedConnection(
            parent=r_robot_tip, child=whisk.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_axis_angle(z=-0.03, axis=(0, 1, 0),
                                                                                               angle=np.pi / 2,
                                                                                               reference_frame=r_robot_tip
                                                                                               )
        )
        world.add_connection( connection_sponge)
        world.merge_world(whisk, connection_whisk)
        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.4, 2.2, 1, reference_frame=world.root
            ),
        )
    return world



def main():
    world = _setup_world()

    rclpy.init()
    node = rclpy.create_node("pycram_demo")

    tf_wrapper = TFWrapper(node=node)
    TFPublisher(node=node, world=world)
    VizMarkerPublisher(world, node)

    tf_wrapper.wait_for_transform(
        "apartment/apartment_root",
        "pr2/base_footprint",
        timeout=RclpyDuration(seconds=1.0),
        time=Time(),
    )

    PR2.from_world(world)
    context = Context.from_world(world)


    whisk_body = try_get_body(world, "whisk.stl")
    sponge_body = try_get_body(world, "sponge")
    bowl_body = try_get_body(world, "bowl.stl")


    clean_up_pose=PoseStamped()
    clean_up_pose.pose.position.x=2.26
    clean_up_pose.pose.position.y=2.59
    clean_up_pose.pose.position.z=0.95



    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
        MixingActionDescription(container=bowl_body, arm=Arms.RIGHT, tool_body=whisk_body),
        # WipingActionDescription(
        #     target_pose=clean_up_pose,
        #     arm=Arms.LEFT,
        #     tool_name=SPONGE_TOOL_NAME,
        #     tool_body=sponge_body,
        # )
        # SimpleMoveTCPAction(target_location=poses[0], arm=Arms.RIGHT),
    )

    with simulated_robot:
        plan.perform()

if __name__ == "__main__":
    main()
