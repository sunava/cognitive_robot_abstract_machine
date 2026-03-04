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
    ParkArmsActionDescription, NavigateActionDescription, CuttingActionDescription, WipingActionDescription

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
from semantic_digital_twin.semantic_annotations.semantic_annotations import Knife, Whisk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import Body

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)


def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def _set_uniform_scale(body, scale_xyz, color=None):
    scale = Scale(*scale_xyz)
    for shape in body.root.visual.shapes:
        shape.scale = scale
        if color is not None:
            shape.color = color
    for shape in body.root.collision.shapes:
        shape.scale = scale


def setup_complex_world():
    world = setup_world()

    bowl = _parse_stl("objects", "bowl.stl")
    bowl.root.name.name = "bowl_small"
    _set_uniform_scale(bowl, (1.0, 1.0, 1.0), color=Color(R=1, G=1, B=0))

    bowl_middle = _parse_stl("objects", "bowl.stl")
    bowl_middle.root.name.name = "bowl_middle"
    _set_uniform_scale(bowl_middle, (1.3, 1.3, 1.3), color=Color(R=1, G=1, B=0))

    bowl_big = _parse_stl("objects", "bowl.stl")
    bowl_big.root.name.name = "bowl_big"
    _set_uniform_scale(bowl_big, (1.5, 1.5, 1.5), color=Color(R=1, G=1, B=0))

    bread = _parse_stl("pycram_object_gap_demo", "bread.stl")
    bread.root.name.name = "bread_small"
    _set_uniform_scale(bread, (1.0, 1.0, 1.0), color=Color(R=0.76, G=0.60, B=0.42))

    bread_middle = _parse_stl("pycram_object_gap_demo", "bread.stl")
    bread_middle.root.name.name = "bread_middle"
    _set_uniform_scale(
        bread_middle,
        (1.3, 1.3, 1.3),
        color=Color(R=0.76, G=0.60, B=0.42),
    )

    bread_big = _parse_stl("pycram_object_gap_demo", "bread.stl")
    bread_big.root.name.name = "bread_big"
    _set_uniform_scale(bread_big, (1.5, 1.5, 1.5), color=Color(R=0.76, G=0.60, B=0.42))

    knife = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "pycram_object_gap_demo", "big-knife.stl"
        )
    ).parse()


    whisk = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "pycram_object_gap_demo", "whisk.stl"
        )
    ).parse()


    l_robot_tip = world.get_body_by_name("l_gripper_tool_frame")
    r_robot_tip = world.get_body_by_name("r_gripper_tool_frame")
    sponge = add_box(
        world, BoxSpec(name="sponge", scale_xyz=(0.05, 0.05, 0.05)), tf_frame="/map", color=Color(R=1, G=1, B=0)
    )

    with world.modify_world():
        connection_knife = FixedConnection(
            parent=r_robot_tip, child=knife.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.08, y=0, z=0, roll=0, pitch=0, yaw=0,
                                                         reference_frame=r_robot_tip)
        )
        connection_whisk = FixedConnection(
            parent=l_robot_tip, child=whisk.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(x=0, y=0, z=0.08, roll=0, pitch=-np.pi/2, yaw=0,
                                                         reference_frame=l_robot_tip)
        )
        # world.merge_world(sponge)
        world.merge_world(knife, connection_knife)
        world.merge_world(whisk, connection_whisk)
        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                3.1, 2, 1, reference_frame=world.root
            ),
        )
        world.merge_world_at_pose(
            bowl_middle,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                3.1, 2.4, 1, reference_frame=world.root
            ),
        )
        world.merge_world_at_pose(
            bowl_big,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                3.1, 2.8, 1, reference_frame=world.root
            ),
        )

        world.merge_world_at_pose(
            bread,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.5, 2, 1, reference_frame=world.root
            ),
        )
        world.merge_world_at_pose(
            bread_middle,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.5, 2.4, 1, reference_frame=world.root
            ),
        )
        world.merge_world_at_pose(
            bread_big,
            HomogeneousTransformationMatrix.from_xyz_rpy(x=2.5, y=2.8, z=1, roll=0, pitch=0, yaw=np.pi/2,
                                                         reference_frame=world.root),

        )
    return world


def main():
    world = setup_complex_world()

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


    knife_body = try_get_body(world, "big-knife.stl")
    knife = Knife(root=knife_body)
    whisk_body = try_get_body(world, "whisk.stl")
    whisk = Whisk(root=whisk_body)
    # sponge_body = try_get_body(world, "sponge")
    bread_body = try_get_body(world, "bread_small")
    bread_middle_body = try_get_body(world, "bread_middle")
    bread_big_body = try_get_body(world, "bread_big")
    #
    bowl_small_body = try_get_body(world, "bowl_small")
    bowl_middle_body = try_get_body(world, "bowl_middle")
    bowl_big_body = try_get_body(world, "bowl_big")
    clean_up_pose = PoseStamped.from_list([2.5,4,0.95])
    context.ros_node = node
    print(PoseStamped.from_spatial_type(context.robot.root.global_pose))
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
        NavigateActionDescription(
            PoseStamped.from_list([2.0 ,2.5,0.0], [0, 0, 0, 1], world.root),
            True,
        ),
        # NavigateActionDescription(target_location=PoseStamped.from_list([1.5,2.5,0.0])),
        CuttingActionDescription(container=bread_body, arm=Arms.RIGHT, tool=knife, technique="saw", clear_viz=True),
        CuttingActionDescription(container=bread_middle_body, arm=Arms.RIGHT, tool=knife, technique="saw"),
        CuttingActionDescription(container=bread_big_body, arm=Arms.RIGHT, tool=knife, technique="saw"),
        MixingActionDescription(container=bowl_small_body, arm=Arms.LEFT, tool=whisk),
        MixingActionDescription(container=bowl_middle_body, arm=Arms.LEFT, tool=whisk),
        MixingActionDescription(container=bowl_big_body, arm=Arms.LEFT, tool=whisk),
        #
        # WipingActionDescription(
        #     target_pose=clean_up_pose,
        #     arm=Arms.LEFT,
        #     tool=None,
        # )
        # SimpleMoveTCPAction(target_location=poses[0], arm=Arms.RIGHT),
    )
    with simulated_robot:
        plan.perform()

if __name__ == "__main__":
    main()
    while True:
        pass
