import os

import rclpy
from black.nodes import parent_type

from demos.thesis.simulation_setup import add_box, BoxSpec
from demos.thesis_new.frame_provider import WorldTransformFrameProvider
from demos.thesis_new.motion_presets import build_default_sequence, build_container_sequence
from demos.thesis_new.motion_models import Pose, FixedFrameProvider
from demos.thesis_new.rviz import MotionSequenceRviz
from demos.thesis_new.world_utils import try_get_body, make_identity_pose_stamped, body_local_aabb
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import MoveTorsoActionDescription, MoveTCPMotion, SimpleMoveTCPAction
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
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.abstract_robot import Arm
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Scale


def _setup_world():
    """Create a demo world with a bowl and a box."""
    world = setup_world()

    bowl = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
        )
    ).parse()


    with world.modify_world():
        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.4, 2.2, 1, reference_frame=world.root
            ),

        )

    with world.modify_world():
        box = add_box(
            world, BoxSpec(name="muh_box", scale_xyz=(0.3, 0.1, 0.1)), tf_frame="/map"
        )
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.9, y=1.0, z=0.95
                ),
            )
        )

    return world



def main():
    """Run the RViz demo for default and bowl-constrained sequences."""
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

    with world.modify_world():
        world_reasoner = WorldReasoner(world)
        world_reasoner.reason()
        world.add_semantic_annotations(
            [
                Bowl(root=world.get_body_by_name("bowl.stl")),
            ]
        )


    plan = SequentialPlan(
        context,
        MoveTorsoActionDescription(TorsoState.HIGH)
    )
    with simulated_robot:
        plan.perform()

    with world.modify_world():
        whisk = STLParser(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "resources", "pycram_object_gap_demo", "whisk.stl"
            )
        ).parse()
        robot_tip=world.get_body_by_name("r_gripper_tool_frame")
        connection = FixedConnection(
            parent=robot_tip, child=whisk.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_quaternion(
            0.1, 0, 0, reference_frame=robot_tip
        ),
        )
        world.merge_world(whisk, connection)


if __name__ == "__main__":
    main()
