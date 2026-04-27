import os

from demos.bachelor_thesis.hsrb_setup_world import hsrb_setup_world
from pycram.locations.costmaps import VisibilityCostmap
from rclpy.node import Node

from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential, execute_single
from pycram.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Quaternion, HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

from pycram.datastructures.dataclasses import Context

# robot camera stats
min_distance_toya = 0.8
max_distance_toya = 3.5
horizontal_fov_toya = 58 # degrees
vertical_fov_toya = 45 # degrees

def check_distance(min_distance: float, max_distance: float, location: Pose, world: World):
    """
    Checks if a location is within a certain distance. (In this case if a location is within the visible distance of the
    robot.
    """
    # world.rob
    # print(world.get_body_by_name("base_link").global_pose)





# class TalkingNode(Node):
#     """
#     ROS2 node that interfaces with a speech/NLP system.
#
#     Responsibilities:
#     -----------------
#     - Publish a trigger message to start the speech recognition/NLP pipeline.
#     - Subscribe to the processed NLP output.
#     - Parse NLP output into a structured Python list.
#     - Expose a blocking `talk_nlp()` call that waits for NLP results.
#     """
#
#     def __init__(self):
#
#         # Initialize ROS2 node with name "talking"
#         super().__init__('nlp_interface_talking')
#
#         # Publisher to let Toya talk
#         self.talk_pub = self.create_publisher(
#             String,
#             '/tts_text',
#             10
#         )
#
#     def pub(self, text: str, delay: int = 0):
#         msg = String()
#         msg.data = text
#         self.get_logger().info(f"Publishing: {text}")
#
#         self.talk_pub.publish(msg)
#         time.sleep(delay)


bowl = STLParser(
    os.path.join(
        os.path.dirname(__file__), "../../..", "resources", "objects", "bowl.stl"
    )
).parse()

# height chosen based on Toyas min and max height
def look_at(location: Pose, robot_world: World):
    vis = VisibilityCostmap(min_height=1, max_height=1.3, target_object=location, resolution=1080, world=robot_world)
    return vis





if __name__ == '__main__':
    world = hsrb_setup_world()


    try:
        import rclpy

        try:
            rclpy.init()
        except:
            pass
        from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
            VizMarkerPublisher,
        )

        node = rclpy.create_node("viz_marker")
        v = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
    except ImportError:
        node = None

    hsrb = HSRB.from_world(world)
    context = Context(world=world, robot=hsrb)

    with world.modify_world():
        world_reasoner = WorldReasoner(world)
        world_reasoner.reason()

        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.4, 2.2, 1.1, reference_frame=world.root
            ),
        )

    context.evaluate_conditions = False

    with simulated_robot:
        execute_single(
            NavigateAction(
                target_location=Pose(Point3(3.2095706, 6.522722, 0),
                                     orientation=(Quaternion(z=-0.9995140, w=0.03117147)),
                                     reference_frame=world.root), keep_joint_states=True),
            context=context,

        ).perform()

        #check_distance(min_distance_toya, max_distance_toya, Pose(Point3(2.3, 6.2, 1)), world)




        visualize = look_at(location=Pose(Point3(2.3, 8, 1.25), orientation=(Quaternion(z=-0.9995140, w=0.03117147))), robot_world=world)

        print(visualize.map)
        import matplotlib.pyplot as plt

        plt.imshow(visualize.map)
        plt.colorbar()
        plt.title("Visibility Costmap2")

        plt.savefig("visibility_costmap2.png", dpi=300)
        plt.close()


