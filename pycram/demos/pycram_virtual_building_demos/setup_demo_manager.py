import sys
import time

import rclpy
from IPython.core.display_functions import clear_output

# sys.path.insert(0, '/home/vee/robocup_workspaces/pycram_ws/src/pycram')
sys.path.insert(0, "/home/jovyan/workspace/ros/src/pycram")
# sys.path.insert(0, '/home/me/IAI_work/binder_ws/src/pycram')
import rclpy
from rclpy.node import Node
from demos.pycram_virtual_building_demos.setup.setup_utils import (
    display_loading_gif_with_text,
    update_text,
    get_robot_name,
)

output = None

#
# def start_demo():
#     # get params
#     environment_param = rospy.get_param("/nbparam_environments")
#     robot_param = rospy.get_param("/nbparam_robots")
#     task_param = rospy.get_param("/nbparam_tasks")
#
#     robot_name = get_robot_name(robot_param)
#
#     extension = ObjectDescription.get_file_extension()
#     # text widget for the virtual building
#     text_widget = display_loading_gif_with_text()
#     update_text(text_widget, "Loading process~ Please wait...")
#     # world = BulletWorld(WorldMode.DIRECT)
#     #
#     # # Set this to True to publish costmaps and axis marker during the demo. May slow down the simulation.
#     # world.allow_publish_debug_poses = False
#     #
#     # VizMarkerPublisher()
#     # robot = Object(robot_name, ObjectType.ROBOT, f"{robot_name}{extension}", pose=Pose([1, 2, 0]))
#     # apartment = Object(environment_param, ObjectType.ENVIRONMENT, f"{environment_param}{extension}")
#     # if robot_param not in ["pr2", "icub", "fetch", "tiago", "donbot"]:
#     #     VizMarkerRobotPublisher()
#     # else:
#     #     TFBroadcaster()
#
#     clear_output(wait=True)
#
#     update_text(text_widget, "Executing Demo: " + task_param)
#
#     demo_selecting(environment_param, robot_name, task_param)
#
#     update_text(text_widget, "Done with: " + task_param)
#
# #
# # def start_demo_local():
# #     # get params
# #     environment_param = "robocane"
# #     robot_param = "pr2"
# #     task_param = "transporting"
# #
#     robot_name = get_robot_name(robot_param)
# #
# #     extension = ObjectDescription.get_file_extension()
# #     world = BulletWorld(WorldMode.DIRECT)
# #
# #     robot = Object(
# #         robot_name, ObjectType.ROBOT, f"robots/{robot_param}.urdf", pose=Pose([1, 2, 0])
# #     )
# #     apartment = Object(
# #         environment_param, ObjectType.ENVIRONMENT, f"{environment_param}{extension}"
# #     )
# #
# #     VizMarkerPublisher(interval=0.4, spawn_floor=False)
# #     VizMarkerRobotPublisher(interval=0.2)
# #     time.sleep(10)
# #     demo_selecting(environment_param, robot_name, task_param)
# #     extension = ObjectDescription.get_file_extension()
#


def test_example():
    print("this is just a test")


def demo_selecting(envi, robot, task_param):
    if task_param == "test":
        test_example()
    # if task_param == "navigate":
    #     navigate_simple_example()
    # if task_param == "follow":
    #     follow_simple_example(robot)
    # elif task_param == "transporting" or task_param == "navigate":
    #     specialized_task = None
    #     # specialized_task = rospy.get_param('/nbparam_specialized_task')
    #     if specialized_task == "clean":
    #         cleanup_demo(envi, robot)
    #     else:
    #         transporting_demo(envi, robot)
    # elif task_param in ["cutting", "mixing", "pouring"]:
    #     # object_target = rospy.get_param('/nbparam_object')
    #     # object_tool = rospy.get_param('/nbparam_object_tool')
    #     if task_param == "mixing":
    #         object_target = "big-bowl"
    #         object_tool = "whisk"
    #     elif task_param == "pouring":
    #         object_target = "bowl"
    #         object_tool = "jeroen_cup"
    #     else:
    #         object_target = "banana"
    #         object_tool = "butter_knife"
    #     specialized_task = rospy.get_param('/nbparam_specialized_task')
    #     start_generalized_demo(task_param, object_tool, object_target, specialized_task)


# start_demo_local()
class MyNode(Node):
    def __init__(self):
        super().__init__("my_node")

        self.declare_parameter("nbparam_environments", 0)
        self.declare_parameter("nbparam_robots", 0)
        self.declare_parameter("nbparam_tasks", 0)

        environment_param = self.get_parameter("nbparam_environments").value
        robot_param = self.get_parameter("nbparam_robots").value
        task_param = self.get_parameter("nbparam_tasks").value

        self.get_logger().info(
            f"env={environment_param}, robots={robot_param}, tasks={task_param}"
        )
        text_widget = display_loading_gif_with_text()
        update_text(text_widget, "Loading process~ Please wait...")
        clear_output(wait=True)

        update_text(text_widget, "Executing Demo: " + task_param)
        robot_name = get_robot_name(robot_param)
        demo_selecting(environment_param, robot_name, task_param)

        update_text(text_widget, "Done with: " + task_param)


def demo_start():
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


demo_start()
