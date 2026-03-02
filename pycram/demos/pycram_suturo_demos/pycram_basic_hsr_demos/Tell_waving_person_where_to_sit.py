from time import sleep
import time
import rclpy
from suturo_resources.suturo_map import load_environment
from enum import Enum
from pycram.external_interfaces import robokudo
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram_suturo_demos.pycram_basic_hsr_demos.start_up import setup_hsrb_context

from pycram.external_interfaces.robokudo import shutdown_robokudo_interface
from pycram.external_interfaces.robokudo_ros1 import query_specific_region
from pycram.ros_utils.text_to_image import TextToImagePublisher
from semantic_digital_twin.datastructures.definitions import TorsoState
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot, real_robot
from pycram.robot_plans import (
    ParkArmsActionDescription,
    LookAtActionDescription,
)

# rclpy.init()
rclpy_node, world, robot_view, context = setup_hsrb_context()

camera_frame = robot_view.get_default_camera().root
base_frame = world.get_body_by_name("base_link")


class Direction(Enum):
    LEFT = [0.1, 1, 0.75]
    RIGHT = [0.1, -1, 0.75]
    BACK = [-1, 0, 0.75]
    FRONT = [1, 0, 0.75]
    FRONT_DOWN = [1, 0, 0.5]


def look_in_direction(direction):
    look = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        LookAtActionDescription([PoseStamped.from_list(direction, frame=base_frame)]),
    )
    return look


with real_robot:
    text_pub = TextToImagePublisher()
    # look around for human
    human = robokudo.query_waving_human()
    if human is None:
        look_in_direction(Direction.LEFT).perform()
        human = robokudo.query_waving_human()
    if human is None:
        look_in_direction(Direction.RIGHT).perform()
        human = robokudo.query_waving_human()
    if human is None:
        look_in_direction(Direction.BACK).perform()
        human = robokudo.query_waving_human()
    if human is None:
        look_in_direction(Direction.FRONT).perform()
        human = robokudo.query_waving_human()
    # drive to human

    # Screen text "Please follow me"
    text_pub.publish_text(f"Please follow me")

    # drive to sofa

    # look for empty spot
    look_in_direction(Direction.FRONT_DOWN).perform()
    result = query_specific_region("sofa")

    # tell human which spot is empty (with screen)
    text_pub.publish_text(f"Free seat at {result}")
    shutdown_robokudo_interface()
