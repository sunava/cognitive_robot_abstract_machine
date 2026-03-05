from rclpy import logging

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, PickUpType
from pycram.language import SequentialPlan
from pycram.motion_executor import real_robot, simulated_robot, ExecutionEnvironment
from pycram.robot_plans import (
    ParkArmsActionDescription,
    GiskardPickUpActionDescription,
)

from demos.pycram_suturo_demos.helper_methods_and_useful_classes.pickup_helper_methods import (
    attach_object_to_hsrb,
    try_perceiving_and_spawning_and_find_object, get_nearest_object, get_object_with_color,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body
from suturo_resources.queries import query_get_next_object_euclidean_x_y, query_annotations_by_color, \
    query_semantic_annotations_on_surfaces


# ------------------------ BASE-DEFINITIONS
def pickup_demo(
    simulation: bool = True,
    world: World = None,
    context: Context = None,
    mode : PickUpType = PickUpType.PICK_UP_OBJECT_SEARCH,
    object_name: str = "",
    color : Color = Color.WHITE()
):
    logger = logging.get_logger(__name__)

    SIMULATED: bool = simulation
    object_name: str = object_name

    robot_type: ExecutionEnvironment = simulated_robot if SIMULATED else real_robot

    # -------------------------------- DETERMIN OBJECT_TO_PICKUP

    if mode == PickUpType.PICK_UP_OBJECT_SEARCH:
        object_to_pickup = try_perceiving_and_spawning_and_find_object(
            world=world, object_name=object_name
        )
        logger.info(f"object_to_pickup: {object_to_pickup}")
    elif mode == PickUpType.PICK_UP_OBJECT_BY_NEAREST:
        object_to_pickup = get_nearest_object(world=world)
    elif mode == PickUpType.PICK_UP_OBJECT_BY_COLOR:
        object_to_pickup = get_object_with_color(world, color)

    # -------------------------------- PLANNING
    plan = SequentialPlan(
        context,
        GiskardPickUpActionDescription(
            simulated=simulation,
            object_designator=object_to_pickup,
            arm=Arms.LEFT,
            gripper_vertical=True,
        ),
    )
    plan_park = SequentialPlan(context, ParkArmsActionDescription(Arms.BOTH))

    # ------------------------ EXECUTION
    with robot_type:
        logger.info("Starting pickup demo")
        plan.perform()
        object_grasped :str= input("Was the object grasped?")
        while object_grasped != "yes":
            # TODO retract and regrasp
            object_grasped = input("Was the object grasped?")
        logger.info("pickup finished")
        attach_object_to_hsrb(world=world, object_designator=object_to_pickup)
        logger.info("parking arms")
        plan_park.perform()
        logger.info("parking arms finished")
