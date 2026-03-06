from __future__ import annotations

import logging
import time
from typing import Any

import rclpy

from demos.pycram_suturo_demos.helper_methods_and_useful_classes.pickup_helper_methods import parse_color, \
    try_perceive_and_spawn
from demos.pycram_suturo_demos.pycram_basic_hsr_demos.move_demo import move_demo
from demos.pycram_suturo_demos.pycram_basic_hsr_demos.pickup_demo_marc import (
    pickup_demo, get_nearest_object,
)
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import PickUpType
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color
from suturo_resources.queries import query_get_next_object_euclidean_x_y, query_semantic_annotations_on_surfaces, \
    query_annotations_by_color

from pycram.datastructures.enums import PickUpType
"""
was brauche ich:
[X] - Home position
[X] - Navigate to table
[X] - Retrieve item from world
[ ] - Handle if item not existing
[X] - PickUp
[X] - Attach to gripper
[X] - Drive back

"""

from dataclasses import field

from demos.pycram_suturo_demos.helper_methods_and_useful_classes.robot_setup import (
    robot_setup,
)

logger = logging.getLogger(__name__)


def initialization(simulation: bool, with_simulated_objects: bool = False ):
    result = robot_setup(
        simulation=simulation, with_simulated_objects=with_simulated_objects
    )
    logger.info("initialization done")
    return (
        result.node,
        result.world,
        result.robot_view,
        result.context,
        result.manipulator,
    )





def main():
    simulated = True
    with_simulated_objects = True
    object_name = ""
    object_color : Color= Color.WHITE()

    rclpy_node, world, robot_view, context, manipulator = initialization(
        simulation=simulated, with_simulated_objects=with_simulated_objects
    )

    # objects = {1: "milk.stl", 2: "breakfast_cereal.stl"}

    mode: str = input(f"What mode?")
    if mode == "obje1ct search":
        pickup_mode = PickUpType.PICK_UP_OBJECT_SEARCH
        object_name: str = input(f"Which object do you want to pick up?")
        logger.info(f"looking for object: {object_name}")
    elif mode == "object by color":
        pickup_mode = PickUpType.PICK_UP_OBJECT_BY_COLOR
        object_color = parse_color(
            input(f"Which color should the object be? In lowercase please (e.g: red, blue, green)"))
        logger.info(f"object_color: {object_color}")
    else:
        logger.info("defaulting to nearest object")
        pickup_mode = PickUpType.PICK_UP_OBJECT_BY_NEAREST

    move_demo(
        simulated=simulated,
        world=world,
        context=context,
        target_pose="POPCORN_TABLE",
    )

    try_perceive_and_spawn(world)

    pickup_demo(
        simulation=simulated,
        world=world,
        context=context,
        mode=pickup_mode,
        object_name=object_name,
        color=object_color
    )
    move_demo(
        simulated=simulated,
        world=world,
        context=context,
        target_pose="ROBOT_START_POSE",
    )

if __name__ == "__main__":
    main()
