from typing import Any

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import ParallelGripper
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from suturo_resources.queries import query_semantic_annotations_on_surfaces, query_annotations_by_color, \
    query_get_next_object_euclidean_x_y
from suturo_resources.suturo_map import load_environment

import semantic_digital_twin
import logging

logger = logging.getLogger(__name__)


def try_get_object_to_pickup(world, object_name_method) -> Body | None:
    try:
        object_to_pickup_method = world.get_body_by_name(object_name_method)
        logger.info(f"picking up object with name '{object_name_method}'")

        return object_to_pickup_method
    except semantic_digital_twin.exceptions.WorldEntityNotFoundError:
        raise Exception(f"object with name '{object_name_method}' not found")


def add_box(name: str, scale_xyz: tuple[float, float, float]):
    body = Body(
        name=PrefixedName(name),
        collision=ShapeCollection([Box(scale=Scale(*scale_xyz))]),
    )
    new_world = World()
    with new_world.modify_world():
        new_world.add_body(body)
    return new_world


def perceive_and_spawn_all_objects(world: World):
    try:
        from pycram.external_interfaces import robokudo
    except ImportError:
        raise ImportError()
        return {}
    perceived_objects = {}
    perceived_objects_result = robokudo.query_all_objects().res
    for perceived_object in perceived_objects_result:
        object_size = perceived_object.shape_size[0].dimensions
        object_pose = perceived_object.pose[0].pose
        object_time = perceived_object.pose[0].header.stamp
        object_name = f"{perceived_object.type}"
        try:
            object_to_spawn = world.get_body_by_name(object_name)
            with world.modify_world():
                world.move_branch_to_new_world(object_to_spawn)
        except semantic_digital_twin.exceptions.WorldEntityNotFoundError:
            pass
        object_to_spawn = add_box(
            object_name,
            (object_size.x, object_size.y, object_size.z),
        )
        env_world = load_environment()
        perceived_objects[object_name] = object_to_spawn
        with world.modify_world():
            world.merge_world(env_world)
            world.merge_world_at_pose(
                object_to_spawn,
                pose=HomogeneousTransformationMatrix.from_xyz_quaternion(
                    pos_x=object_pose.position.x,
                    pos_y=object_pose.position.y,
                    pos_z=object_pose.position.z,
                    quat_x=object_pose.orientation.x,
                    quat_y=object_pose.orientation.y,
                    quat_z=object_pose.orientation.z,
                    quat_w=object_pose.orientation.w,
                ),
            )
    return perceived_objects


"""
It is a helper method that attaches the object to the robot, since the attaching withing Actions doesnt work with motions

:param world: The world in which to attach the object
:param object_designator: The object to attach
"""


def attach_object_to_hsrb(world: World, object_designator: Body):
    # Attach the object to the end effector
    manipulator = world.get_semantic_annotations_by_type(ParallelGripper)[0]

    with world.modify_world():
        world.move_branch_with_fixed_connection(
            object_designator, manipulator.tool_frame
        )


"""
It is a helper method that detaches the object from the robot, since the attaching withing Actions doesnt work with motions

:param world: The world in which to detach the object
:param object_designator: The object to detach
"""


def detach_object_from_hsrb(world: World, object_designator: Body):
    manipulator = world.get_semantic_annotations_by_type(ParallelGripper)[0]
    with world.modify_world():
        world.move_branch_with_fixed_connection(object_designator, world.root)


"""
Method to perceive and spawn all objects and find the object to pickup
It is a helper method that contains import error handling, since I do not want to bother about import errors

:param world: The world in which to spawn the perceived objects
:param object_name: The name of the object to pickup
"""


def try_perceiving_and_spawning_and_find_object(world: World, object_name: str):
    try:
        from demos.pycram_suturo_demos.helper_methods_and_useful_classes.object_creation import (
            perceive_and_spawn_all_objects,
        )

        perceived_objects: dict[Any, Any] = perceive_and_spawn_all_objects(world)
        logger.info(f"perceived following objects: '{perceived_objects}'")
    except ImportError:
        logger.info("Could not import robokudo")
        perceived_objects = {}
    object_to_pickup = try_get_object_to_pickup(world, object_name)
    logger.info(f"object_to_Pickup: '{object_to_pickup}'")
    return object_to_pickup

def get_nearest_object(world: World) -> Body| None:
    robot_view = world.get_semantic_annotations_by_type(HSRB)[0]
    cooking_table_annotation = world.get_semantic_annotation_by_name(
        "cooking_table"
    )
    nearest_objects_list = query_get_next_object_euclidean_x_y(
        robot_view.root, cooking_table_annotation
    ).tolist()
    object_to_pickup = nearest_objects_list[0].bodies[0]  # The nearest object
    return object_to_pickup

def get_object_with_color(world: World, color: Color) -> Body| None:
    robot_view = world.get_semantic_annotations_by_type(HSRB)[0]
    cooking_table_annotation = world.get_semantic_annotation_by_name(
        "cooking_table"
    )
    objects_on_table = query_semantic_annotations_on_surfaces([cooking_table_annotation], world).tolist()
    print()
    colored_objects = query_annotations_by_color(color, objects_on_table)
    object_to_pickup = colored_objects[0].bodies[0]  # The first colored object with that color
    return object_to_pickup

def parse_color(color_str: str) -> Color:
    """Convert a color string input to a Color object."""
    color_map = {
        "red": Color.RED(),
        "yellow": Color.YELLOW(),
        "green": Color.GREEN(),
        "cyan": Color.CYAN(),
        "blue": Color.BLUE(),
        "magenta": Color.MAGENTA(),
        "white": Color.WHITE(),
        "black": Color.BLACK(),
        "gray": Color.GRAY(),
        "grey": Color.GRAY(),
        "beige": Color.BEIGE(),
        "orange": Color.ORANGE(),
    }
    return color_map.get(color_str.strip().lower(), Color.WHITE())