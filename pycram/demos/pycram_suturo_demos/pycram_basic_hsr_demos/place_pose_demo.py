import logging

import semantic_digital_twin
from pycram_suturo_demos.helper_methods_and_useful_classes.object_creation import (
    move_object_to_new_pose,
)
from demos.pycram_suturo_demos.helper_methods_and_useful_classes.A_robot_setup import (
    robot_setup,
)
from pycram.motion_executor import simulated_robot, real_robot
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3
from semantic_digital_twin.world_description.geometry import Scale

SIMULATED = True
"""
Set this flag to True to run the demo in a simulated environment, 
or False to run it on the real robot.
"""

logging.getLogger(semantic_digital_twin.world.__name__).setLevel(logging.WARN)
logger = logging.getLogger(__name__)


def simulation_demo():
    setup_result = robot_setup(simulation=True, with_simulated_objects=False)
    world, robot_view, context = (
        setup_result.world,
        setup_result.robot_view,
        setup_result.context,
    )

    desk = world.get_semantic_annotation_by_name("desk")

    with world.modify_world():
        milk = Milk.create_with_new_body_in_world(
            name=PrefixedName(f"milk_carton"),
            world=world,
            scale=Scale(0.1, 0.1, 0.2),
        )

    points = desk.sample_points_from_surface(milk)
    point = points[0] if points else Point3()
    pose = Pose(position=point, reference_frame=point.reference_frame)

    with world.modify_world():
        move_object_to_new_pose(milk, pose.to_homogeneous_matrix())
        desk.add_object(milk)

    for i in range(0, 10):
        with world.modify_world():
            obj = Milk.create_with_new_body_in_world(
                name=PrefixedName(f"milk_carton_{i}"),
                world=world,
                scale=Scale(0.1, 0.1, 0.2),
            )
        points = desk.sample_points_from_surface(obj)
        point = points[0] if points else Point3()
        pose = Pose(position=point, reference_frame=point.reference_frame)

        with world.modify_world():
            move_object_to_new_pose(obj, pose.to_homogeneous_matrix())
            desk.add_object(obj)


def real_demo():
    # TODO: Implement
    pass


if __name__ == "__main__":
    if SIMULATED:
        with simulated_robot:
            simulation_demo()
    else:
        with real_robot:
            real_demo()
