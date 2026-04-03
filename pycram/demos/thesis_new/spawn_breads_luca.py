from time import sleep

import numpy as np
import rclpy

from demos.thesis_new import setup_thesis_world
from demos.thesis_new.spawn_random_breads import (
    _collect_surface_bodies,
    _tint_surfaces_light_brown,
    _spawn_bread_at_local_pose,
    _parse_stl,
    _set_uniform_scale,
)
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bread, Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color


def _spawn_bread_at_local_pose(
    world: World,
    bread_name,
    scale,
):
    bread = _parse_stl("pycram_object_gap_demo", "bread.stl")
    bread_id = bread.root.id
    bread.root.name.name = bread_name
    _set_uniform_scale(
        bread,
        (scale, scale, scale),
        color=Color(R=0.76, G=0.60, B=0.42),
    )

    world.merge_world(bread)
    bread_body = world.get_kinematic_structure_entity_by_id(bread_id)
    bread_annotation = Bread(root=bread_body)
    world.add_semantic_annotation(bread_annotation)

    return bread_annotation


def setup_random_bread_world(seed=None, robot_name=None, environment_name=None):
    world = setup_thesis_world(robot_name=robot_name, environment_name=environment_name)
    scale_choices = np.array([0.8, 1.0, 1.2, 1.4, 1.6], dtype=float)
    rng = np.random.default_rng(seed)
    surfaces = _collect_surface_bodies(world)
    if not surfaces:
        raise RuntimeError("No support surfaces found for random bread placement.")
    print(f"[spawn] evaluating {len(surfaces)} support surfaces")

    with world.modify_world():
        _tint_surfaces_light_brown(world)
        for surface_body in surfaces:
            table = Table(root=surface_body)
            world.add_semantic_annotation(table)
            table.calculate_supporting_surface()
            world.update_forward_kinematics()

            for i in range(15):

                scale = float(rng.choice(scale_choices))
                bread = _spawn_bread_at_local_pose(
                    world=world,
                    scale=scale,
                    bread_name=f"bread_{i}_{surface_body.name.name}",
                )
                world.update_forward_kinematics()

                surface_P_bread_list = table.sample_points_from_surface(
                    body_to_sample_for=bread
                )
                if not surface_P_bread_list:
                    continue
                surface_P_bread = surface_P_bread_list[0]

                yaw = float(rng.uniform(-np.pi, np.pi))
                surface_T_bread = HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=surface_P_bread.x,
                    y=surface_P_bread.y,
                    z=surface_P_bread.z,
                    yaw=yaw,
                    reference_frame=surface_P_bread.reference_frame,
                )
                world_T_bread = world.transform(surface_T_bread, world.root)
                bread.root.parent_connection.origin = world_T_bread
                table.add_object(bread)

    return world
