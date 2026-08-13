"""
A Unitree G1 clears an incoming pallet in a storage warehouse.

A pallet of three crates has been dropped in the working aisle between the east centre
rack and the east wall rack. The robot does what a warehouse robot does with a receiving
pallet: it takes the crates off the pallet load one by one, turns to the rack behind it
and stows each one on the free stretch of the wall rack's shelf.

Both racks are full, so the one shelf the crates fit on is also the only one within the
robot's reach. :mod:`warehouse_layout` records where everything is and :mod:`stow_tasks`
derives the poses from it.
"""

from __future__ import annotations

import os

import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans import MoveJointsMotion
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.testing import start_visualization
from coraplex.view_manager import ViewManager
from semantic_digital_twin.adapters.package_resolver import FileUriResolver
from semantic_digital_twin.api import (
    BodySpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

from stow_tasks import CRATE_SCALE, ROBOT_START_POSE, STOW_TASKS, StowTask

# %% the warehouse

WAREHOUSE_URDF = os.path.join(os.path.dirname(__file__), "storage_warehouse.urdf")
"""
The storage warehouse the demo plays in.
"""

# %% building the world and the plan


def build_world() -> World:
    """
    :return: The warehouse with the G1 and the incoming crates in it.
    """
    return WorldSpecification.from_urdf(
        WAREHOUSE_URDF,
        path_resolver=FileUriResolver(base_directory=os.path.dirname(WAREHOUSE_URDF)),
        robots=[
            RobotSpecification(
                semantic_annotation_type=UnitreeG1,
                world_T_odom=ROBOT_START_POSE.to_homogeneous_matrix(),
            )
        ],
        objects=[
            BodySpecification.box(
                task.name,
                CRATE_SCALE,
                color=task.color,
                parent_T_self=task.pick_pose.to_homogeneous_matrix(),
            )
            for task in STOW_TASKS
        ],
    ).to_domain_object()


def in_world(pose: Pose, world: World) -> Pose:
    """
    :param pose: A pose from :mod:`stow_tasks`, which carries no reference frame.
    :param world: The world to express it in.
    :return: The same pose, anchored to the world's root.
    """
    return Pose(pose.to_position(), pose.to_quaternion(), reference_frame=world.root)


def build_stow_plan(world: World, robot: UnitreeG1, task: StowTask) -> Plan:
    """
    :param world: The world the plan acts in.
    :param robot: The robot carrying out the plan.
    :param task: The crate to take off the pallet and where to stow it.
    :return: The plan moving that crate from the pallet onto the shelf.
    """
    crate = world.get_body_by_name(task.name)
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(Arms.LEFT, robot),
    )
    context = Context(world=world, robot=robot, evaluate_conditions=False)
    torso_connection_names = [
        connection.name for connection in robot.torso.active_connections
    ]

    return sequential(
        [
            # %% take the crate off the pallet
            ParkArmsAction(Arms.BOTH),
            NavigateAction(in_world(task.pallet_standing_pose, world)),
            PickUpAction(crate, Arms.LEFT, grasp),
            ParkArmsAction(Arms.BOTH),
            MoveJointsMotion(
                names=torso_connection_names,
                positions=[0.0] * len(torso_connection_names),
            ),
            # %% turn to the rack behind and stow the crate on the shelf
            NavigateAction(
                Pose.from_xyz_rpy(
                    yaw=task.turn_towards_shelf, reference_frame=robot.root
                )
            ),
            NavigateAction(in_world(task.shelf_standing_pose, world)),
            PlaceAction(crate, in_world(task.shelf_pose, world), Arms.LEFT),
            ParkArmsAction(Arms.BOTH),
            MoveJointsMotion(
                names=torso_connection_names,
                positions=[0.0] * len(torso_connection_names),
            ),
        ],
        context=context,
    ).plan


def lowest_collision_point_of(robot: UnitreeG1, world: World) -> float:
    """
    :param robot: The robot to measure.
    :param world: The world the height is expressed in.
    :return: The height of the robot's lowest collision geometry above the warehouse
        floor.
    """
    return min(
        body.collision.as_bounding_box_collection_in_frame(world.root)
        .bounding_box()
        .min_z
        for body in world.get_kinematic_structure_entities_of_branch(robot.root)
        if body.collision
    )


# %% running the demo

world = build_world()
robot = world.get_semantic_annotations_by_type(UnitreeG1)[0]
# Keeps PELVIS_HEIGHT_ABOVE_FLOOR honest: the robot has to stand on the floor rather
# than sink into it or hover above it.
assert abs(lowest_collision_point_of(robot, world)) < 1e-3

start_visualization(world)

with simulated_robot:
    for task in STOW_TASKS:
        build_stow_plan(world, robot, task).perform()

for task in STOW_TASKS:
    crate_position = world.get_body_by_name(task.name).global_pose
    print(
        "%s stowed at %s, expected %s"
        % (
            task.name,
            np.round(crate_position.to_position(), 3),
            np.round(task.shelf_pose.to_position(), 3),
        )
    )
    assert np.allclose(crate_position, task.shelf_pose, atol=0.05)
