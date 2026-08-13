"""
A Unitree G1 prepares a wind turbine for service.

The demo plays at the foot of the 300 m turbine at the centre of a 25-machine wind farm.
The robot walks down the pad towards the service area while the turbine is shut down
around it: the nacelle yaws out of the wind and the rotor turns into its parked
position, one blade straight down. It then lays the delivered service parts out on the
bench at the tower door, taking them one by one off the trailer they arrived on.

The wind farm is a MuJoCo scene with no collision geometry and nothing at human scale,
so the trailer and the bench are equipment the demo brings with it;
:mod:`service_layout` records where they stand and :mod:`service_tasks` derives the
poses from it.
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
from semantic_digital_twin.api import (
    BodySpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

from service_layout import (
    PAD_SURFACE_HEIGHT,
    ROTOR_CONNECTION,
    SERVICE_SURFACES,
    YAW_CONNECTION,
)
from service_tasks import (
    APPROACH_WAYPOINTS,
    PART_SCALE,
    ROBOT_START_POSE,
    SERVICE_TRANSFERS,
    ServiceTransfer,
    turbine_state,
)

# %% the wind farm

WIND_FARM_MJCF = os.path.join(os.path.dirname(__file__), "wind_farm.xml")
"""
The wind farm the demo plays in.
"""

# %% building the world


def build_world() -> World:
    """
    :return: The wind farm with the G1, the service equipment and the parts in it.
    """
    return WorldSpecification.from_mjcf(
        WIND_FARM_MJCF,
        robots=[
            RobotSpecification(
                semantic_annotation_type=UnitreeG1,
                world_T_odom=ROBOT_START_POSE.to_homogeneous_matrix(),
            )
        ],
        objects=[
            BodySpecification.box(
                surface.name,
                surface.scale,
                color=surface.color,
                parent_T_self=Pose.from_xyz_rpy(
                    *surface.box_center
                ).to_homogeneous_matrix(),
            )
            for surface in SERVICE_SURFACES
        ]
        + [
            BodySpecification.box(
                transfer.name,
                PART_SCALE,
                color=transfer.color,
                parent_T_self=transfer.trailer_pose.to_homogeneous_matrix(),
            )
            for transfer in SERVICE_TRANSFERS
        ],
    ).to_domain_object()


def in_world(pose: Pose, world: World) -> Pose:
    """
    :param pose: A pose from :mod:`service_tasks`, which carries no reference frame.
    :param world: The world to express it in.
    :return: The same pose, anchored to the world's root.
    """
    return Pose(pose.to_position(), pose.to_quaternion(), reference_frame=world.root)


# %% shutting the turbine down


def shut_turbine_down(world: World, progress: float) -> None:
    """
    Command the turbine a step further towards its parked position.

    :param world: The world holding the turbine.
    :param progress: How far the shutdown has run, from ``0.0`` to ``1.0``.
    """
    angles = turbine_state(progress)
    world.set_positions_1DOF_connection(
        {
            world.get_connection_by_name(YAW_CONNECTION): angles["yaw"],
            world.get_connection_by_name(ROTOR_CONNECTION): angles["rotor"],
        }
    )


# %% the transfer plan


def build_transfer_plan(
    world: World, robot: UnitreeG1, transfer: ServiceTransfer
) -> Plan:
    """
    :param world: The world the plan acts in.
    :param robot: The robot carrying out the plan.
    :param transfer: The part to take off the trailer and where to lay it out.
    :return: The plan moving that part from the trailer onto the bench.
    """
    part = world.get_body_by_name(transfer.name)
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
            # %% take the part off the delivery trailer
            ParkArmsAction(Arms.BOTH),
            NavigateAction(in_world(transfer.trailer_standing_pose, world)),
            PickUpAction(part, Arms.LEFT, grasp),
            ParkArmsAction(Arms.BOTH),
            MoveJointsMotion(
                names=torso_connection_names,
                positions=[0.0] * len(torso_connection_names),
            ),
            # %% turn to the tower door and lay the part out on the bench
            NavigateAction(
                Pose.from_xyz_rpy(
                    yaw=transfer.turn_towards_bench, reference_frame=robot.root
                )
            ),
            NavigateAction(in_world(transfer.bench_standing_pose, world)),
            PlaceAction(part, in_world(transfer.bench_pose, world), Arms.LEFT),
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
    :return: The height of the robot's lowest collision geometry above the world floor.
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
# Keeps STANDING_HEIGHT honest: the robot has to stand on the turbine's pad rather than
# sink into it or hover above it.
assert abs(lowest_collision_point_of(robot, world) - PAD_SURFACE_HEIGHT) < 1e-3

start_visualization(world)

context = Context(world=world, robot=robot, evaluate_conditions=False)
with simulated_robot:
    for leg, waypoint in enumerate(APPROACH_WAYPOINTS, start=1):
        sequential(
            [NavigateAction(in_world(waypoint, world))], context=context
        ).plan.perform()
        shut_turbine_down(world, leg / len(APPROACH_WAYPOINTS))

    for transfer in SERVICE_TRANSFERS:
        build_transfer_plan(world, robot, transfer).perform()

for transfer in SERVICE_TRANSFERS:
    part_position = world.get_body_by_name(transfer.name).global_pose
    print(
        "%s laid out at %s, expected %s"
        % (
            transfer.name,
            np.round(part_position.to_position(), 3),
            np.round(transfer.bench_pose.to_position(), 3),
        )
    )
    assert np.allclose(part_position, transfer.bench_pose, atol=0.05)
