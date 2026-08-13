"""
A Unitree G1 supplies a wind-turbine assembly team in the AWS RoboMaker small
warehouse.

Two technicians assemble a turbine nacelle and rotor blade inside a marked assembly
zone. The robot fetches the components they need next — a bolt crate, a pitch motor
and a sensor unit — from the storage pallet stack and delivers them to the drop-off
point at the assembly station.

Needs the ``aws_robomaker_small_warehouse_world`` package built in the workspace,
since the world and its meshes are read from its share directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.api import (
    BodySpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale

# %% where everything stands in the warehouse

WORLD_URI = (
    "package://aws_robomaker_small_warehouse_world/worlds/no_roof_small_warehouse/"
    "no_roof_small_warehouse.world"
)
"""
The roofless variant of the warehouse, which can be looked into from above.
"""

SCENERY_URDF = os.path.join(os.path.dirname(__file__), "wind_turbine_assembly.urdf")
"""
The purely visual assembly-station dressing: blade, nacelle, workbench, technicians.
"""

PELVIS_HEIGHT_ABOVE_FLOOR = 0.7923
"""
How far the G1's pelvis stands above the floor with all of its leg joints at zero.

The pelvis is the robot's root, so its ``odom`` has to be lifted by this much for the
robot's feet to rest on the floor rather than sink through it.
"""

ROBOT_START_POSE = Pose.from_xyz_rpy(1.9, 9.1, PELVIS_HEIGHT_ABOVE_FLOOR)
"""
Where the robot starts, in the aisle west of the storage pallet stack.
"""

PART_SCALE = Scale(0.08, 0.08, 0.14)
"""
The extents of each transported component.
"""

STORAGE_SURFACE_HEIGHT = 0.722
"""
Top of the storage boxes on the pallet stack the components are picked from.
"""

DROP_OFF_SURFACE_HEIGHT = 0.727
"""
Top of the boxes at the assembly station's drop-off point.
"""

STANDING_DISTANCE = 0.6
"""
How far the robot stands from a pose, in meters, opposite its FRONT-facing side.

Within the G1's reach, and far enough from the pallet stack to leave its footprint
free.
"""


@dataclass
class ComponentDelivery:
    """
    One component the assembly team needs, with its storage and drop-off pose.
    """

    name: str
    """
    The component's body name, shown in the visualization's captions.
    """

    color: Color
    """
    The component's display color.
    """

    pick_pose: Pose
    """
    Where the component waits on the storage pallet stack.
    """

    place_pose: Pose
    """
    Where the component is dropped off for the assembly team.
    """


def component_pose(x: float, y: float, surface_height: float) -> Pose:
    """
    :param x: The pose's x coordinate in the world.
    :param y: The pose's y coordinate in the world.
    :param surface_height: Top of the surface the component rests on.
    :return: The pose of a component's center resting on that surface.
    """
    return Pose.from_xyz_rpy(x, y, surface_height + PART_SCALE.z / 2)


DELIVERIES = [
    ComponentDelivery(
        name="bolt_crate",
        color=Color(0.85, 0.45, 0.10),
        pick_pose=component_pose(2.75, 9.30, STORAGE_SURFACE_HEIGHT),
        place_pose=component_pose(2.60, 7.70, DROP_OFF_SURFACE_HEIGHT),
    ),
    ComponentDelivery(
        name="pitch_motor",
        color=Color(0.00, 0.55, 0.60),
        pick_pose=component_pose(2.75, 9.12, STORAGE_SURFACE_HEIGHT),
        place_pose=component_pose(2.60, 7.88, DROP_OFF_SURFACE_HEIGHT),
    ),
    ComponentDelivery(
        name="sensor_unit",
        color=Color(0.55, 0.58, 0.62),
        pick_pose=component_pose(2.75, 8.95, STORAGE_SURFACE_HEIGHT),
        place_pose=component_pose(2.60, 8.06, DROP_OFF_SURFACE_HEIGHT),
    ),
]
"""
The components the robot delivers, in the order the assembly team needs them.
"""

# %% building the world and the plan


def build_world() -> World:
    """
    :return: The warehouse with the G1, the assembly-station scenery and the
        components in it.
    """
    world = WorldSpecification.from_gazebo(
        WORLD_URI,
        robots=[
            RobotSpecification(
                semantic_annotation_type=UnitreeG1,
                world_T_odom=ROBOT_START_POSE.to_homogeneous_matrix(),
            )
        ],
        objects=[
            BodySpecification.box(
                delivery.name,
                PART_SCALE,
                color=delivery.color,
                parent_T_self=delivery.pick_pose.to_homogeneous_matrix(),
            )
            for delivery in DELIVERIES
        ],
    ).to_domain_object()
    scenery = URDFParser.from_file(
        SCENERY_URDF,
        path_resolver=FileUriResolver(base_directory=os.path.dirname(SCENERY_URDF)),
    ).parse()
    world.merge_world_at_pose(scenery, HomogeneousTransformationMatrix.from_xyz_rpy())
    return world


def standing_pose_in_front_of(pose: Pose, world: World) -> Pose:
    """
    :param pose: The pose the robot should approach from its FRONT-facing side.
    :param world: The world the pose is expressed in.
    :return: The pose the robot stands in to reach that pose with a FRONT grasp.
    """
    yaw = float(pose.yaw)
    return Pose.from_xyz_rpy(
        pose.x - STANDING_DISTANCE * np.cos(yaw),
        pose.y - STANDING_DISTANCE * np.sin(yaw),
        PELVIS_HEIGHT_ABOVE_FLOOR,
        yaw=yaw,
        reference_frame=world.root,
    )


def build_delivery_plan(
    world: World, robot: UnitreeG1, delivery: ComponentDelivery
) -> Plan:
    """
    :param world: The world the plan acts in.
    :param robot: The robot carrying out the plan.
    :param delivery: The component to fetch and where to drop it off.
    :return: The plan transporting that component from storage to the assembly
        station.
    """
    component = world.get_body_by_name(delivery.name)
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(Arms.LEFT, robot),
    )
    context = Context(world=world, robot=robot, evaluate_conditions=False)
    place_pose = Pose(
        delivery.place_pose.to_position(),
        delivery.place_pose.to_quaternion(),
        reference_frame=world.root,
    )
    torso_connection_names = [
        connection.name for connection in robot.torso.active_connections
    ]

    return sequential(
        [
            # %% fetch the component and bring it to the drop-off point
            ParkArmsAction(Arms.BOTH),
            NavigateAction(standing_pose_in_front_of(delivery.pick_pose, world)),
            PickUpAction(component, Arms.LEFT, grasp),
            ParkArmsAction(Arms.BOTH),
            MoveJointsMotion(
                names=torso_connection_names,
                positions=[0.0] * len(torso_connection_names),
            ),
            NavigateAction(Pose.from_xyz_rpy(yaw=-1.57, reference_frame=robot.root)),
            NavigateAction(standing_pose_in_front_of(delivery.place_pose, world)),
            PlaceAction(component, place_pose, Arms.LEFT),
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
    :return: The height of the robot's lowest collision geometry above the world's
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
    for delivery in DELIVERIES:
        build_delivery_plan(world, robot, delivery).perform()

for delivery in DELIVERIES:
    component_position = world.get_body_by_name(delivery.name).global_pose
    print(
        "%s delivered to %s, expected %s"
        % (
            delivery.name,
            np.round(component_position.to_position(), 3),
            np.round(delivery.place_pose.to_position(), 3),
        )
    )
    assert np.allclose(component_position, delivery.place_pose, atol=0.05)
