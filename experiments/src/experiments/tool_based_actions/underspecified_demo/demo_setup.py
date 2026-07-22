"""
Shared construction blocks of the underspecified tool-action demos.

All four demos navigate to a base pose sampled from the same region in front of the
kitchen counter and spawn their manipulated object at the same counter position, so both
are defined once here.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.factories import a
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.robot_plans.actions.core.navigation import NavigateAction

from experiments.tool_based_actions.simple_demo.demo_world import (
    TARGET_POSITION_XYZ,
    parse_object,
)


@dataclass(frozen=True)
class SampledRegion:
    """
    An axis-aligned XY rectangle the probabilistic backend samples poses from.
    """

    minimum_x: float
    """
    Lower X bound of the region in the world frame.
    """

    maximum_x: float
    """
    Upper X bound of the region in the world frame.
    """

    minimum_y: float
    """
    Lower Y bound of the region in the world frame.
    """

    maximum_y: float
    """
    Upper Y bound of the region in the world frame.
    """


BASE_POSE_REGION = SampledRegion(
    minimum_x=1.7, maximum_x=1.95, minimum_y=2.1, maximum_y=2.35
)
"""
Region in front of the kitchen counter the robot's base pose is sampled from.
"""


def build_underspecified_navigation(world: World) -> Match[NavigateAction]:
    """
    Build a navigation whose base pose is a free variable bounded to
    :data:`BASE_POSE_REGION`.

    :param world: The world the base pose is expressed in.
    :return: The underspecified navigation.
    """
    navigate = a(NavigateAction)(
        target_location=a(Pose.from_xyz_rpy)(
            x=...,
            y=...,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            reference_frame=world.root,
        ),
    )
    navigate.where(
        navigate.variable.target_location.x > BASE_POSE_REGION.minimum_x,
        navigate.variable.target_location.x < BASE_POSE_REGION.maximum_x,
        navigate.variable.target_location.y > BASE_POSE_REGION.minimum_y,
        navigate.variable.target_location.y < BASE_POSE_REGION.maximum_y,
    )
    return navigate


def place_target_on_counter(world: World, mesh_file_name: str, color: Color) -> Body:
    """
    Spawn a mesh object at the counter target position.

    :param world: The world to spawn into.
    :param mesh_file_name: Mesh file in the demo resources; also the spawned body's
        name.
    :param color: Color the mesh's visual shapes are dyed with.
    :return: The spawned body inside ``world``.
    """
    object_world = parse_object(mesh_file_name, color=color)
    with world.modify_world():
        world.merge_world_at_pose(
            object_world,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                *TARGET_POSITION_XYZ, reference_frame=world.root
            ),
        )
    return world.get_body_by_name(mesh_file_name)
