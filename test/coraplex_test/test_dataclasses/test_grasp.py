import os
from copy import deepcopy

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture(scope="session")
def tracy_milk_world(tracy_world):
    tracy_copy = deepcopy(tracy_world)
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "coraplex",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()
    tracy_copy.merge_world_at_pose(
        milk_world, HomogeneousTransformationMatrix.from_xyz_rpy(1, 0, 1)
    )
    with tracy_copy.modify_world():
        box = Body(
            name=PrefixedName("box"),
            collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
        )
        connection = Connection6DoF.create_with_dofs(tracy_copy, tracy_copy.root, box)
        tracy_copy.add_connection(connection)
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1, 0, 1)

    return tracy_copy, tracy_copy.get_semantic_annotations_by_type(Tracy)[0]


@pytest.fixture(scope="session")
def immutable_simple_pr2_holding_world(simple_pr2_world_setup):
    world, robot_view, context = simple_pr2_world_setup
    copy_world = deepcopy(world)
    robot_view = copy_world.get_semantic_annotation_by_id(robot_view.id)

    milk = copy_world.get_body_by_name("milk.stl")
    tcp = copy_world.get_body_by_name("l_gripper_tool_frame")
    with copy_world.modify_world():
        copy_world.move_branch(milk, tcp)
    return copy_world, robot_view, Context(copy_world, robot_view)


def test_grasp_pose_front(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose(world.get_body_by_name("milk.stl"))

    assert grasp_pose.to_quaternion().to_list() == [0, 0, 0, 1]
    assert grasp_pose.to_position().to_list() == [0, 0, 0, 1]

    offset_pose = grasp_desc.grasp_pose(
        world.get_body_by_name("milk.stl"), grasp_edge=True
    )

    assert grasp_pose.to_quaternion().to_list() == [0, 0, 0, 1]
    assert offset_pose.to_position().to_list() == pytest.approx(
        [-0.03, 0, 0, 1], abs=0.01
    )


def test_grasp_pose_right(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.RIGHT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose(world.get_body_by_name("milk.stl"))

    assert grasp_pose.to_quaternion().to_list() == pytest.approx(
        [0, 0, 0.707, 0.707], abs=0.001
    )

    offset_pose = grasp_desc.grasp_pose(
        world.get_body_by_name("milk.stl"), grasp_edge=True
    )

    assert grasp_pose.to_quaternion().to_list() == pytest.approx(
        [0, 0, 0.707, 0.707], abs=0.001
    )
    assert offset_pose.to_position().to_list() == pytest.approx(
        [-0.03, 0, 0, 1], abs=0.01
    )


def test_grasp_pose_left(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.LEFT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose(world.get_body_by_name("milk.stl"))

    assert grasp_pose.to_quaternion().to_list() == pytest.approx(
        [0, 0, -0.707, 0.707], abs=0.001
    )

    offset_pose = grasp_desc.grasp_pose(
        world.get_body_by_name("milk.stl"), grasp_edge=True
    )

    assert grasp_pose.to_quaternion().to_list() == pytest.approx(
        [0, 0, -0.707, 0.707], abs=0.001
    )
    assert offset_pose.to_position().to_list() == pytest.approx(
        [-0.03, 0, 0, 1], abs=0.01
    )


def test_grasp_pose_top(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose(world.get_body_by_name("milk.stl"))

    assert grasp_pose.to_quaternion().to_list() == pytest.approx(
        [0, 0.707, 0, 0.707], abs=0.001
    )


def test_grasp_front_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world

    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose(world.get_body_by_name("milk.stl"))

    assert grasp_pose.to_quaternion().to_list() == pytest.approx(
        [0.5, 0.5, 0.5, 0.5], abs=0.001
    )


def test_grasp_back_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world

    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.BACK,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose(world.get_body_by_name("milk.stl"))

    assert grasp_pose.to_quaternion().to_list() == pytest.approx(
        [0.5, -0.5, -0.5, 0.5], abs=0.001
    )


def test_grasp_top_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world

    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose(world.get_body_by_name("milk.stl"))

    assert grasp_pose.to_quaternion().to_list() == pytest.approx(
        [0.707, 0.707, 0.0, 0.0], abs=0.001
    )


def test_grasp_left(tracy_milk_world):
    world, robot_view = tracy_milk_world

    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.LEFT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose(world.get_body_by_name("milk.stl"))

    assert grasp_pose.to_quaternion().to_list() == pytest.approx(
        [0.707, 0.0, 0.0, 0.707], abs=0.001
    )


def test_grasp_sequence_front(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_sequence = grasp_desc.grasp_pose_sequence(world.get_body_by_name("milk.stl"))

    assert np.allclose(grasp_sequence[0].to_quaternion().to_list(), [0, 0, 0, 1])
    assert np.allclose(grasp_sequence[1].to_quaternion().to_list(), [0, 0, 0, 1])
    assert np.allclose(grasp_sequence[2].to_quaternion().to_list(), [0, 0, 0, 1])

    assert grasp_sequence[0].to_position().to_list() == pytest.approx(
        [-0.082, 0, 0, 1], abs=0.01
    )
    assert grasp_sequence[1].to_position().to_list() == pytest.approx(
        [0, 0, 0, 1], abs=0.01
    )
    assert grasp_sequence[2].to_position().to_list() == pytest.approx(
        [0, 0, 0.05, 1], abs=0.01
    )


def test_man_axis(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    assert grasp_desc.manipulation_axis() == [1, 0, 0]


def test_lift_axis(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    assert grasp_desc.lift_axis() == [0, 0, 1]


def test_lift_axis_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    assert grasp_desc.lift_axis() == [0, 1, 0]


def test_man_axis_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    assert grasp_desc.manipulation_axis() == [0, 0, 1]


def test_man_axis_tracy_right(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.RIGHT,
        VerticalAlignment.NoAlignment,
        man,
    )

    assert grasp_desc.manipulation_axis() == [0, 0, 1]


def test_grasp_sequence(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    sequence = grasp_desc.grasp_pose_sequence(
        world.get_body_by_name("milk.stl"),
    )

    assert sequence[0].to_quaternion().to_list() == pytest.approx(
        [0, 0, 0, 1], abs=0.001
    )
    assert sequence[1].to_quaternion().to_list() == pytest.approx(
        [0, 0, 0, 1], abs=0.001
    )
    assert sequence[2].to_quaternion().to_list() == pytest.approx(
        [0, 0, 0, 1], abs=0.001
    )

    assert sequence[0].to_position().to_list() == pytest.approx(
        [-0.082, 0, 0, 1], abs=0.01
    )
    assert sequence[1].to_position().to_list() == pytest.approx(
        [0, 0, 0.0, 1], abs=0.01
    )
    assert sequence[2].to_position().to_list() == pytest.approx(
        [0, 0.0, 0.05, 1], abs=0.01
    )


def test_grasp_sequence_reverse(immutable_simple_pr2_holding_world):
    world, robot_view, context = immutable_simple_pr2_holding_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    sequence = grasp_desc.place_pose_sequence(
        Pose.from_xyz_quaternion(reference_frame=world.get_body_by_name("milk.stl"))
    )

    assert sequence[2].to_quaternion().to_list() == pytest.approx(
        [0, 0, 0, 1], abs=0.001
    )
    assert sequence[1].to_quaternion().to_list() == pytest.approx(
        [0, 0, 0, 1], abs=0.001
    )
    assert sequence[0].to_quaternion().to_list() == pytest.approx(
        [0, 0, 0, 1], abs=0.001
    )

    assert sequence[2].to_position().to_list() == pytest.approx(
        [-0.082, 0, 0, 1], abs=0.01
    )
    assert sequence[1].to_position().to_list() == pytest.approx(
        [0, 0, 0.0, 1], abs=0.01
    )
    assert sequence[0].to_position().to_list() == pytest.approx(
        [0, 0.0, 0.05, 1], abs=0.01
    )


def test_grasp_sequence_front_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    sequence = grasp_desc.grasp_pose_sequence(
        world.get_body_by_name("milk.stl"),
    )

    assert sequence[0].to_quaternion().to_list() == pytest.approx(
        [0.5, 0.5, 0.5, 0.5], abs=0.001
    )
    assert sequence[1].to_quaternion().to_list() == pytest.approx(
        [0.5, 0.5, 0.5, 0.5], abs=0.001
    )
    assert sequence[2].to_quaternion().to_list() == pytest.approx(
        [0.5, 0.5, 0.5, 0.5], abs=0.001
    )

    assert sequence[0].to_position().to_list() == pytest.approx(
        [-0.082, 0, 0, 1], abs=0.01
    )
    assert sequence[1].to_position().to_list() == pytest.approx(
        [0, 0, 0.0, 1], abs=0.01
    )
    assert sequence[2].to_position().to_list() == pytest.approx(
        [0, 0.0, 0.05, 1], abs=0.01
    )


def test_grasp_sequence_right_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.RIGHT,
        VerticalAlignment.NoAlignment,
        man,
    )

    sequence = grasp_desc.grasp_pose_sequence(world.get_body_by_name("milk.stl"))

    assert sequence[0].to_quaternion().to_list() == pytest.approx(
        [0.0, 0.707, 0.707, 0.0], abs=0.001
    )
    assert sequence[1].to_quaternion().to_list() == pytest.approx(
        [0.0, 0.707, 0.707, 0.0], abs=0.001
    )
    assert sequence[2].to_quaternion().to_list() == pytest.approx(
        [0.0, 0.707, 0.707, 0.0], abs=0.001
    )

    assert sequence[0].to_position().to_list() == pytest.approx(
        [0, -0.082, 0, 1], abs=0.01
    )
    assert sequence[1].to_position().to_list() == pytest.approx(
        [0, 0, 0.0, 1], abs=0.01
    )
    assert sequence[2].to_position().to_list() == pytest.approx(
        [0, 0.0, 0.05, 1], abs=0.01
    )


def test_place_sequence(immutable_simple_pr2_holding_world):
    world, robot_view, context = immutable_simple_pr2_holding_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    sequence = grasp_desc.pose_sequence(
        Pose.from_xyz_quaternion(1, 1, 1, 0, 0, 0, 1, world.root),
        world.get_body_by_name("milk.stl"),
        reverse=True,
    )

    assert sequence[2].to_position().to_list() == pytest.approx(
        [0.9179, 1, 1, 1], abs=0.01
    )
    assert sequence[1].to_position().to_list() == pytest.approx([1, 1, 1, 1], abs=0.01)
    assert sequence[0].to_position().to_list() == pytest.approx(
        [1, 1, 1.05, 1], abs=0.01
    )


def test_place_sequence_right_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.RIGHT,
        VerticalAlignment.NoAlignment,
        man,
    )

    sequence = grasp_desc.pose_sequence(
        Pose.from_xyz_quaternion(1, 1, 1, 0, 0, 0, 1, world.root),
        world.get_body_by_name("milk.stl"),
        reverse=True,
    )

    assert sequence[0].reference_frame == world.root

    assert sequence[0].to_position().to_list() == pytest.approx(
        [1, 1, 1.05, 1], abs=0.01
    )
    assert sequence[1].to_position().to_list() == pytest.approx([1, 1, 1, 1], abs=0.01)
    assert sequence[2].to_position().to_list() == pytest.approx(
        [1, 0.918, 1.0, 1], abs=0.01
    )

    assert sequence[0].to_quaternion().to_list() == pytest.approx(
        [0.0, 0.707, 0.707, 0.0], abs=0.001
    )
    assert sequence[1].to_quaternion().to_list() == pytest.approx(
        [0.0, 0.707, 0.707, 0.0], abs=0.001
    )
    assert sequence[2].to_quaternion().to_list() == pytest.approx(
        [0.0, 0.707, 0.707, 0.0], abs=0.001
    )


def test_pose_sequence_top(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    man = robot_view.left_arm.end_effector

    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        man,
    )

    sequence = grasp_desc.grasp_pose_sequence(world.get_body_by_name("milk.stl"))

    assert sequence[0].reference_frame == world.get_body_by_name("milk.stl")

    assert sequence[0].to_position().to_list() == pytest.approx(
        [0, 0, 0.083, 1], abs=0.01
    )
    assert sequence[1].to_position().to_list() == pytest.approx([0, 0, 0, 1], abs=0.01)
    assert sequence[2].to_position().to_list() == pytest.approx(
        [0, 0.0, 0.05, 1], abs=0.01
    )

    assert sequence[0].to_quaternion().to_list() == pytest.approx(
        [0, 0.707, 0, 0.707], abs=0.001
    )
    assert sequence[1].to_quaternion().to_list() == pytest.approx(
        [0, 0.707, 0, 0.707], abs=0.001
    )
    assert sequence[2].to_quaternion().to_list() == pytest.approx(
        [0, 0.707, 0, 0.707], abs=0.001
    )


def test_pose_sequence_top_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.end_effector
    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        man,
    )
    sequence = grasp_desc.grasp_pose_sequence(world.get_body_by_name("milk.stl"))

    assert sequence[0].reference_frame == world.get_body_by_name("milk.stl")

    assert sequence[0].to_position().to_list() == pytest.approx(
        [0, 0, 0.083, 1], abs=0.01
    )
    assert sequence[1].to_position().to_list() == pytest.approx([0, 0, 0, 1], abs=0.01)
    assert sequence[2].to_position().to_list() == pytest.approx(
        [0, 0.0, 0.05, 1], abs=0.01
    )

    assert sequence[0].to_quaternion().to_list() == pytest.approx(
        [0.707, 0.707, 0.0, 0.0], abs=0.001
    )
    assert sequence[1].to_quaternion().to_list() == pytest.approx(
        [0.707, 0.707, 0.0, 0.0], abs=0.001
    )
    assert sequence[2].to_quaternion().to_list() == pytest.approx(
        [0.707, 0.707, 0.0, 0.0], abs=0.001
    )


def test_pose_sequence_top_tracy_box(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.end_effector
    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        man,
    )
    sequence = grasp_desc.pose_sequence(
        Pose.from_xyz_quaternion(1, 0, 1, reference_frame=world.root),
        world.get_body_by_name("box"),
    )

    assert sequence[0].reference_frame == world.root

    assert sequence[0].to_position().to_list() == pytest.approx(
        [1, 0, 1.1, 1], abs=0.01
    )
    assert sequence[1].to_position().to_list() == pytest.approx([1, 0, 1, 1], abs=0.01)
    assert sequence[2].to_position().to_list() == pytest.approx(
        [1, 0.0, 1.05, 1], abs=0.01
    )

    assert sequence[0].to_quaternion().to_list() == pytest.approx(
        [0.707, 0.707, 0.0, 0.0], abs=0.001
    )
    assert sequence[1].to_quaternion().to_list() == pytest.approx(
        [0.707, 0.707, 0.0, 0.0], abs=0.001
    )
    assert sequence[2].to_quaternion().to_list() == pytest.approx(
        [0.707, 0.707, 0.0, 0.0], abs=0.001
    )


# %% robot-relative approach directions


def test_approach_direction_object_frame_translation():
    assert (
        ApproachDirection.FRONT.to_object_frame(ApproachDirection.BACK)
        == ApproachDirection.BACK
    )
    assert (
        ApproachDirection.LEFT.to_object_frame(ApproachDirection.LEFT)
        == ApproachDirection.BACK
    )
    assert (
        ApproachDirection.RIGHT.to_object_frame(ApproachDirection.FRONT)
        == ApproachDirection.RIGHT
    )
    assert (
        ApproachDirection.BACK.to_robot_relative(ApproachDirection.BACK)
        == ApproachDirection.FRONT
    )
    assert (
        ApproachDirection.FRONT.to_object_frame(
            ApproachDirection.BACK
        ).to_robot_relative(ApproachDirection.BACK)
        == ApproachDirection.FRONT
    )


def test_resolve_approach_direction_for_rotated_target(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.end_effector
    rotated_target = Pose.from_xyz_quaternion(
        1, 0, 1, 0, 0, 1, 0, reference_frame=world.root
    )

    front_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )
    assert (
        front_description.resolve_approach_direction(rotated_target)
        == ApproachDirection.BACK
    )

    left_description = GraspDescription(
        ApproachDirection.LEFT,
        VerticalAlignment.NoAlignment,
        man,
    )
    assert (
        left_description.resolve_approach_direction(rotated_target)
        == ApproachDirection.RIGHT
    )


def test_resolve_approach_direction_with_explicit_robot_pose(
    immutable_simple_pr2_world,
):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.end_effector
    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    target = Pose.from_xyz_quaternion(1, 0, 1, reference_frame=world.root)
    standing_pose_north_of_target = Pose.from_xyz_quaternion(
        1, 2, 0, reference_frame=world.root
    )

    assert (
        grasp_desc.resolve_approach_direction(
            target, robot_pose=standing_pose_north_of_target
        )
        == ApproachDirection.LEFT
    )


def test_pose_sequence_front_faces_robot_regardless_of_target_yaw(
    immutable_simple_pr2_world,
):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.end_effector
    grasp_desc = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )
    sequence = grasp_desc.pose_sequence(
        Pose.from_xyz_quaternion(1, 0, 1, 0, 0, 1, 0, reference_frame=world.root),
        world.get_body_by_name("milk.stl"),
    )

    assert sequence[0].reference_frame == world.root

    assert sequence[0].to_quaternion().to_list() == pytest.approx(
        [0, 0, 0, 1], abs=0.001
    )
    assert sequence[1].to_quaternion().to_list() == pytest.approx(
        [0, 0, 0, 1], abs=0.001
    )
    assert sequence[2].to_quaternion().to_list() == pytest.approx(
        [0, 0, 0, 1], abs=0.001
    )

    assert sequence[0].to_position().to_list() == pytest.approx(
        [0.917, 0, 1, 1], abs=0.001
    )
    assert sequence[1].to_position().to_list() == pytest.approx(
        [1.0, 0, 1, 1], abs=0.001
    )
    assert sequence[2].to_position().to_list() == pytest.approx(
        [1.0, 0, 1.05, 1], abs=0.001
    )


def test_pose_sequence_left_of_rotated_target_is_robots_left(
    immutable_simple_pr2_world,
):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.end_effector
    grasp_desc = GraspDescription(
        ApproachDirection.LEFT,
        VerticalAlignment.NoAlignment,
        man,
    )
    sequence = grasp_desc.pose_sequence(
        Pose.from_xyz_quaternion(1, 0, 1, 0, 0, 1, 0, reference_frame=world.root),
        world.get_body_by_name("milk.stl"),
    )

    assert sequence[0].to_quaternion().to_list() == pytest.approx(
        [0, 0, -0.707, 0.707], abs=0.001
    )
    assert sequence[1].to_quaternion().to_list() == pytest.approx(
        [0, 0, -0.707, 0.707], abs=0.001
    )

    assert sequence[0].to_position().to_list() == pytest.approx(
        [1, 0.082, 1, 1], abs=0.01
    )
    assert sequence[1].to_position().to_list() == pytest.approx([1, 0, 1, 1], abs=0.01)


def test_calculate_grasp_descriptions_labels_are_robot_relative(
    immutable_simple_pr2_world,
):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.end_effector
    rotated_target = Pose.from_xyz_quaternion(
        1, 0, 1, 0, 0, 1, 0, reference_frame=world.root
    )

    descriptions = GraspDescription.calculate_grasp_descriptions(man, rotated_target)

    assert descriptions[0].approach_direction == ApproachDirection.FRONT
    assert (
        descriptions[0].resolve_approach_direction(rotated_target)
        == ApproachDirection.BACK
    )
