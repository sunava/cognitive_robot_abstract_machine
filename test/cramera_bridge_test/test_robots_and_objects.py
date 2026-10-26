"""
The robots and free objects of a world.
"""

from __future__ import annotations

import pytest

from cramera_bridge.exceptions import UnknownArm, UnknownRobot
from cramera_bridge.robots import RobotParts, RobotRoster
from cramera_bridge.world_objects import WorldObjects
from cramera_protocol.bundle.scene import PartRole, PartSide
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Milk,
)

# %% robots


def test_a_robot_is_identified_by_its_annotation(robot_world):
    roster = RobotRoster(robot_world.world)
    identifier = RobotRoster.identifier(robot_world.robot)
    assert identifier == str(robot_world.robot.id)
    assert roster.named(identifier) is robot_world.robot


def test_an_unknown_robot_is_refused(robot_world):
    with pytest.raises(UnknownRobot):
        RobotRoster(robot_world.world).named("nobody")


def test_a_single_robot_model_is_named_after_its_class(robot_world):
    assert RobotRoster(robot_world.world).model_name(robot_world.robot) == "pr2"


def test_the_instance_lists_the_movable_joints(robot_world):
    instance = RobotRoster(robot_world.world).instance(robot_world.robot)
    assert instance.model == type(robot_world.robot).__name__
    assert "pr2/torso_lift_joint" in instance.joint_positions


def test_arms_and_end_effectors_come_from_the_annotations(robot_world):
    parts = RobotParts(robot_world.world, robot_world.robot)
    descriptions = parts.descriptions()
    arms = [part for part in descriptions if part.role is PartRole.ARM]
    grippers = [part for part in descriptions if part.role is PartRole.END_EFFECTOR]
    assert {arm.side for arm in arms} == {PartSide.LEFT, PartSide.RIGHT}
    assert {gripper.attached_to for gripper in grippers} == {arm.name for arm in arms}
    assert not set(arms[0].links) & set(grippers[0].links)


def test_an_arm_is_found_by_name(robot_world):
    parts = RobotParts(robot_world.world, robot_world.robot)
    left = robot_world.robot.get_left_arm_if_specified()
    assert parts.arm_named(str(left.name)) is left
    with pytest.raises(UnknownArm):
        parts.arm_named("tail")


# %% objects


def test_free_bodies_are_objects_whatever_their_shapes(kitchen):
    objects = WorldObjects(kitchen.world).objects()
    assert objects == [kitchen.milk, kitchen.cup, kitchen.frame]


def test_a_tracked_object_stays_an_object_when_fixed(kitchen):
    world = kitchen.world
    with world.modify_world():
        world.remove_connection(kitchen.milk_connection)
        world.add_connection(
            type(kitchen.table.parent_connection)(
                parent=kitchen.table, child=kitchen.milk
            )
        )
    assert kitchen.milk not in WorldObjects(world).objects()
    assert kitchen.milk in WorldObjects(world).objects(tracked=[kitchen.milk.id])


def fix_to_table(kitchen) -> None:
    """
    Attach the milk to the table with a fixed connection, as a demo that places its
    objects on the furniture does.
    """
    world = kitchen.world
    with world.modify_world():
        world.remove_connection(kitchen.milk_connection)
        world.add_connection(
            type(kitchen.table.parent_connection)(
                parent=kitchen.table, child=kitchen.milk
            )
        )


def test_a_fixed_body_annotated_as_something_to_take_hold_of_is_an_object(kitchen):
    fix_to_table(kitchen)
    with kitchen.world.modify_world():
        kitchen.world.add_semantic_annotation(Milk(root=kitchen.milk))
    assert kitchen.milk in WorldObjects(kitchen.world).objects()


def test_a_part_of_something_else_is_no_object_though_it_offers_grasps(kitchen):
    fix_to_table(kitchen)
    with kitchen.world.modify_world():
        kitchen.world.add_semantic_annotation(Handle(root=kitchen.milk))
    assert kitchen.milk not in WorldObjects(kitchen.world).objects()


def test_the_robot_and_its_odometry_are_no_objects(robot_world):
    objects = WorldObjects(robot_world.world).objects()
    assert not {body.id for body in objects} & {
        body.id for body in robot_world.robot.bodies
    }
