"""
Validate fixed-heading workshop corridors against native collision geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from coraplex.locations.navigation import NavigationPath
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import TaskStatus
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.plans.factories import execute_single
from cramera.workshop_navigation import (
    WorkshopTranslation,
    WorkshopTranslationBlocked,
    WorkshopTranslateMotion,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale


# %% private geometry sampling
@dataclass(eq=False)
class PreviewStateChanges(StateChangeCallback):
    """
    Count any state callback leaked by a hypothetical corridor check.
    """

    count: int = 0
    """
    Number of native state notifications observed.
    """

    def on_state_change(self, **kwargs) -> None:
        """
        Register a published state transition.
        """
        self.count += 1


def translation(world: World, target: Pose) -> WorkshopTranslation:
    """
    Use explicit external collision rules for the small native robot fixture.
    """
    [robot] = world.get_semantic_annotations_by_type(AbstractRobot)
    with world.modify_world():
        world.collision_manager.add_default_rule(
            AvoidCollisionBetweenGroups(
                body_group_a=robot.bodies_with_collision,
                body_group_b=[
                    body
                    for body in world.bodies_with_collision
                    if body not in robot.bodies
                ],
            )
        )
    return WorkshopTranslation(world=world, robot=robot, target=target)


def test_empty_corner_of_robot_projection_is_a_clear_corridor(
    cylinder_bot_world: World,
) -> None:
    """
    Separate raised parts do not occupy their combined bounding rectangle.
    """
    world = cylinder_bot_world
    [robot] = world.get_semantic_annotations_by_type(AbstractRobot)
    for name, x, y in (("left_payload", -0.6, 0), ("forward_payload", 0, 0.6)):
        body = BodySpecification.box(
            name,
            Scale(0.2, 0.2, 0.2),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x, y, 0.5),
        ).spawn(world)
        with world.modify_world():
            world.move_branch_with_fixed_connection(body, robot.root)
    BodySpecification.box(
        "corner_post",
        Scale(0.03, 0.03, 1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(-0.6, 0.6, 0.5),
    ).spawn(world)
    target = Pose.from_xyz_rpy(-0.2, -0.2, reference_frame=world.root)
    route = translation(world, target)
    preview = PreviewStateChanges(_world=world)
    original_state = world.state._data.copy()
    original_pose = robot.root.global_pose.to_np().copy()
    projected = NavigationPath(world, robot, target, clearance=0).translation_obstacles(
        robot.root.global_pose, robot.as_bounding_box_collection_in_frame(world.root)
    )
    assert any(
        box.min_x <= 0 <= box.max_x and box.min_y <= 0 <= box.max_y for box in projected
    )

    result = route.validate()

    assert result.samples > 1
    assert result.sample_spacing <= route.maximum_sample_spacing
    assert result.minimum_clearance >= result.sample_spacing / 2
    np.testing.assert_array_equal(world.state._data, original_state)
    np.testing.assert_array_equal(robot.root.global_pose.to_np(), original_pose)
    assert preview.count == 0


def test_a_thin_obstacle_between_endpoints_blocks_translation(
    cylinder_bot_world: World,
) -> None:
    """
    Sampling must reject an obstacle even though both endpoint poses are clear.
    """
    world = cylinder_bot_world
    BodySpecification.box(
        "aisle_obstruction",
        Scale(0.001, 10, 1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(-0.1),
    ).spawn(world)
    route = translation(world, Pose.from_xyz_rpy(-0.2, reference_frame=world.root))
    preview = PreviewStateChanges(_world=world)
    original = world.state._data.copy()
    with pytest.raises(WorkshopTranslationBlocked):
        route.validate()
    np.testing.assert_array_equal(world.state._data, original)
    assert preview.count == 0


def test_intentional_floor_contact_uses_the_current_collision_policy(
    cylinder_bot_world: World,
) -> None:
    """
    An explicitly allowed support remains allowed without ignoring furniture.
    """
    world = cylinder_bot_world
    [robot] = world.get_semantic_annotations_by_type(AbstractRobot)
    floor = BodySpecification.box(
        "allowed_support",
        Scale(10, 10, 0.02),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-0.25),
    ).spawn(world)
    route = translation(world, Pose.from_xyz_rpy(-0.2, reference_frame=world.root))
    with world.modify_world():
        world.collision_manager.add_ignore_collision_rule(
            AllowCollisionBetweenGroups(
                body_group_a=robot.bodies_with_collision, body_group_b=[floor]
            )
        )
    assert route.validate().samples > 1


@pytest.mark.parametrize("pose", [Pose.from_xyz_rpy(yaw=0.1), Pose.from_xyz_rpy(z=0.1)])
def test_heading_and_height_changes_are_outside_the_corridor_contract(
    cylinder_bot_world: World, pose: Pose
) -> None:
    """
    A translation-only clearance certificate cannot authorize a turning sweep.
    """
    pose.reference_frame = cylinder_bot_world.root
    route = translation(cylinder_bot_world, pose)
    with pytest.raises(WorkshopTranslationBlocked):
        route.validate()


@pytest.mark.parametrize("initial_heading", [0.0, 1e-5])
def test_clear_corridor_executes_with_the_native_base_controller(
    cylinder_bot_world: World,
    initial_heading: float,
) -> None:
    """
    A certified segment reaches its goal through simulated robot motion ticks.
    """
    world = cylinder_bot_world
    target = Pose.from_xyz_rpy(-0.12, reference_frame=world.root)
    route = translation(world, target)
    route.robot.set_root_pose(
        Pose.from_xyz_rpy(yaw=initial_heading, reference_frame=world.root)
    )
    expected = route.robot.root.global_pose.to_np().copy()
    expected[0, 3] = target.x
    node = execute_single(
        WorkshopTranslateMotion(target), context=Context.from_world(world)
    )
    changes = PreviewStateChanges(_world=world)
    with simulated_robot_advanced:
        node.plan.perform()
    assert node.status == TaskStatus.SUCCEEDED
    np.testing.assert_allclose(
        route.robot.root.global_pose.to_np(), expected, atol=0.001
    )
    assert changes.count > 1


def test_translation_uses_heading_measured_after_plan_parsing(
    cylinder_bot_world: World,
) -> None:
    """
    Prior manipulation drift must not change a translation into a turn request.
    """
    world = cylinder_bot_world
    target = Pose.from_xyz_rpy(-0.12, reference_frame=world.root)
    route = translation(world, target)
    motion = WorkshopTranslateMotion(target)
    execute_single(motion, context=Context.from_world(world))
    goal = motion.motion_chart
    chart = MotionStatechart()
    chart.add_node(goal)
    route.robot.set_root_pose(Pose.from_xyz_rpy(yaw=1e-5, reference_frame=world.root))
    expected = route.robot.root.global_pose.to_np().copy()
    expected[0, 3] = target.x

    goal.expand(MotionStatechartContext(world=world))

    np.testing.assert_allclose(goal.translation.world_target().to_np(), expected)
