"""
Tests for which bodies a detector watches.
"""

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import TranslationEvent
from segmind.detectors.atomic_event_detectors_nodes import TranslationDetector
from segmind.detectors.base import SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% a world with a movable gripper and a loose object


def collidable_body(name: str) -> Body:
    """
    A body a detector can measure, and so produce events about.

    :param name: The body's name.
    """
    return Body(
        name=PrefixedName(name),
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
    )


def graspable_world() -> World:
    """
    A world holding a gripper that can be driven and a loose object beside it, both free
    to move the way a demo's own objects are.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    gripper, cup = collidable_body("gripper"), collidable_body("cup")
    with world.modify_world():
        world.add_body(root)
        for body in (gripper, cup):
            connection = Connection6DoF.create_with_dofs(
                world=world, parent=root, child=body
            )
            world.add_connection(connection)
    return world


class TestAGraspedBodyStaysWatched:
    """
    A plan that picks something up re-parents it to the gripper with a fixed connection,
    so the body stops being one of the world's free bodies.

    It is exactly then that it is
    worth watching: what it does while it is carried is what a pick-up is made of.
    """

    def grasp(self, world: World, body: Body, gripper: Body) -> None:
        """
        Screw ``body`` onto ``gripper`` the way a plan does when it grasps something.

        :param world: The world both bodies live in.
        :param body: The body being grasped.
        :param gripper: The body it is attached to.
        """
        held = world.compute_forward_kinematics(gripper, body)
        with world.modify_world():
            world.remove_connection(body.parent_connection)
            world.add_connection(
                FixedConnection(
                    parent=gripper, child=body, parent_T_connection_expression=held
                )
            )

    def translations_of(self, segmind_context: SegmindContext, body: Body) -> list:
        """
        Every translation detected for one body.

        :param segmind_context: The context the detectors logged their events to.
        :param body: The body the translations are asked about.
        """
        return [
            event
            for event in segmind_context.logger.get_events()
            if isinstance(event, TranslationEvent) and event.tracked_object is body
        ]

    def test_a_carried_body_still_reports_that_it_moves(self):
        world = graspable_world()
        cup = world.get_body_by_name("cup")
        gripper = world.get_body_by_name("gripper")
        executor = EpisodeSegmenterExecutor(
            context=MotionStatechartContext(world=world)
        )
        segmind_context = executor.context.require_extension(SegmindContext)
        executor.compile(SegmindStatechart().build_statechart([TranslationDetector()]))

        executor.tick()  # the cup is free here, and so watched
        self.grasp(world, cup, gripper)
        for step in range(6):
            gripper.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    0, 0, 0.1 * step, reference_frame=world.root
                )
            )
            executor.tick()

        assert self.translations_of(segmind_context, cup)

    def test_a_free_body_is_watched_without_ever_being_grasped(self):
        world = graspable_world()
        cup = world.get_body_by_name("cup")
        executor = EpisodeSegmenterExecutor(
            context=MotionStatechartContext(world=world)
        )
        segmind_context = executor.context.require_extension(SegmindContext)
        executor.compile(SegmindStatechart().build_statechart([TranslationDetector()]))

        for step in range(6):
            cup.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                0, 0, 0.1 * step, reference_frame=world.root
            )
            executor.tick()

        assert self.translations_of(segmind_context, cup)
