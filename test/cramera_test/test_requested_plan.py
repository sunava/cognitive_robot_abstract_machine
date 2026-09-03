"""
The plan the Plan Builder asks a running scene to perform: reading it off the wire, and
turning it into the coraplex actions that carry it out.
"""

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import CounterTop
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.placement_surface import UnknownPlacementSurface
from cramera.live.requested_plan import (
    MalformedPlanRequest,
    MoveTorso,
    ParkArms,
    RequestedPlan,
    StepType,
    SurfaceNotInWorld,
)

from .test_live_bridge import shaped_body, world_with

TRANSPORTED = "milk.stl"
"""
The mesh name a transported body carries, which is how the plan names it.
"""


def transporting_world():
    """
    A world holding the transported body and two counter tops to place it on.
    """
    world = world_with(
        Body(name=PrefixedName(TRANSPORTED)),
        shaped_body("apartment", "sink_area_counter_top"),
        shaped_body("apartment", "island_counter_top"),
    )
    with world.modify_world():
        for name in ("sink_area_counter_top", "island_counter_top"):
            world.add_semantic_annotation(CounterTop(root=world.get_body_by_name(name)))
    return world


def context_on(world) -> Context:
    """
    The context a step is resolved against, for a scene with no robot in it.
    """
    return Context(world=world, robot=None)


def robot_context(world) -> Context:
    """
    The context of a scene with a robot in it, which choosing a grasp needs.
    """
    return Context(world=world, robot=world.get_semantic_annotations_by_type(PR2)[0])


def requested(*steps):
    """
    A plan request built from the step payloads the builder posts.
    """
    return RequestedPlan.from_payload({"steps": list(steps)})


def step(step_type, **parameters):
    """
    One step payload, shaped the way the builder's steps are.
    """
    return {"type": step_type.value, "params": parameters}


# %% reading a plan off the wire
class TestReadingAPlan:
    def test_a_plan_keeps_the_order_its_steps_were_written_in(self):
        plan = requested(
            step(StepType.PARK_ARMS, arm="BOTH"),
            step(StepType.MOVE_TORSO, torso="HIGH"),
        )
        assert [type(found) for found in plan.steps] == [ParkArms, MoveTorso]
        assert plan.steps[0].arm is Arms.BOTH
        assert plan.steps[1].torso_state is TorsoState.HIGH

    def test_a_plan_without_steps_is_refused(self):
        with pytest.raises(MalformedPlanRequest):
            RequestedPlan.from_payload({"steps": []})

    def test_a_step_of_an_unknown_type_is_refused(self):
        with pytest.raises(MalformedPlanRequest):
            requested({"type": "make_coffee", "params": {}})

    def test_an_arm_the_robot_does_not_have_is_refused(self):
        with pytest.raises(MalformedPlanRequest):
            requested(step(StepType.PARK_ARMS, arm="THIRD"))

    def test_a_torso_state_that_is_not_one_is_refused(self):
        with pytest.raises(MalformedPlanRequest):
            requested(step(StepType.MOVE_TORSO, torso="SLIGHTLY_UP"))

    def test_a_coordinate_that_is_not_a_number_is_refused(self):
        with pytest.raises(MalformedPlanRequest):
            requested(step(StepType.NAVIGATE, x="over there", y=1.0, z=0.0, yaw=0.0))

    def test_a_transport_naming_no_object_is_refused(self):
        with pytest.raises(MalformedPlanRequest):
            requested(step(StepType.TRANSPORT, arm="LEFT", targetMode="pose"))


# %% the actions a plan performs
class TestActionsAPlanPerforms:
    def test_parking_arms_parks_the_named_arm(self):
        action = (
            requested(step(StepType.PARK_ARMS, arm="LEFT"))
            .steps[0]
            .action(context_on(world_with()))
        )
        assert isinstance(action, ParkArmsAction)
        assert action.arm is Arms.LEFT

    def test_moving_the_torso_moves_it_to_the_named_state(self):
        action = (
            requested(step(StepType.MOVE_TORSO, torso="LOW"))
            .steps[0]
            .action(context_on(world_with()))
        )
        assert isinstance(action, MoveTorsoAction)
        assert action.torso_state is TorsoState.LOW

    def test_navigating_goes_to_the_given_place_facing_the_given_way(self):
        world = world_with()
        action = (
            requested(step(StepType.NAVIGATE, x=2.6, y=1.8, z=0.0, yaw=1.57))
            .steps[0]
            .action(context_on(world))
        )
        assert isinstance(action, NavigateAction)
        assert action.target_location.to_position().to_np()[:3] == pytest.approx(
            [2.6, 1.8, 0.0]
        )

    def test_transporting_to_a_pose_carries_the_named_body_there(self):
        world = transporting_world()
        action = (
            requested(
                step(
                    StepType.TRANSPORT,
                    object=TRANSPORTED,
                    arm="LEFT",
                    targetMode="pose",
                    x=3.0,
                    y=2.0,
                    z=1.0,
                    yaw=0.0,
                )
            )
            .steps[0]
            .action(context_on(world))
        )
        assert isinstance(action, TransportAction)
        assert action.object_designator is world.get_body_by_name(TRANSPORTED)
        assert action.arm is Arms.LEFT
        assert action.target_location.to_position().to_np()[:3] == pytest.approx(
            [3.0, 2.0, 1.0]
        )

    def test_a_transport_looks_where_it_operates_when_asked_to(self):
        action = (
            requested(
                step(
                    StepType.TRANSPORT,
                    object=TRANSPORTED,
                    arm="LEFT",
                    targetMode="pose",
                    x=3.0,
                    y=2.0,
                    z=1.0,
                    yaw=0.0,
                    look_at_operation_site=True,
                )
            )
            .steps[0]
            .action(context_on(transporting_world()))
        )
        assert action.look_at_operation_site is True

    def test_a_transport_does_not_look_around_unasked(self):
        action = (
            requested(
                step(
                    StepType.TRANSPORT,
                    object=TRANSPORTED,
                    arm="LEFT",
                    targetMode="pose",
                    x=3.0,
                    y=2.0,
                    z=1.0,
                    yaw=0.0,
                )
            )
            .steps[0]
            .action(context_on(transporting_world()))
        )
        assert action.look_at_operation_site is False


# %% placing on a named surface
class TestPlacingOnASurface:
    def surface_step(self, **parameters):
        return requested(
            step(
                StepType.TRANSPORT,
                object=TRANSPORTED,
                arm="LEFT",
                targetMode="semantic",
                **parameters,
            )
        ).steps[0]

    def test_a_named_surface_is_the_one_placed_on(self):
        world = transporting_world()
        surface = self.surface_step(
            surfaceType=CounterTop.__name__, surfaceName="island_counter_top"
        ).target.surface(world)
        assert surface.root is world.get_body_by_name("island_counter_top")

    def test_an_unnamed_surface_takes_the_first_of_its_kind(self):
        world = transporting_world()
        surface = self.surface_step(
            surfaceType=CounterTop.__name__, surfaceName=""
        ).target.surface(world)
        assert surface is world.get_semantic_annotations_by_type(CounterTop)[0]

    def test_a_surface_the_world_does_not_have_says_so(self):
        world = transporting_world()
        target = self.surface_step(
            surfaceType=CounterTop.__name__, surfaceName="balcony_counter_top"
        ).target
        with pytest.raises(SurfaceNotInWorld):
            target.surface(world)

    def test_an_annotation_nothing_can_be_placed_on_is_refused_while_reading(self):
        with pytest.raises(UnknownPlacementSurface):
            self.surface_step(surfaceType="Milk", surfaceName="")


# %% picking an object up and putting it down as two steps
class TestPickingAndPlacing:
    """
    A pick and a place are a transport spelled out, for a world whose floor carries no
    costmap the transport could search — so the running scene has to understand them
    too, or a plan built from them can only be generated, never run.
    """

    def picked(self, context, **parameters):
        return (
            requested(step(StepType.PICK, object=TRANSPORTED, arm="LEFT", **parameters))
            .steps[0]
            .action(context)
        )

    def placed(self, world, **parameters):
        return (
            requested(
                step(
                    StepType.PLACE,
                    object=TRANSPORTED,
                    arm="LEFT",
                    targetMode="pose",
                    x=3.0,
                    y=2.0,
                    z=1.0,
                    yaw=0.0,
                    **parameters,
                )
            )
            .steps[0]
            .action(context_on(world))
        )

    def test_picking_up_takes_the_named_body_with_the_named_arm(
        self, pr2_apartment_world
    ):
        world = pr2_apartment_world
        action = self.picked(robot_context(world))
        assert isinstance(action, PickUpAction)
        assert action.object_designator is world.get_body_by_name(TRANSPORTED)
        assert action.arm is Arms.LEFT

    def test_picking_up_brings_a_grasp_the_robot_can_reach_with(
        self, pr2_apartment_world
    ):
        action = self.picked(robot_context(pr2_apartment_world))
        assert action.grasp_description is not None

    def test_placing_puts_the_named_body_at_the_given_pose(self):
        world = transporting_world()
        action = self.placed(world)
        assert isinstance(action, PlaceAction)
        assert action.object_designator is world.get_body_by_name(TRANSPORTED)
        assert action.arm is Arms.LEFT
        assert action.target_location.to_position().to_np()[:3] == pytest.approx(
            [3.0, 2.0, 1.0]
        )

    def test_placing_can_be_told_a_surface_instead_of_a_pose(self):
        placed = requested(
            step(
                StepType.PLACE,
                object=TRANSPORTED,
                arm="LEFT",
                targetMode="semantic",
                surfaceType=CounterTop.__name__,
                surfaceName="island_counter_top",
            )
        ).steps[0]
        world = transporting_world()
        assert placed.target.surface(world).root is world.get_body_by_name(
            "island_counter_top"
        )

    def test_a_pick_naming_no_object_is_refused(self):
        with pytest.raises(MalformedPlanRequest):
            requested(step(StepType.PICK, arm="LEFT"))
