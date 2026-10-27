"""
The plan with native statuses, and the constraints attached to its nodes.
"""

from __future__ import annotations

import json
from http import HTTPStatus

import pytest

from cramera_bridge.exceptions import (
    InvalidConstraint,
    RobotSelectionBusy,
    UnknownBody,
    UnknownPlanNode,
)
from coraplex.robot_plans.actions.composite.transporting import MoveAndPickUpAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from cramera_bridge.live.plan import DesignatorSummary, PLAN_NODE_PREFIX, PlanNodeStatus
from cramera_protocol.live import LiveRoute
from cramera_protocol.live_messages import (
    ConstraintActivation,
    ConstraintMonitorType,
    ConstraintRequest,
    LivePlan,
    MonitorSpecification,
    PlanNodeReference,
    RobotSelection,
)
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.motions.container import OpeningMotion
from giskardpy.motion_statechart.data_types import LifeCycleValues
from krrood.entity_query_language.factories import a, variable
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types.spatial_types import Pose
from giskardpy.motion_statechart.data_types import ObservationStateValues
from cramera.http.messages import HttpMethod

from ..dataset.live_worlds import HeightChart
from .conftest import dataset_body, live_request

# %% plan


def test_every_node_shows_its_own_native_status(publisher, small_plan):
    small_plan.root.status = LifeCycleValues.RUNNING
    publisher.begin_plan(small_plan.plan)
    plan = publisher.current_plan()
    assert [node.status for node in plan.nodes] == [
        PlanNodeStatus.RUNNING,
        PlanNodeStatus.CREATED,
        PlanNodeStatus.CREATED,
    ]
    assert plan.nodes[1].parent == plan.nodes[0].identifier
    assert plan.nodes[1].label == type(small_plan.action.designator).__name__


def test_a_parent_status_is_never_derived_from_its_children(publisher, small_plan):
    small_plan.action.status = LifeCycleValues.SUCCEEDED
    publisher.begin_plan(small_plan.plan)
    root = publisher.current_plan().nodes[0]
    assert root.status == PlanNodeStatus.CREATED


@pytest.mark.parametrize(
    "motion_status, action_status",
    [
        (LifeCycleValues.NOT_STARTED, PlanNodeStatus.CREATED),
        (LifeCycleValues.RUNNING, PlanNodeStatus.RUNNING),
        (LifeCycleValues.SUCCEEDED, PlanNodeStatus.SUCCEEDED),
        (LifeCycleValues.FAILED, PlanNodeStatus.FAILED),
    ],
)
def test_an_action_coraplex_left_unstarted_shows_what_its_motion_went_through(
    publisher, small_plan, motion_status, action_status
):
    small_plan.motion.status = motion_status
    publisher.begin_plan(small_plan.plan)
    _, action, _ = publisher.current_plan().nodes
    assert action.status == action_status


def test_an_action_coraplex_started_keeps_its_own_status(publisher, small_plan):
    small_plan.action.status = LifeCycleValues.FAILED
    small_plan.motion.status = LifeCycleValues.SUCCEEDED
    publisher.begin_plan(small_plan.plan)
    _, action, _ = publisher.current_plan().nodes
    assert action.status == PlanNodeStatus.FAILED


def test_an_action_runs_until_its_motions_ran_or_the_plan_ended(
    publisher, small_plan, kitchen, robot_world
):
    later = MotionNode(
        designator=OpeningMotion(
            object_part=kitchen.milk, arm=robot_world.robot.left_arm
        )
    )
    small_plan.action.add_child(later)
    small_plan.motion.status = LifeCycleValues.SUCCEEDED
    small_plan.root.status = LifeCycleValues.RUNNING
    publisher.begin_plan(small_plan.plan)
    running = publisher.current_plan().nodes[1].status
    small_plan.root.status = LifeCycleValues.SUCCEEDED
    ended = publisher.refresh_plan().nodes[1].status
    assert (running, ended) == (PlanNodeStatus.RUNNING, PlanNodeStatus.SUCCEEDED)


def test_an_action_whose_motion_runs_names_the_running_step(publisher, small_plan):
    small_plan.motion.status = LifeCycleValues.RUNNING
    publisher.begin_plan(small_plan.plan)
    assert (
        publisher.plan.running_action() == type(small_plan.action.designator).__name__
    )


def test_a_motion_names_its_arm_and_object(publisher, small_plan, kitchen, robot_world):
    publisher.begin_plan(small_plan.plan)
    motion = publisher.current_plan().nodes[2]
    assert motion.arm == str(robot_world.robot.left_arm.name)
    assert motion.target == str(kitchen.milk.id)


def test_a_node_renders_every_field_of_its_designator(
    publisher, small_plan, kitchen, robot_world
):
    publisher.begin_plan(small_plan.plan)
    root, action, motion = publisher.current_plan().nodes
    assert root.parameters == {}
    assert action.parameters == {"torso_state": TorsoState.HIGH.name}
    assert motion.parameters["arm"] == str(robot_world.robot.left_arm.name)
    assert motion.parameters["object_part"] == str(kitchen.milk.id)


def test_a_node_is_found_by_its_identity(publisher, small_plan):
    publisher.begin_plan(small_plan.plan)
    identifier = publisher.plan.active_plan.identifier(small_plan.action)
    assert identifier.startswith(PLAN_NODE_PREFIX)
    assert publisher.plan.published_node(identifier).identifier == identifier
    with pytest.raises(UnknownPlanNode):
        publisher.plan.published_node("plan_node_99")


def test_no_plan_has_no_nodes(publisher):
    assert publisher.current_plan() == LivePlan()
    assert publisher.plan.running_action() is None


def test_the_running_action_names_the_step(publisher, small_plan):
    small_plan.action.status = LifeCycleValues.RUNNING
    publisher.begin_plan(small_plan.plan)
    assert (
        publisher.plan.running_action() == type(small_plan.action.designator).__name__
    )


def test_a_later_plan_keeps_its_node_identities_apart_from_the_first(
    publisher, robot_plans
):
    first, second = robot_plans
    publisher.begin_plan(first.plan)
    first_identities = {node.identifier for node in publisher.current_plan().nodes}
    publisher.begin_plan(second.plan)
    second_identities = {node.identifier for node in publisher.current_plan().nodes}
    assert len(second_identities) == len(second.plan.all_nodes)
    assert first_identities.isdisjoint(second_identities)


def test_a_plan_begun_again_is_published_with_its_first_identities(
    publisher, robot_plans
):
    first, second = robot_plans
    publisher.begin_plan(first.plan)
    identities = [node.identifier for node in publisher.current_plan().nodes]
    publisher.begin_plan(second.plan)
    publisher.begin_plan(first.plan)
    assert [node.identifier for node in publisher.current_plan().nodes] == identities


def test_a_robot_cannot_be_selected_while_the_plan_runs(publisher, small_plan):
    small_plan.root.status = LifeCycleValues.RUNNING
    publisher.begin_plan(small_plan.plan)
    with pytest.raises(RobotSelectionBusy):
        publisher.select_robot(RobotSelection(identifier="any"))


# %% constraints


def height_constraint(publisher, kitchen, small_plan, activation) -> ConstraintRequest:
    """
    :return: A height constraint on the milk attached to the root of the plan being
        performed.
    """
    return ConstraintRequest(
        target_plan_node=PlanNodeReference(
            identifier=publisher.plan.active_plan.identifier(small_plan.root)
        ),
        monitor=MonitorSpecification(
            monitor_type=ConstraintMonitorType.HEIGHT,
            parameters={"tip_link": kitchen.milk.name.name, "lower_limit": 0.5},
        ),
        text="keep the milk up",
        apply=activation,
    )


def test_an_attached_constraint_is_observed_on_the_next_tick(
    publisher, kitchen, small_plan
):
    publisher.begin_plan(small_plan.plan)
    publisher.attach_constraint(
        height_constraint(
            publisher, kitchen, small_plan, ConstraintActivation.IMMEDIATE
        )
    )
    chart = HeightChart.build(kitchen.world, kitchen.milk)
    publisher.observe_motion_tick(chart.chart)
    constraint = publisher.refresh_plan().nodes[0].constraints[0]
    assert constraint.observation == ObservationStateValues.TRUE.name


def test_a_constraint_waits_for_the_next_motion(publisher, kitchen, small_plan):
    publisher.begin_plan(small_plan.plan)
    publisher.attach_constraint(
        height_constraint(
            publisher, kitchen, small_plan, ConstraintActivation.NEXT_ACTIVATION
        )
    )
    chart = HeightChart.build(kitchen.world, kitchen.milk)
    publisher.observe_motion_tick(chart.chart)
    assert publisher.refresh_plan().nodes[0].constraints[0].observation is None
    publisher.observe_node_started(small_plan.motion)
    publisher.observe_motion_tick(chart.chart)
    observed = publisher.refresh_plan().nodes[0].constraints[0]
    assert observed.observation == ObservationStateValues.TRUE.name


def test_a_constraint_sees_the_world_change(publisher, kitchen, small_plan):
    publisher.begin_plan(small_plan.plan)
    publisher.attach_constraint(
        height_constraint(
            publisher, kitchen, small_plan, ConstraintActivation.IMMEDIATE
        )
    )
    chart = HeightChart.build(kitchen.world, kitchen.milk)
    publisher.observe_motion_tick(chart.chart)
    kitchen.milk_connection.origin = type(kitchen.milk_connection.origin).from_xyz_rpy(
        1.0, 0.1, 0.1, reference_frame=kitchen.root
    )
    publisher.observe_motion_tick(chart.chart)
    observed = publisher.refresh_plan().nodes[0].constraints[0]
    assert observed.observation == ObservationStateValues.FALSE.name


def test_a_constraint_reaches_a_node_of_a_later_plan(publisher, kitchen, robot_plans):
    first, second = robot_plans
    publisher.begin_plan(first.plan)
    publisher.begin_plan(second.plan)
    plan = publisher.attach_constraint(
        height_constraint(publisher, kitchen, second, ConstraintActivation.IMMEDIATE)
    )
    assert [node.constraints != [] for node in plan.nodes] == [True, False, False]


@pytest.mark.parametrize(
    "monitor_type, parameters",
    [
        (ConstraintMonitorType.VECTORS_ALIGNED, {"tip_link": "milk"}),
        (
            ConstraintMonitorType.POINTING_AT,
            {"tip_link": "milk", "goal_point_body": "cup"},
        ),
        (
            ConstraintMonitorType.POINTING_AT,
            {"tip_link": "milk", "goal_point": [1, 0, 0]},
        ),
        (ConstraintMonitorType.DISTANCE, {"root_link": "map", "tip_link": "milk"}),
    ],
)
def test_every_monitor_type_is_built_and_observed(
    publisher, kitchen, small_plan, monitor_type, parameters
):
    publisher.begin_plan(small_plan.plan)
    request = ConstraintRequest(
        target_plan_node=PlanNodeReference(
            identifier=publisher.plan.active_plan.identifier(small_plan.action)
        ),
        monitor=MonitorSpecification(monitor_type=monitor_type, parameters=parameters),
        apply=ConstraintActivation.IMMEDIATE,
    )
    publisher.attach_constraint(request)
    publisher.observe_motion_tick(HeightChart.build(kitchen.world, kitchen.milk).chart)
    constraint = publisher.refresh_plan().nodes[1].constraints[0]
    assert constraint.monitor == monitor_type
    assert constraint.observation in ObservationStateValues.__members__


@pytest.mark.parametrize(
    "parameters, error",
    [
        ({"tip_link": "fridge"}, UnknownBody),
        ({"tip_link": 3}, InvalidConstraint),
        ({"tip_link": "milk", "goal_normal": [1, 0]}, InvalidConstraint),
    ],
)
def test_an_unreadable_constraint_is_refused(publisher, small_plan, parameters, error):
    publisher.begin_plan(small_plan.plan)
    request = ConstraintRequest(
        target_plan_node=PlanNodeReference(
            identifier=publisher.plan.active_plan.identifier(small_plan.root)
        ),
        monitor=MonitorSpecification(
            monitor_type=ConstraintMonitorType.VECTORS_ALIGNED, parameters=parameters
        ),
    )
    with pytest.raises(error):
        publisher.attach_constraint(request)


def test_a_limit_must_be_a_number(publisher, small_plan):
    publisher.begin_plan(small_plan.plan)
    request = ConstraintRequest(
        target_plan_node=PlanNodeReference(
            identifier=publisher.plan.active_plan.identifier(small_plan.root)
        ),
        monitor=MonitorSpecification(
            monitor_type=ConstraintMonitorType.HEIGHT,
            parameters={"tip_link": "milk", "lower_limit": "low"},
        ),
    )
    with pytest.raises(InvalidConstraint):
        publisher.attach_constraint(request)


def test_pointing_needs_a_goal(publisher, small_plan):
    publisher.begin_plan(small_plan.plan)
    request = ConstraintRequest(
        target_plan_node=PlanNodeReference(
            identifier=publisher.plan.active_plan.identifier(small_plan.root)
        ),
        monitor=MonitorSpecification(
            monitor_type=ConstraintMonitorType.POINTING_AT,
            parameters={"tip_link": "milk"},
        ),
    )
    with pytest.raises(InvalidConstraint):
        publisher.attach_constraint(request)


def test_a_constraint_posted_by_the_plan_view_reaches_its_node(
    publisher, answerer, small_plan, kitchen
):
    publisher.begin_plan(small_plan.plan)
    answer = answerer.answer(
        live_request(
            HttpMethod.POST,
            LiveRoute.CONSTRAINT,
            dataset_body("height_constraint.json"),
        )
    )
    plan = LivePlan.from_json(json.loads(answer.body))
    assert answer.status is HTTPStatus.OK
    assert [node.constraints != [] for node in plan.nodes] == [False, True, False]


def test_a_sub_action_left_to_be_found_is_named_by_what_it_matches():
    pick_up = a(MoveAndPickUpAction)(
        navigate=a(NavigateAction)(target_location=variable(Pose, domain=[])),
        pick_up=a(PickUpAction)(),
    )
    assert DesignatorSummary.render(pick_up, set()) == (
        "MoveAndPickUpAction: navigate NavigateAction, pick_up PickUpAction"
    )
    assert DesignatorSummary.render(a(NavigateAction)(), set()) == "NavigateAction"


def test_a_value_left_open_is_named_by_its_type():
    assert DesignatorSummary.render(variable(Pose, domain=[]), set()) == "any Pose"
