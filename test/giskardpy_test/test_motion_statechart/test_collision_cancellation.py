"""
Collision cancellation checks current observations after its activation tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import pytest

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from giskardpy.motion_statechart.goals.collision_avoidance import (
    _CancelBecauseCollisionViolated,
    _CancelBecauseExternalCollisionViolated,
    _CancelBecauseSelfCollisionViolated,
    _CollisionAvoidanceTask,
    _ExternalCollisionAvoidanceTask,
    _SelfCollisionAvoidanceTask,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.collision_checking.collision_detector import ClosestPoints
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World


# %% observed collision pair and cancellation state
@dataclass
class ObservedCollisionCancellation:
    """
    A real detector contact and its independently updated observation state.
    """

    context: MotionStatechartContext
    """Collision variables shared with the cancellation node."""
    statechart: MotionStatechart
    """
    Observation buffer whose values can change after activation.
    """

    task: _CollisionAvoidanceTask
    """Avoidance task observing the selected contact."""
    cancellation: _CancelBecauseCollisionViolated
    """
    Terminal node requested to check the latest observations.
    """

    contact: ClosestPoints
    """Actual closest-points result expected in a violation exception."""


@pytest.fixture(
    params=[
        _CancelBecauseExternalCollisionViolated,
        _CancelBecauseSelfCollisionViolated,
    ]
)
def observed_cancellation(
    cylinder_bot_world: World, request: pytest.FixtureRequest
) -> Iterator[ObservedCollisionCancellation]:
    """
    Exercise both cancellation paths using real registered collision groups.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(MinimalRobot)[0]
    obstacle = world.get_body_by_name("environment")
    manager = world.collision_manager
    manager.temporary_rules.append(
        AvoidCollisionBetweenGroups(
            body_group_a=[robot.root],
            body_group_b=[obstacle],
            buffer_zone_distance=1.0,
            violated_distance=1.0,
        )
    )
    manager.update_collision_matrix()
    context = MotionStatechartContext(world)
    if request.param is _CancelBecauseExternalCollisionViolated:
        contacts = context.external_collision_manager
        contacts.register_group_of_body(robot.root)
        group = contacts.get_collision_group(robot.root)
        task = _ExternalCollisionAvoidanceTask(
            collision_group=group, external_collision_manager=contacts
        )
        manager.compute_collisions()
        contact = contacts.last_closest_contacts[group][0]
    else:
        contacts = context.self_collision_manager
        contacts.register_groups_of_body_combination(robot.root, obstacle)
        groups = contacts.body_pair_to_group_pair(robot.root, obstacle)
        task = _SelfCollisionAvoidanceTask(
            collision_group_a=groups[0],
            collision_group_b=groups[1],
            self_collision_manager=contacts,
        )
        manager.compute_collisions()
        contact = contacts.last_closest_contacts[groups][0]
    statechart = MotionStatechart()
    statechart.add_node(task)
    cancellation = request.param(tasks=[task])
    statechart.add_node(cancellation)
    yield ObservedCollisionCancellation(
        context, statechart, task, cancellation, contact
    )
    context.cleanup()


# %% current observations govern cancellation
@pytest.mark.parametrize(
    "recovered_state", [ObservationStateValues.TRUE, ObservationStateValues.UNKNOWN]
)
def test_activated_cancellation_does_not_raise_without_a_current_violation(
    observed_cancellation: ObservedCollisionCancellation,
    recovered_state: ObservationStateValues,
) -> None:
    """
    A previously violated task may recover before the cancel node runs again.
    """
    observed = observed_cancellation
    observed.statechart.observation_state[observed.task] = ObservationStateValues.FALSE
    with pytest.raises(CollisionViolatedError):
        observed.cancellation.on_tick(observed.context)
    observed.statechart.observation_state[observed.task] = recovered_state
    assert observed.cancellation.on_tick(observed.context) is None


def test_current_collision_still_raises_with_its_contact_and_threshold(
    observed_cancellation: ObservedCollisionCancellation,
) -> None:
    """
    Genuine current violations retain the detector pair and required distance.
    """
    observed = observed_cancellation
    observed.statechart.observation_state[observed.task] = ObservationStateValues.FALSE
    with pytest.raises(CollisionViolatedError) as failure:
        observed.cancellation.on_tick(observed.context)
    assert failure.value.violated_collisions == [observed.contact]
    assert failure.value.thresholds == [observed.task.violated_distance.evaluate()[0]]
