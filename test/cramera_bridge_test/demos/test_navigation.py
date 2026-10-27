"""
The navigation demo: its environment, plan and measurement.

Performing the drive is not tested: the navigation route planner of the CRAM workspace
this package is tested against fails before the robot moves.
"""

from __future__ import annotations

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from cramera.locations import DataLocations
from cramera_bridge.demos.navigation import (
    BarrierEnvironment,
    NavigationBody,
    NavigationDemo,
    NavigationMeasurement,
)
from cramera_bridge.exceptions import TaskFailed
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.robots.pr2 import PR2

# %% fixtures


@pytest.fixture
def demo(tmp_path) -> NavigationDemo:
    """
    :return: A navigation demo of a PR2 that starts at the world's origin.
    """
    return NavigationDemo(
        environment=BarrierEnvironment(),
        robot=RobotSpecification(semantic_annotation_type=PR2),
        name="barrier_pr2_navigation",
        task="PR2: drive past the barrier",
        locations=DataLocations(data=tmp_path, shared_scenes=None),
    )


@pytest.fixture
def context(demo) -> Context:
    """
    :return: The demo's world and robot before the drive.
    """
    return demo.build_context(demo.build_world())


# %% environment and plan


def test_the_barrier_stands_halfway(demo, context):
    barrier = context.world.get_body_by_name(NavigationBody.BARRIER)
    position = barrier.global_pose.position.to_np()
    assert position[0] == pytest.approx(demo.environment.distance / 2)


def test_the_plan_navigates_to_the_target(demo, context):
    plan = demo.build_plan(context)
    action = plan.root.designator
    assert isinstance(action, NavigateAction)
    assert action.target_location.position.to_np()[0] == pytest.approx(demo.target_x)


# %% measurement


def test_a_robot_at_its_target_passes(demo, context):
    demo.target_x = 0.0
    metrics = demo.measure(context)
    assert metrics.measurements[NavigationMeasurement.BASE_ERROR.value] == 0.0


def test_a_robot_short_of_its_target_fails(demo, context):
    with pytest.raises(TaskFailed):
        demo.measure(context)
