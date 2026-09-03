"""
Performing a requested plan in a scene that is already running: one at a time, and
saying afterwards how it went.
"""

import pytest

from coraplex.datastructures.dataclasses import Context
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.plan_runner import (
    PlanAlreadyRunning,
    PlanRun,
    PlanRunner,
    RunState,
)
from cramera.live.requested_plan import RequestedPlan, StepType

from .test_live_bridge import world_with

MISSING_BODY = "cereal.stl"
"""
A body no world in these tests holds, so transporting it cannot be resolved.
"""


def runner_on(*bodies: Body) -> PlanRunner:
    """
    A runner serving a world holding the given bodies and no robot.

    :param bodies: The bodies the world is built from.
    """
    world = world_with(*bodies)
    return PlanRunner(context=Context(world=world, robot=None), visualization=None)


def transport_of(name: str) -> RequestedPlan:
    """
    A plan carrying the named body to a fixed pose.
    """
    return RequestedPlan.from_payload(
        {
            "steps": [
                {
                    "type": StepType.TRANSPORT.value,
                    "params": {
                        "object": name,
                        "arm": "LEFT",
                        "targetMode": "pose",
                        "x": 1.0,
                        "y": 1.0,
                        "z": 1.0,
                        "yaw": 0.0,
                    },
                }
            ]
        }
    )


# %% what became of a run
class TestRunOutcome:
    def test_nothing_has_run_yet(self):
        assert PlanRun().state is RunState.IDLE

    def test_a_begun_run_is_running(self):
        run = PlanRun()
        run.begin()
        assert run.state is RunState.RUNNING

    def test_a_second_run_is_refused_while_one_is_going(self):
        run = PlanRun()
        run.begin()
        with pytest.raises(PlanAlreadyRunning):
            run.begin()

    def test_a_run_that_ended_lets_the_next_one_start(self):
        run = PlanRun()
        run.begin()
        run.finish()
        run.begin()
        assert run.state is RunState.RUNNING

    def test_a_failed_run_keeps_what_went_wrong(self):
        run = PlanRun()
        run.begin()
        run.fail(ValueError("no reachable base pose"))
        assert run.state is RunState.FAILED
        assert "no reachable base pose" in run.error

    def test_a_run_that_ended_well_has_nothing_to_report(self):
        run = PlanRun()
        run.begin()
        run.finish()
        assert run.state is RunState.FINISHED
        assert run.error is None


# %% performing a plan in a running scene
class TestPerformingAPlan:
    def test_a_plan_with_nothing_in_it_finishes(self):
        runner = runner_on()
        runner.submit(RequestedPlan(steps=()))
        runner.wait()
        assert runner.run.state is RunState.FINISHED

    def test_a_plan_the_world_cannot_carry_out_reports_why(self):
        runner = runner_on(Body(name=PrefixedName("milk.stl")))
        runner.submit(transport_of(MISSING_BODY))
        runner.wait()
        assert runner.run.state is RunState.FAILED
        assert MISSING_BODY in runner.run.error

    def test_the_scene_performs_one_plan_at_a_time(self):
        runner = runner_on()
        runner.submit(RequestedPlan(steps=()))
        runner.wait()
        runner.submit(RequestedPlan(steps=()))
        runner.wait()
        assert runner.run.state is RunState.FINISHED
