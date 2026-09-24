"""
Publish native plan status without exposing partially updated worker state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from threading import Event, Thread

import numpy as np
import pytest

from coraplex.datastructures.enums import ExecutionType, TaskStatus
from coraplex.execution_environment import no_execution, simulated_robot_advanced
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.factories import sequential
from cramera.laboratory_actions import (
    LaboratoryContext,
    LaboratoryOperation,
    WeighSampleAction,
)
from cramera.laboratory_plan import (
    LaboratoryPlan,
    LaboratoryPlanExecution,
    LaboratoryPlanField,
)
from cramera.laboratory_plan_world import LaboratoryPlanWorld, LaboratoryPoseMeasurement
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World
from typing_extensions import Any

from .test_laboratory_actions import RecordedExecution, make_context


# %% independent status readers
@dataclass
class PendingStatusRead:
    """
    Read one public snapshot from a separate caller thread.
    """

    execution: LaboratoryPlanExecution
    """Execution whose in-progress publication must be serialized."""
    started: Event = field(default_factory=Event)
    """
    Signal that the independent reader is about to request status.
    """

    returned: Event = field(default_factory=Event)
    """
    Signal that the read has returned its detached result.
    """

    result: dict[str, Any] | None = None
    """
    The complete status returned by the execution.
    """

    def run(self) -> None:
        """
        Wait for a consistent publication before exposing the read result.
        """
        self.started.set()
        self.result = self.execution.snapshot()
        self.returned.set()


def test_snapshot_waits_for_the_complete_action_publication(
    cylinder_bot_world: World,
) -> None:
    """
    A concurrent reader cannot observe a half-published action and completion count.
    """
    execution = LaboratoryPlanExecution()
    context = LaboratoryContext(
        world=cylinder_bot_world,
        robot=cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0],
        executor=execution,
    )
    execution.plan = LaboratoryPlan(("sample",)).build(context)
    reader = PendingStatusRead(execution)
    worker = Thread(target=reader.run, daemon=True)
    request = WeighSampleAction("sample").request
    try:
        with execution.condition:
            execution.active_request = request
            worker.start()
            assert reader.started.wait(timeout=1.0)
            assert not reader.returned.wait(timeout=0.05)
            execution.completed = 1
        worker.join(timeout=1.0)
        assert not worker.is_alive()
        assert reader.result[LaboratoryPlanField.ACTION] == request.operation
        assert reader.result[LaboratoryPlanField.COMPLETED] == execution.completed
    finally:
        worker.join(timeout=1.0)


def test_native_worker_preserves_an_enclosing_standard_simulation_scope(
    cylinder_bot_world: World,
) -> None:
    """
    A native worker cannot overwrite its caller's saved execution environment.
    """
    executor = RecordedExecution()
    execution = LaboratoryPlanExecution()
    action = WeighSampleAction("sample")
    execution.plan = sequential(
        [action], context=make_context(cylinder_bot_world, executor)
    ).plan
    execution.measurement = LaboratoryPlanWorld(0.0, (), ())
    with no_execution:
        with simulated_robot_advanced:
            worker = Thread(target=execution._perform, daemon=True)
            worker.start()
            worker.join(timeout=2.0)
            assert not worker.is_alive()
            assert execution.plan.root.status == TaskStatus.SUCCEEDED
            assert executor.requests == [action.request]
            assert GiskardExecutable.execution_type == ExecutionType.SIMULATED

        assert GiskardExecutable.execution_type == ExecutionType.NO_EXECUTION
        assert GiskardExecutable.collision_avoidance == no_execution.collision_avoidance


@pytest.mark.parametrize(
    ("failure", "expected"),
    [(None, TaskStatus.SUCCEEDED), (LaboratoryOperation.WEIGH, TaskStatus.FAILED)],
)
def test_cancel_preserves_a_finished_native_plan_result(
    cylinder_bot_world: World,
    failure: LaboratoryOperation | None,
    expected: TaskStatus,
) -> None:
    """
    Releasing an already finished plan retains its success or failure evidence.
    """
    executor = RecordedExecution(fail_operation=failure)
    execution = LaboratoryPlanExecution()
    execution.plan = sequential(
        [WeighSampleAction("sample")],
        context=make_context(cylinder_bot_world, executor),
    ).plan
    execution.measurement = LaboratoryPlanWorld(0.0, (), ())
    execution._perform()
    assert execution.finished
    assert execution.plan.root.status == expected

    execution.cancel()

    assert execution.plan.root.status == expected


@pytest.mark.parametrize("acknowledged", [False, True])
def test_cancel_records_only_an_acknowledged_failure_measurement(
    cylinder_bot_world: World, acknowledged: bool
) -> None:
    """
    Cancellation preserves an accepted stop measurement without replaying stale state.
    """
    execution = LaboratoryPlanExecution()
    context = make_context(cylinder_bot_world, execution)
    execution.plan = sequential([WeighSampleAction("sample")], context=context).plan
    execution.measurement = LaboratoryPlanWorld(0.0, (), ())
    worker = Thread(target=execution._perform, daemon=True)
    worker.start()
    try:
        request = execution.wait_for_request()
        assert request == WeighSampleAction("sample").request
        updated = LaboratoryPlanWorld(
            1.0,
            (),
            (
                LaboratoryPoseMeasurement(
                    str(context.robot.root.name),
                    (0.25, 0.1, 0.05),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            ),
        )
        expected_world = copy.deepcopy(execution.plan.world)
        if acknowledged:
            updated.apply(expected_world)
        expected = expected_world.state._data.copy()

        # Publish both outcomes before the waiting physical action can resume.
        with execution.condition:
            if acknowledged:
                execution.complete("Measured stop", updated)
            else:
                execution.measurement = updated
            execution.cancel()
        worker.join(timeout=2.0)

        assert not worker.is_alive()
        np.testing.assert_array_equal(execution.plan.world.state._data, expected)
        [action] = execution.plan.actions
        np.testing.assert_array_equal(
            action.execution_data.execution_end_world_state, expected
        )
    finally:
        execution.cancel()
        worker.join(timeout=2.0)
