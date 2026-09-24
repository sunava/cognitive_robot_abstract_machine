"""
Native laboratory actions execute individual physical operations in plan order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Thread

import numpy as np
import pytest

from coraplex.datastructures.enums import TaskStatus
from coraplex.execution_environment import no_execution, simulated_robot_advanced
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan import Plan
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import PlanNode
from cramera.laboratory_actions import (
    CloseInstrumentAction,
    DispenseSampleAction,
    LaboratoryActionFailed,
    LaboratoryActionRequest,
    LaboratoryContext,
    LaboratoryGrasp,
    LaboratoryInstrument,
    LaboratoryOperation,
    LaboratoryOperationNode,
    OpenInstrumentAction,
    SampleDestination,
    TransferSampleAction,
    WeighSampleAction,
)
from cramera.laboratory_pipetting_command import PipettingSource
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World


# %% independently observed execution
@dataclass
class RecordedExecution:
    """
    Record physical dispatch and optionally reject a selected operation.
    """

    requests: list[LaboratoryActionRequest] = field(default_factory=list)
    """
    Requests received only when their native executable begins.
    """

    fail_operation: LaboratoryOperation | None = None
    """
    Operation whose physical acceptance should fail.
    """

    def execute(self, request: LaboratoryActionRequest) -> None:
        """
        Publish the attempted operation before reporting its result.
        """
        self.requests.append(request)
        if request.operation == self.fail_operation:
            raise LaboratoryActionFailed("The measured operation failed")


@dataclass
class PausedDispatch(PlanCallback):
    """
    Pause a native operation immediately when its lifecycle begins.
    """

    paused: Event = field(default_factory=Event)
    """
    Indicates that the plan has reached its paused physical operation.
    """

    def on_start(self, node: PlanNode) -> None:
        """
        Request a pause before the operation issues its physical command.
        """
        if isinstance(node, LaboratoryOperationNode) and not self.paused.is_set():
            node.plan.root.pause()
            self.paused.set()


@dataclass
class SimulatedExecution:
    """
    Perform a native plan on the independent caller thread.
    """

    plan: Plan
    """
    Native plan whose execution may wait for resume.
    """

    def run(self) -> None:
        """
        Apply the standard simulated robot environment to plan execution.
        """
        with simulated_robot_advanced:
            self.plan.perform()


@dataclass
class MeasuredProgress:
    """
    Publish independent measured world updates at physical action boundaries.
    """

    world: World
    """
    Semantic observation world receiving the next measured robot pose.
    """

    starts: list[np.ndarray] = field(default_factory=list)
    """
    World arrays observed when individual operations begin.
    """

    ends: list[np.ndarray] = field(default_factory=list)
    """
    World arrays observed after the corresponding measurements arrive.
    """

    def execute(self, request: LaboratoryActionRequest) -> None:
        """
        Advance measured odometry before returning physical completion.
        """
        self.starts.append(self.world.state._data.copy())
        robot = self.world.get_semantic_annotations_by_type(MinimalRobot)[0]
        connection = robot.root.parent_connection
        connection.origin = Pose.from_xyz_rpy(
            x=len(self.starts) * 0.1, reference_frame=connection.parent
        ).to_homogeneous_matrix()
        self.ends.append(self.world.state._data.copy())


def make_context(world: World, executor: RecordedExecution) -> LaboratoryContext:
    """
    Reuse the small articulated robot fixture as a native plan context.
    """
    return LaboratoryContext(
        world=world,
        robot=world.get_semantic_annotations_by_type(MinimalRobot)[0],
        executor=executor,
    )


# %% native execution boundaries
def test_plan_expansion_does_not_issue_physical_commands(cylinder_bot_world):
    executor = RecordedExecution()
    root = sequential(
        [
            OpenInstrumentAction(LaboratoryInstrument.CENTRIFUGE),
            CloseInstrumentAction(LaboratoryInstrument.CENTRIFUGE),
        ],
        context=make_context(cylinder_bot_world, executor),
    )

    root.notify()
    root.parse()

    assert executor.requests == []
    assert (
        len(
            [
                node
                for node in root.descendants
                if isinstance(node, LaboratoryOperationNode)
            ]
        )
        == 2
    )


def test_native_plan_runs_actions_in_order_and_records_success(cylinder_bot_world):
    executor = RecordedExecution()
    root = sequential(
        [
            DispenseSampleAction("sample", PipettingSource.AMBER, 1.0),
            DispenseSampleAction("sample", PipettingSource.TEAL, 1.0),
            WeighSampleAction("sample"),
        ],
        context=make_context(cylinder_bot_world, executor),
    )

    with simulated_robot_advanced:
        root.plan.perform()

    assert [request.operation for request in executor.requests] == [
        LaboratoryOperation.DISPENSE,
        LaboratoryOperation.DISPENSE,
        LaboratoryOperation.WEIGH,
    ]
    assert [request.source for request in executor.requests[:2]] == [
        PipettingSource.AMBER,
        PipettingSource.TEAL,
    ]
    assert root.status == TaskStatus.SUCCEEDED
    assert all(node.status == TaskStatus.SUCCEEDED for node in root.plan.actions)


def test_transfer_is_an_independently_executable_native_action(cylinder_bot_world):
    executor = RecordedExecution()
    action = TransferSampleAction(
        "sample",
        SampleDestination.RACK,
        grasp=LaboratoryGrasp.TOP,
        from_held=True,
        release_at_destination=False,
    )
    node = execute_single(action, context=make_context(cylinder_bot_world, executor))

    with simulated_robot_advanced:
        node.plan.perform()

    assert executor.requests == [
        LaboratoryActionRequest(
            LaboratoryOperation.TRANSFER,
            object_key=action.object_key,
            destination=action.destination,
            grasp=action.grasp,
            from_held=action.from_held,
            release_at_destination=action.release_at_destination,
        )
    ]
    assert node.status == TaskStatus.SUCCEEDED


def test_failed_physical_action_stops_later_native_actions(cylinder_bot_world):
    executor = RecordedExecution(fail_operation=LaboratoryOperation.WEIGH)
    root = sequential(
        [
            WeighSampleAction("sample"),
            OpenInstrumentAction(LaboratoryInstrument.CENTRIFUGE),
        ],
        context=make_context(cylinder_bot_world, executor),
    )

    with simulated_robot_advanced, pytest.raises(LaboratoryActionFailed):
        root.plan.perform()

    assert [request.operation for request in executor.requests] == [
        LaboratoryOperation.WEIGH
    ]
    assert root.status == TaskStatus.FAILED
    failed = [node for node in root.plan.actions if node.status == TaskStatus.FAILED]
    assert len(failed) == 1
    assert isinstance(failed[0].reason, LaboratoryActionFailed)
    assert all(
        node.status == TaskStatus.CREATED
        for node in root.plan.actions
        if isinstance(node.designator, OpenInstrumentAction)
    )


def test_native_pause_before_dispatch_waits_for_resume(cylinder_bot_world):
    executor = RecordedExecution()
    node = execute_single(
        WeighSampleAction("sample"),
        context=make_context(cylinder_bot_world, executor),
    )
    observer = PausedDispatch()
    node.plan.node_callbacks.append(observer)
    worker = Thread(target=SimulatedExecution(node.plan).run, daemon=True)
    worker.start()
    try:
        assert observer.paused.wait(timeout=5)
        assert executor.requests == []
        node.plan.root.resume()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert [request.operation for request in executor.requests] == [
            LaboratoryOperation.WEIGH
        ]
        assert node.status == TaskStatus.SUCCEEDED
    finally:
        node.plan.root.resume()
        worker.join(timeout=5)


def test_no_execution_environment_keeps_physical_state_unchanged(cylinder_bot_world):
    executor = RecordedExecution()
    node = execute_single(
        WeighSampleAction("sample"),
        context=make_context(cylinder_bot_world, executor),
    )

    with no_execution:
        node.plan.perform()

    assert executor.requests == []


def test_interrupt_while_paused_prevents_remaining_physical_actions(
    cylinder_bot_world,
):
    executor = RecordedExecution()
    root = sequential(
        [
            WeighSampleAction("sample"),
            OpenInstrumentAction(LaboratoryInstrument.CENTRIFUGE),
        ],
        context=make_context(cylinder_bot_world, executor),
    )
    observer = PausedDispatch()
    root.plan.node_callbacks.append(observer)
    worker = Thread(target=SimulatedExecution(root.plan).run, daemon=True)
    worker.start()
    try:
        assert observer.paused.wait(timeout=5)
        root.interrupt()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert executor.requests == []
        assert root.status == TaskStatus.INTERRUPTED
    finally:
        root.interrupt()
        worker.join(timeout=5)


def test_missing_simulation_environment_fails_before_physical_dispatch(
    cylinder_bot_world,
):
    executor = RecordedExecution()
    node = execute_single(
        WeighSampleAction("sample"),
        context=make_context(cylinder_bot_world, executor),
    )

    with pytest.raises(LaboratoryActionFailed):
        node.plan.perform()

    assert executor.requests == []
    assert node.status == TaskStatus.FAILED


def test_action_records_hold_individual_measured_start_and_end_states(
    cylinder_bot_world: World,
) -> None:
    executor = MeasuredProgress(cylinder_bot_world)
    context = LaboratoryContext(
        world=cylinder_bot_world,
        robot=cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0],
        executor=executor,
    )
    root = sequential(
        [
            TransferSampleAction("sample", SampleDestination.SCALE),
            TransferSampleAction("sample", SampleDestination.RACK),
        ],
        context=context,
    )

    with simulated_robot_advanced:
        root.plan.perform()

    for index, action in enumerate(root.plan.actions):
        recorded = action.execution_data
        np.testing.assert_array_equal(
            recorded.execution_start_world_state, executor.starts[index]
        )
        np.testing.assert_array_equal(
            recorded.execution_end_world_state, executor.ends[index]
        )
        assert not np.shares_memory(
            recorded.execution_start_world_state, context.world.state._data
        )
        assert not np.shares_memory(
            recorded.execution_end_world_state, context.world.state._data
        )
