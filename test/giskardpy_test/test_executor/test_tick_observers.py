"""
Tick observers are told about every tick of an executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from giskardpy.executor import Executor, ExecutorTickObserver
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstTrueNode,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointState,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection

# %% observers


@dataclass
class TickRecorder(ExecutorTickObserver):
    """
    Remembers the control cycle of every tick it is told about.
    """

    control_cycles: list[float] = field(default_factory=list)
    """
    The executor's control cycle at each notification, in notification order.
    """

    def on_tick(self, executor: Executor) -> None:
        self.control_cycles.append(executor.control_cycles)


@dataclass
class JointPositionRecorder(ExecutorTickObserver):
    """
    Remembers the joint positions of the executor's world at every tick it is told
    about.
    """

    positions: list[np.ndarray] = field(default_factory=list)
    """
    A copy of the world's joint positions at each notification, in notification order.
    """

    def on_tick(self, executor: Executor) -> None:
        self.positions.append(executor.context.world.state.positions.copy())


# %% fixtures


@pytest.fixture
def idle_executor() -> Executor:
    """
    An executor whose chart has settled, so that further ticks change no state.
    """
    chart = MotionStatechart()
    chart.add_node(ConstTrueNode())
    executor = Executor(MotionStatechartContext(world=World()))
    executor.compile(chart)
    executor.tick()
    return executor


# %% notification


def test_observer_is_told_about_every_tick_even_without_state_change(
    idle_executor,
):
    """
    A tick that changes no state of the chart is reported like every other tick.
    """
    recorder = TickRecorder()
    idle_executor.tick_observers.append(recorder)
    recorded_states = len(idle_executor.motion_statechart.history)
    first_control_cycle = idle_executor.control_cycles

    for _ in range(3):
        idle_executor.tick()

    assert len(idle_executor.motion_statechart.history) == recorded_states
    assert recorder.control_cycles == [
        first_control_cycle + 1,
        first_control_cycle + 2,
        first_control_cycle + 3,
    ]


def test_removed_observer_is_not_told_about_later_ticks(idle_executor):
    """
    An observer taken off the executor hears of no tick after its removal.
    """
    recorder = TickRecorder()
    idle_executor.tick_observers.append(recorder)
    idle_executor.tick()
    told_control_cycles = list(recorder.control_cycles)

    idle_executor.tick_observers.remove(recorder)
    idle_executor.tick()

    assert recorder.control_cycles == told_control_cycles


def test_observer_sees_the_world_moved_by_the_tick(prismatic_bot: World):
    """
    An observer is told only after the tick has applied its command to the world.
    """
    [connection] = prismatic_bot.get_connections_by_type(PrismaticConnection)
    chart = MotionStatechart()
    chart.add_node(
        JointPositionList(goal_state=JointState.from_mapping({connection: 0.5}))
    )
    executor = Executor(MotionStatechartContext(world=prismatic_bot))
    executor.compile(chart)
    recorder = JointPositionRecorder()
    executor.tick_observers.append(recorder)
    positions_before_tick = prismatic_bot.state.positions.copy()

    executor.tick()

    assert not np.array_equal(prismatic_bot.state.positions, positions_before_tick)
    [told_positions] = recorder.positions
    assert np.array_equal(told_positions, prismatic_bot.state.positions)
