"""
The callbacks of a plan are told about every tick of the plan's motion statecharts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.ros_executor import Ros2Executor
from semantic_digital_twin.datastructures.definitions import TorsoState

# %% callbacks


@dataclass
class MotionTickRecorder(PlanCallback):
    """
    Remembers the statechart of every motion tick it is told about.
    """

    statecharts: list[MotionStatechart] = field(default_factory=list)
    """
    The ticked statecharts in notification order.
    """

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        self.statecharts.append(statechart)


# %% performed plans


def test_performed_motion_tells_the_plan_callbacks_about_every_tick(
    pr2_apartment_context, monkeypatch
):
    """
    Every executor tick of a performed motion reaches the plan's callbacks, with the
    statechart the executor ticked.
    """
    world, robot, context = pr2_apartment_context
    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context).plan
    recorder = MotionTickRecorder(plan=plan)
    plan.node_callbacks.append(recorder)
    ticked_statecharts = []
    tick = Ros2Executor.tick

    def recorded_tick(executor: Ros2Executor) -> None:
        """
        Tick the executor and remember the statechart it ticked.
        """
        tick(executor)
        ticked_statecharts.append(executor.motion_statechart)

    monkeypatch.setattr(Ros2Executor, "tick", recorded_tick)

    with kinematically_simulated_robot:
        plan.perform()

    motion_statecharts = {
        id(node.motion_statechart)
        for node in plan.all_nodes
        if isinstance(node, MotionNode)
    }
    assert {id(statechart) for statechart in ticked_statecharts} == motion_statecharts
    assert [id(statechart) for statechart in recorder.statecharts] == [
        id(statechart) for statechart in ticked_statecharts
    ]
