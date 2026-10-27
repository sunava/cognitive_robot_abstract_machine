from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from coraplex.plans.plan_entity import PlanEntity

if TYPE_CHECKING:
    from coraplex.plans.plan_node import PlanNode
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart


@dataclass
class PlanCallback(PlanEntity):
    """
    Observe plan execution; unimplemented events leave execution unchanged.
    """

    def on_start(self, node: PlanNode) -> None:
        """
        Observe a node whose execution has begun.

        :param node: The started node.
        """

    def on_end(self, node: PlanNode) -> None:
        """
        Observe a node after its execution ends.

        :param node: The completed node, including its outcome.
        """

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        """
        Observe a motion statechart of the plan after each tick of its executor, once
        the tick's command has been applied to the world.

        :param statechart: The ticked statechart.
        """
