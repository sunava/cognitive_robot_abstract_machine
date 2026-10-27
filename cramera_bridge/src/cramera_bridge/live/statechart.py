"""
The motion statechart being ticked, read from its native state history.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, List, Optional

from cramera_protocol.bundle.statecharts import ChartEdge
from cramera_protocol.live_messages import LiveChartNode, LiveStatechart
from giskardpy.motion_statechart.data_types import ObservationStateValues

if TYPE_CHECKING:
    from giskardpy.motion_statechart.graph_node import MotionStatechartNode
    from giskardpy.motion_statechart.motion_statechart import (
        MotionStatechart,
        StateHistoryItem,
    )

CHART_NODE_PREFIX = "chart_node_"
"""
The start of every chart node identity; the rest is the node's index in its chart.
"""

SIGNATURE_SEPARATOR = "|"
"""
Separates the nodes of a chart's signature.
"""

# %% structure


@dataclass(frozen=True)
class ChartStructure:
    """
    The nodes and transitions of one chart, which do not change while it is ticked.
    """

    chart: MotionStatechart
    """
    The chart.
    """

    @staticmethod
    def identifier(index: int) -> str:
        """
        :param index: The index of a node in its chart.
        :return: The node's identity.
        """
        return f"{CHART_NODE_PREFIX}{index}"

    def edges(self) -> List[ChartEdge]:
        """
        :return: The chart's transitions with their native kinds.
        """
        graph = self.chart.rx_graph
        return [
            ChartEdge(
                source=self.identifier(graph.get_node_data(source).index),
                destination=self.identifier(graph.get_node_data(destination).index),
                kind=transition.kind.name,
            )
            for source, destination, transition in graph.edge_index_map().values()
        ]

    def signature(self) -> str:
        """
        :return: The identity of the chart's structure.
        """
        return SIGNATURE_SEPARATOR.join(
            f"{self.identifier(node.index)}:{node.name}" for node in self.chart.nodes
        )


# %% state


@dataclass
class StatechartPublisher:
    """
    Publishes the chart being ticked in the state its history recorded last.
    """

    latest: Optional[LiveStatechart] = None
    """
    The newest snapshot, or None before a chart was ticked.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards :attr:`latest`.
    """

    def observe(self, chart: MotionStatechart, title: str) -> bool:
        """
        Read the chart's last recorded state.

        :param chart: The chart being ticked.
        :param title: What the chart executes.
        :return: Whether the published snapshot changed.
        """
        if not chart.history.history:
            return False
        moment = chart.history.history[-1]
        structure = ChartStructure(chart)
        snapshot = LiveStatechart(
            signature=structure.signature(),
            title=title,
            nodes=[self.node_state(node, moment) for node in chart.nodes],
            edges=structure.edges(),
        )
        with self.lock:
            if snapshot == self.latest:
                return False
            self.latest = snapshot
        return True

    @staticmethod
    def node_state(
        node: MotionStatechartNode, moment: StateHistoryItem
    ) -> LiveChartNode:
        """
        :param node: A node of the chart.
        :param moment: The chart's last recorded state.
        :return: The node with its native life-cycle and observation values.
        """
        parent = node.parent_node_index
        return LiveChartNode(
            identifier=ChartStructure.identifier(node.index),
            name=str(node.name),
            class_name=type(node).__name__,
            parent=None if parent is None else ChartStructure.identifier(parent),
            life_cycle=moment.life_cycle_state[node].name,
            observation=ObservationStateValues(
                float(moment.observation_state[node])
            ).name,
        )

    def current(self) -> Optional[LiveStatechart]:
        """
        :return: The newest snapshot, or None before a chart was ticked.
        """
        with self.lock:
            return self.latest
