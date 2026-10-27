"""
The statecharts a recording went through, in the bundle's ``statecharts.json`` form.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Dict, List, Optional, Tuple

from cramera_protocol.bundle.statecharts import (
    Chart,
    ChartNode,
    Moment,
    Statecharts,
)
from cramera_protocol.live_messages import LiveStatechart

NO_MOMENT = -1
"""
The moment index of a frame during which no statechart was ticked.
"""

# %% collection


@dataclass
class RecordedStatecharts:
    """
    Collects each chart's structure once, each distinct chart state once, and which
    state every frame showed.
    """

    charts: List[Chart] = field(default_factory=list)
    """
    The charts, each once.
    """

    moments: List[Moment] = field(default_factory=list)
    """
    The distinct chart states.
    """

    frames: List[int] = field(default_factory=list)
    """
    The moment each frame showed.
    """

    chart_positions: Dict[str, int] = field(default_factory=dict)
    """
    The position of each chart by its signature.
    """

    moment_positions: Dict[Tuple[int, Tuple[str, ...], Tuple[str, ...]], int] = field(
        default_factory=dict
    )
    """
    The position of each moment by its chart and node states.
    """

    @classmethod
    def of_frames(
        cls, statecharts: List[Optional[LiveStatechart]]
    ) -> RecordedStatecharts:
        """
        :param statecharts: The statechart of every frame, or None for frames without.
        :return: The collected statecharts.
        """
        recorded = cls()
        for statechart in statecharts:
            recorded.add(statechart)
        return recorded

    def add(self, statechart: Optional[LiveStatechart]) -> None:
        """
        :param statechart: The statechart of the next frame.
        """
        if statechart is None:
            self.frames.append(NO_MOMENT)
            return
        chart = self.chart_position(statechart)
        life_cycles = tuple(node.life_cycle for node in statechart.nodes)
        observations = tuple(node.observation for node in statechart.nodes)
        key = (chart, life_cycles, observations)
        if key not in self.moment_positions:
            self.moment_positions[key] = len(self.moments)
            self.moments.append(
                Moment(
                    chart=chart,
                    life_cycles=list(life_cycles),
                    observations=list(observations),
                )
            )
        self.frames.append(self.moment_positions[key])

    def chart_position(self, statechart: LiveStatechart) -> int:
        """
        :param statechart: A statechart.
        :return: The position of its chart, added on first sight.
        """
        if statechart.signature not in self.chart_positions:
            self.chart_positions[statechart.signature] = len(self.charts)
            self.charts.append(
                Chart(
                    signature=statechart.signature,
                    title=statechart.title,
                    nodes=[
                        ChartNode(
                            identifier=node.identifier,
                            name=node.name,
                            class_name=node.class_name,
                            parent=node.parent,
                        )
                        for node in statechart.nodes
                    ],
                    edges=list(statechart.edges),
                )
            )
        return self.chart_positions[statechart.signature]

    def is_empty(self) -> bool:
        """
        :return: Whether no frame showed a statechart.
        """
        return not self.charts

    def document(self) -> Statecharts:
        """
        :return: The statecharts as the bundle stores them.
        """
        return Statecharts(charts=self.charts, moments=self.moments, frames=self.frames)
