"""
Ticking segmind's detectors alongside a recorded run, so the bundle carries what they
saw.

The detectors run inline on the thread that ticks the executor: segmind's expressions
are CasADi-backed and reference counted without atomics, so detecting from a second
thread while the planner builds expressions of its own corrupts them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.detectors.base import SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.world import World
from typing_extensions import List, Optional

from cramera.knowledge.detected_events import DetectedEventRecord
from cramera.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionRecorder:
    """
    Segmind's detectors, ticked along one recorded run.

    Built against the world the run modifies and ticked once per executor tick, so every
    moment its detectors recognize is recorded in the order it happened.
    """

    world: World
    """
    The world whose bodies the detectors watch.
    """

    _executor: Optional[EpisodeSegmenterExecutor] = field(default=None, init=False)
    """
    The executor ticking the detector statechart, once it is compiled.
    """

    _context: Optional[SegmindContext] = field(default=None, init=False)
    """
    Where the detectors log what they saw.
    """

    def start(self) -> None:
        """
        Compile segmind's detectors against the world, ready to be ticked.

        Every detector segmind offers by default is compiled: what a run is worth asking
        about afterwards is not known while it is being recorded.
        """
        executor = EpisodeSegmenterExecutor(
            context=MotionStatechartContext(world=self.world)
        )
        self._context = executor.context.require_extension(SegmindContext)
        executor.compile(SegmindStatechart().build_statechart())
        self._executor = executor
        logger.info("detecting events with segmind's default detectors")

    def tick(self) -> None:
        """
        Let the detectors look at the world as it now stands.

        Does nothing before :meth:`start`, so a run recorded without detection simply
        never detects.
        """
        if self._executor is not None:
            self._executor.tick()

    def records(self) -> List[DetectedEventRecord]:
        """
        Everything detected so far, oldest first, as the bundle carries it.
        """
        if self._context is None:
            return []
        return DetectedEventRecord.of_events(list(self._context.logger.get_events()))
