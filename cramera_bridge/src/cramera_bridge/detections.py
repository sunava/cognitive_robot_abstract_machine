"""
Segmind's event detectors ticked along a run, so a recording carries what they saw.

The detectors run on the thread that changes the world, through a state-change callback:
segmind's expressions are CasADi-backed and must not be evaluated from a second thread
while the planner builds expressions of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, List, Optional

from cramera_bridge.attachment import AttachmentDetector
from cramera_protocol.bundle.scene import DetectedEvent
from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import EventWithTrackedObjects
from segmind.detectors.base import AbstractDetector, SegmindContext
from segmind.detectors.coarse_event_detector_nodes import (
    PickUpDetector,
    PlacingDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.world import World

EVENT_NAME_SEPARATOR = " "
"""
What separates the object's name from the event type in an event's name.
"""

# %% detection


@dataclass
class DetectionRecorder:
    """
    Segmind's detectors compiled against one world and ticked whenever its state
    changes, from :meth:`start` until :meth:`stop`.
    """

    world: World
    """
    The world whose bodies the detectors watch.
    """

    executor: Optional[EpisodeSegmenterExecutor] = field(default=None, init=False)
    """
    The executor ticking the detectors, once started.
    """

    context: Optional[SegmindContext] = field(default=None, init=False)
    """
    Where the detectors log their events, once started.
    """

    callback: Optional[DetectionTicks] = field(default=None, init=False)
    """
    The callback ticking the detectors, while started.
    """

    @staticmethod
    def detectors() -> List[AbstractDetector]:
        """
        :return: Every detector segmind offers, with picking up and putting down read
            from the attachment itself rather than inferred from motion, which would
            give one event per surface an object touched instead of one per grasp.
        """
        inferred = (PickUpDetector, PlacingDetector)
        return [
            detector
            for detector in SegmindStatechart().build_statechart().nodes
            if not isinstance(detector, inferred)
        ] + [AttachmentDetector()]

    def start(self) -> DetectionRecorder:
        """
        Compile the detectors and tick them whenever the world's state changes.

        :return: This recorder.
        """
        executor = EpisodeSegmenterExecutor(
            context=MotionStatechartContext(world=self.world)
        )
        self.context = executor.context.require_extension(SegmindContext)
        executor.compile(SegmindStatechart().build_statechart(self.detectors()))
        self.executor = executor
        self.callback = DetectionTicks(_world=self.world, recorder=self)
        return self

    def tick(self) -> None:
        """
        Let the detectors look at the world as it stands now.
        """
        self.executor.tick()

    def stop(self) -> None:
        """
        Stop ticking the detectors; what they detected stays readable.
        """
        if self.callback is not None:
            self.callback.stop()
            self.callback = None

    def events(self) -> List[DetectedEvent]:
        """
        :return: Everything detected so far, oldest first; nothing before
            :meth:`start`.
        """
        if self.context is None:
            return []
        return [self.describe(event) for event in self.context.logger.get_events()]

    @staticmethod
    def describe(event: EventWithTrackedObjects) -> DetectedEvent:
        """
        :param event: An event a detector produced; every event segmind detects is
            about tracked objects.
        :return: The event as a scene bundle carries it, named by the full names of the
            bodies it is about.
        """
        event_type = type(event).__name__
        tracked = str(event.tracked_object.name)
        return DetectedEvent(
            name=EVENT_NAME_SEPARATOR.join((tracked, event_type)),
            event_type=event_type,
            timestamp=event.timestamp.isoformat(),
            tracked_object=tracked,
            with_object=(
                None if event.with_object is None else str(event.with_object.name)
            ),
        )


@dataclass(eq=False)
class DetectionTicks(StateChangeCallback):
    """
    Ticks a recorder's detectors whenever the world's state changes.
    """

    recorder: DetectionRecorder = field(kw_only=True)
    """
    The recorder.
    """

    def on_state_change(self, **kwargs: Any) -> None:
        self.recorder.tick()
