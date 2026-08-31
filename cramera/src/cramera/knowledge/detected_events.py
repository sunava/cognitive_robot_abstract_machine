"""
What segmind detected while a demo ran, as something a query can range over.

A detector produces events about bodies of the running world; a question is asked about
them in the same language as everything else, so each event is flattened to the names
and the moment a question actually asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from segmind.datastructures.events import DetectionEvent, EventWithTrackedObjects
from segmind.event_logger import EventLogger
from typing_extensions import List, Optional, Tuple, Type

from cramera.knowledge.presets import Preset, PresetsPerType
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.queryable_knowledge import QueryableKnowledge, QueryScope

EVENT_VARIABLE = "event"
"""
Name a question binds one detected event to.
"""

EVENT_CLASS_SUFFIX = "Event"
"""
The word every detected event's class name ends in, which is what one of them is called
when a question asks for it out loud.
"""


@dataclass(frozen=True)
class DetectedEventRecord:
    """
    One event segmind detected, flattened to what a query asks about it.
    """

    name: str
    """
    The event's label, e.g. ``"milk PickUpEvent"``.
    """

    event_type: str
    """
    The detected event's own type.
    """

    timestamp: datetime
    """
    When the event was detected.
    """

    tracked_object: Optional[str] = None
    """
    Name of the primary body the event was detected on, or None for an event about no
    body in particular.
    """

    with_object: Optional[str] = None
    """
    Name of the second body the event relates the first one to, if any.
    """

    @classmethod
    def of_event(cls, event: DetectionEvent) -> DetectedEventRecord:
        """
        One detected event as a record.

        :param event: The event a detector produced.
        """
        tracked, second = cls._involved_names(event)
        return cls(
            name=" ".join(part for part in [tracked, type(event).__name__] if part),
            event_type=type(event).__name__,
            timestamp=event.timestamp,
            tracked_object=tracked,
            with_object=second,
        )

    @classmethod
    def of_events(cls, events: List[DetectionEvent]) -> List[DetectedEventRecord]:
        """
        Every detected event as a record, in the order they were detected.

        :param events: The events a detector produced.
        """
        return [cls.of_event(event) for event in events]

    @classmethod
    def recordable_event_types(cls) -> Tuple[str, ...]:
        """
        The type of every event a record can be written for, in alphabetical order.

        A type that only serves as the base of others is left out: no detector produces
        one, so no question can ask for it. Only the types this process has imported are
        found, which for a running demo is every type its own detectors can produce.
        """
        return tuple(
            sorted(event_type.__name__ for event_type in cls._detectable_types())
        )

    @classmethod
    def _detectable_types(cls) -> List[Type[DetectionEvent]]:
        """
        Every event type a detector can produce: the leaves of segmind's event tree.
        """

        def leaves(base: Type[DetectionEvent]) -> List[Type[DetectionEvent]]:
            found: List[Type[DetectionEvent]] = []
            for subclass in base.__subclasses__():
                found.extend(
                    leaves(subclass) if subclass.__subclasses__() else [subclass]
                )
            return found

        return leaves(DetectionEvent)

    @staticmethod
    def _involved_names(event: DetectionEvent) -> Tuple[Optional[str], Optional[str]]:
        """
        The names of the bodies an event happened to, the primary one first.

        :param event: The event a detector produced.
        """
        if not isinstance(event, EventWithTrackedObjects):
            return None, None
        second = event.with_object
        return (
            event.tracked_object.name.name,
            second.name.name if second is not None else None,
        )


@dataclass
class DetectedEvents:
    """
    A running demo's detections, as a body of knowledge questions can be asked of.

    Any demo that ticks segmind detectors offers this alongside its own knowledge, so
    what a question about the detections may name does not have to be written twice.
    """

    logger: EventLogger
    """
    The event logger the demo's detectors write their events to.
    """

    def knowledge(self) -> QueryableKnowledge:
        """
        What a question about the detections may range over.

        Read fresh on every call, so an answer names every moment detected up to now.
        """
        return QueryableKnowledge(
            scope=QueryScope.DETECTED_EVENTS,
            domains=[
                QueryDomain(
                    EVENT_VARIABLE,
                    DetectedEventRecord,
                    self.records(),
                )
            ],
        )

    def records(self) -> List[DetectedEventRecord]:
        """
        Everything detected so far, oldest first.
        """
        with self.logger.timeline_lock:
            return DetectedEventRecord.of_events(list(self.logger.timeline))

    def presets(self) -> List[Preset]:
        """
        The ready-made questions the panel offers as buttons.
        """
        return [
            Preset(
                "what was detected, and when?",
                "set_of(event.name, event.event_type, event.timestamp)",
                scope=QueryScope.DETECTED_EVENTS,
            )
        ]

    def unlisted_presets(self) -> List[Preset]:
        """
        "Give me all pick up events", written out for every type of event a detector can
        produce -- more questions than a panel has room to show as buttons.
        """
        return PresetsPerType(
            class_suffix=EVENT_CLASS_SUFFIX,
            class_names=DetectedEventRecord.recordable_event_types(),
            code="an(entity(%s).where(%s.event_type == '%%s'))"
            % (EVENT_VARIABLE, EVENT_VARIABLE),
            scope=QueryScope.DETECTED_EVENTS,
        ).questions()
