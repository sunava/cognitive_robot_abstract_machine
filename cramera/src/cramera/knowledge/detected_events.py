"""
What segmind detected while a demo ran, as something a query can range over.

A detector produces events about bodies of the running world; a question is asked about
them in the same language as everything else, so each event is flattened to the names
and the moment a question actually asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from segmind.datastructures.events import DetectionEvent, EventWithTrackedObjects
from segmind.event_logger import EventLogger
from typing_extensions import Any, Dict, List, Optional, Tuple, Type

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


class SceneField(StrEnum):
    """
    Key a recorded scene bundle carries its detections under.
    """

    DETECTED_EVENTS = "detectedEvents"


class EventField(StrEnum):
    """
    Key one detected event is written to a bundle under.
    """

    NAME = "name"
    EVENT_TYPE = "eventType"
    TIMESTAMP = "timestamp"
    TRACKED_OBJECT = "trackedObject"
    WITH_OBJECT = "withObject"


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

    def to_payload(self) -> Dict[str, Any]:
        """
        The record as a recorded bundle carries it, with the moment as an ISO instant.
        """
        return {
            EventField.NAME.value: self.name,
            EventField.EVENT_TYPE.value: self.event_type,
            EventField.TIMESTAMP.value: self.timestamp.isoformat(),
            EventField.TRACKED_OBJECT.value: self.tracked_object,
            EventField.WITH_OBJECT.value: self.with_object,
        }

    @classmethod
    def of_payload(cls, payload: Dict[str, Any]) -> DetectedEventRecord:
        """
        One record as a bundle carries it.

        :param payload: The event as :meth:`to_payload` wrote it.
        """
        return cls(
            name=payload[EventField.NAME.value],
            event_type=payload[EventField.EVENT_TYPE.value],
            timestamp=datetime.fromisoformat(payload[EventField.TIMESTAMP.value]),
            tracked_object=payload.get(EventField.TRACKED_OBJECT.value),
            with_object=payload.get(EventField.WITH_OBJECT.value),
        )

    @classmethod
    def of_scene(cls, scene: Dict[str, Any]) -> List[DetectedEventRecord]:
        """
        Every detection a recorded scene carries, or none for a scene recorded without
        detectors.

        :param scene: The scene bundle's ``scene.json`` content.
        """
        return [
            cls.of_payload(payload)
            for payload in scene.get(SceneField.DETECTED_EVENTS.value) or []
        ]

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
