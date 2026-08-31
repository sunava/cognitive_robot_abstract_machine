"""
A running demo's segmind detections, offered to the viewer's queries.
"""

from __future__ import annotations

from dataclasses import dataclass

from segmind.event_logger import EventLogger
from typing_extensions import List

from cramera.knowledge.detected_events import EVENT_VARIABLE, DetectedEventRecord
from cramera.knowledge.presets import DetectedEventQuestions, Preset
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.queryable_knowledge import QueryableKnowledge, QueryScope


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
        return self._questions().listed()

    def unlisted_presets(self) -> List[Preset]:
        """
        The questions recognized when one is asked, but not shown as buttons.
        """
        return self._questions().unlisted()

    def _questions(self) -> DetectedEventQuestions:
        """
        The questions this demo's detectors give rise to, one per type they can produce.
        """
        return DetectedEventQuestions(DetectedEventRecord.recordable_event_types())
