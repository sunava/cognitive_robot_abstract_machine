"""
Which marker topics are read, and which the ROS graph offers.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import List, Mapping

from cramera_protocol.live_messages import MarkerTopics

TOPIC_SEPARATOR = ","
"""
Separates the topics of :attr:`MarkerEnvironmentVariable.TOPICS`.
"""


class MarkerEnvironmentVariable(StrEnum):
    """
    The environment variables configuring the marker overlay.
    """

    TOPICS = "CRAMERA_MARKER_TOPICS"
    """
    The topics read at start, separated by commas.
    """


class DefaultMarkerTopic(StrEnum):
    """
    The topics the CRAM packages publish their debug markers on.
    """

    SEMANTIC_WORLD = "/semworld/viz_marker"
    """
    The semantic digital twin's markers.
    """

    CORAPLEX = "/coraplex/viz_marker"
    """
    Coraplex's markers.
    """


@dataclass(frozen=True)
class MarkerTopicConfiguration:
    """
    The marker topics read at start.
    """

    environment: Mapping[str, str]
    """
    The process environment.
    """

    def topics(self) -> List[str]:
        """
        :return: The configured topics, or the CRAM defaults.
        """
        configured = self.environment.get(MarkerEnvironmentVariable.TOPICS, "")
        topics = [
            topic.strip()
            for topic in configured.split(TOPIC_SEPARATOR)
            if topic.strip()
        ]
        return topics or [topic.value for topic in DefaultMarkerTopic]

    @classmethod
    def of_process(cls) -> MarkerTopicConfiguration:
        """
        :return: The configuration of this process's environment.
        """
        return cls(dict(os.environ))


# %% topic source


class MarkerTopicSource(ABC):
    """
    Delivers ``MarkerArray`` messages of the topics it is told to read.
    """

    @abstractmethod
    def subscribe(self, topic: str) -> None:
        """
        :param topic: A topic to read; reading it twice changes nothing.
        """

    @abstractmethod
    def unsubscribe(self, topic: str) -> None:
        """
        :param topic: A topic to stop reading.
        """

    @abstractmethod
    def subscribed_topics(self) -> List[str]:
        """
        :return: The topics read, sorted.
        """

    @abstractmethod
    def available_topics(self) -> List[str]:
        """
        :return: The ``MarkerArray`` topics the ROS graph offers, sorted.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Stop reading and release every resource.
        """

    def topics(self) -> MarkerTopics:
        """
        :return: The topics read and on offer.
        """
        subscribed = self.subscribed_topics()
        return MarkerTopics(
            ros=True,
            subscribed=subscribed,
            available=sorted(set(self.available_topics()) | set(subscribed)),
        )
