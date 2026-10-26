"""
Which marker topics are read at start.
"""

from __future__ import annotations

from cramera_bridge.live.marker_topics import (
    TOPIC_SEPARATOR,
    DefaultMarkerTopic,
    MarkerEnvironmentVariable,
    MarkerTopicConfiguration,
)

CONFIGURED_TOPICS = ["/first/markers", "/second/markers"]
"""
Topics a process configures.
"""


def test_without_configuration_the_cram_topics_are_read():
    assert MarkerTopicConfiguration({}).topics() == [
        topic.value for topic in DefaultMarkerTopic
    ]


def test_the_configured_topics_are_read():
    configured = f" {TOPIC_SEPARATOR} ".join(CONFIGURED_TOPICS) + TOPIC_SEPARATOR
    environment = {MarkerEnvironmentVariable.TOPICS.value: configured}
    assert MarkerTopicConfiguration(environment).topics() == CONFIGURED_TOPICS


def test_the_process_environment_is_read(monkeypatch):
    monkeypatch.setenv(
        MarkerEnvironmentVariable.TOPICS, TOPIC_SEPARATOR.join(CONFIGURED_TOPICS)
    )
    assert MarkerTopicConfiguration.of_process().topics() == CONFIGURED_TOPICS
