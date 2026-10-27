"""
Reading markers from ROS in a context the listener owns.
"""

from __future__ import annotations

import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.context import Context
from rclpy.exceptions import InvalidTopicNameException
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from cramera_bridge.live.markers import MarkerBoard
from cramera_bridge.live.ros_markers import QUEUE_DEPTH, RosMarkerListener

TOPIC = "/cramera_test/markers"
"""
The topic the test publishes on.
"""

DELIVERY_TIMEOUT_SECONDS = 10.0
"""
How long the test waits for a published marker to arrive.
"""

POLL_SECONDS = 0.05
"""
How often the test looks whether the marker arrived.
"""

# %% listener


def test_a_published_marker_reaches_the_board_and_stopping_releases_everything():
    board = MarkerBoard()
    listener = RosMarkerListener(board).start([TOPIC])
    sender_context = Context()
    rclpy.init(context=sender_context)
    node = rclpy.create_node("cramera_test_sender", context=sender_context)
    publisher = node.create_publisher(
        MarkerArray,
        TOPIC,
        QoSProfile(depth=QUEUE_DEPTH, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    marker = Marker()
    marker.ns = "test"
    marker.type = Marker.SPHERE
    publisher.publish(MarkerArray(markers=[marker]))
    deadline = time.monotonic() + DELIVERY_TIMEOUT_SECONDS
    waiting = threading.Event()
    while not board.stores and time.monotonic() < deadline:
        waiting.wait(POLL_SECONDS)
    delivered = list(board.stores)
    available = listener.available_topics()
    thread = listener.thread
    listener.stop()
    node.destroy_node()
    sender_context.try_shutdown()
    assert delivered == [TOPIC]
    assert list(board.stores) == []
    assert TOPIC in available
    assert not thread.is_alive()
    assert listener.context.ok() is False
    assert rclpy.ok() is False


def test_topics_are_added_once_and_dropped():
    board = MarkerBoard()
    listener = RosMarkerListener(board).start([TOPIC])
    listener.subscribe(TOPIC)
    subscribed = listener.subscribed_topics()
    listener.unsubscribe(TOPIC)
    remaining = listener.topics().subscribed
    listener.stop()
    assert subscribed == [TOPIC]
    assert remaining == []


INVALID_TOPIC = "not a topic"
"""
A topic name ROS refuses, which fails a start after the node exists.
"""


def test_a_failed_start_releases_the_node_and_the_context():
    listener = RosMarkerListener(MarkerBoard())
    with pytest.raises(InvalidTopicNameException):
        listener.start([INVALID_TOPIC])
    assert listener.node is None
    assert listener.executor is None
    assert listener.context.ok() is False
