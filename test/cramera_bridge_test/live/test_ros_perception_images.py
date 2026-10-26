"""
Reading the images RoboKudo's pipelines publish, in a ROS context the listener owns.
"""

from __future__ import annotations

import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.context import Context
from sensor_msgs.msg import Image

from cramera_bridge.live.ros_perception_images import (
    RoboKudoImageListener,
    image_title,
)
from cramera_protocol.bundle.scene import PerceptionImageSource

TOPIC = "/cramera_test_pipeline/ClusterColorAnnotator/output_image"
"""
The topic the test publishes on, named as a RoboKudo visualizer names it.
"""

DELIVERY_TIMEOUT_SECONDS = 10.0
"""
How long the test waits for a published image to arrive.
"""

POLL_SECONDS = 0.05
"""
How often the test publishes and looks whether the image arrived.
"""


def red_pixel() -> Image:
    """
    :return: An image of one red pixel, encoded blue first.
    """
    return Image(height=1, width=1, encoding="bgr8", step=3, data=bytes([0, 0, 255]))


def test_an_image_published_while_a_step_ran_is_one_of_its_images():
    listener = RoboKudoImageListener().start()
    sender_context = Context()
    rclpy.init(context=sender_context)
    node = rclpy.create_node("cramera_test_robokudo", context=sender_context)
    publisher = node.create_publisher(Image, TOPIC, 1)
    before = listener.images(None, None, [])
    listener.observe_started()
    deadline = time.monotonic() + DELIVERY_TIMEOUT_SECONDS
    waiting = threading.Event()
    seen = []
    while not seen and time.monotonic() < deadline:
        publisher.publish(red_pixel())
        waiting.wait(POLL_SECONDS)
        seen = listener.images(None, None, [])
    listener.observe_started()
    after_next_start = listener.images(None, None, [])
    thread = listener.thread
    listener.stop()
    node.destroy_node()
    sender_context.try_shutdown()
    assert before == []
    (image,) = seen
    assert (image.title, image.source) == (
        "ClusterColorAnnotator",
        PerceptionImageSource.ROBOKUDO,
    )
    assert image.png.startswith(b"\x89PNG")
    assert after_next_start == []
    assert not thread.is_alive()
    assert listener.context.ok() is False


def test_an_image_is_titled_by_its_annotator_else_its_pipeline():
    assert image_title(TOPIC) == "ClusterColorAnnotator"
    assert image_title("/demo/output_image") == "demo"
