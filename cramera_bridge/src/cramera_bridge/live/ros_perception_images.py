"""
Reading the images RoboKudo's pipelines publish from ROS.

RoboKudo's visualizer publishes what each annotator drew on
``<pipeline>/<annotator>/output_image`` and the active one's on
``<pipeline>/output_image``. Imported only where ROS is installed; the listener runs in
a ROS context of its own, so it neither depends on nor shuts down a context the demo
owns.
"""

from __future__ import annotations

import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription
from sensor_msgs.msg import Image
from typing_extensions import Callable, Dict, List, Optional, Sequence, Tuple

from cramera_bridge.live.perception_images import (
    PerceptionCamera,
    PngImage,
    RawImage,
    SeenImage,
)
from cramera_protocol.bundle.scene import PerceptionImageSource
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

NODE_NAME = "cramera_perception_images"
"""
The name of the listener's ROS node.
"""

IMAGE_TYPE = "sensor_msgs/msg/Image"
"""
The ROS type name of an image.
"""

OUTPUT_IMAGE_SUFFIX = "/output_image"
"""
How the name of every topic a RoboKudo visualizer publishes an image on ends.
"""

QUEUE_DEPTH = 1
"""
How many images a subscription buffers; only the newest is kept.
"""

DISCOVERY_SECONDS = 2.0
"""
How often the listener looks for image topics a pipeline started.
"""

SPIN_TIMEOUT_SECONDS = 0.1
"""
How long one spin waits for a message, which bounds how long stopping takes.
"""

JOIN_TIMEOUT_SECONDS = 2.0
"""
How long stopping waits for the spin thread.
"""


def image_title(topic: str) -> str:
    """
    :param topic: A topic a RoboKudo visualizer publishes on.
    :return: What its images show: the annotator that drew them, else the pipeline.
    """
    return topic.removesuffix(OUTPUT_IMAGE_SUFFIX).rsplit("/", 1)[-1] or topic


@dataclass
class RoboKudoImageListener(PerceptionCamera):
    """
    Subscribes to every image topic of RoboKudo's visualizers and keeps the newest image
    of each; a perception step's images are those that arrived while it ran.
    """

    clock: Callable[[], float] = time.monotonic
    """
    The monotonic clock arrivals are timed with.
    """

    context: Context = field(default_factory=Context, init=False)
    """
    The ROS context the listener owns.
    """

    node: Optional[Node] = field(default=None, init=False)
    """
    The listener's node, while started.
    """

    executor: Optional[SingleThreadedExecutor] = field(default=None, init=False)
    """
    The executor delivering messages, while started.
    """

    thread: Optional[threading.Thread] = field(default=None, init=False)
    """
    The thread spinning the executor, while started.
    """

    stopping: threading.Event = field(default_factory=threading.Event, init=False)
    """
    Set when the spin thread is to end.
    """

    subscriptions: Dict[str, Subscription] = field(default_factory=dict, init=False)
    """
    The subscriptions by topic.
    """

    newest: Dict[str, Tuple[float, RawImage]] = field(default_factory=dict, init=False)
    """
    The newest image of every topic, with when it arrived.
    """

    step_started: float = field(default=0.0, init=False)
    """
    When the last perception step started.
    """

    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    """
    Guards :attr:`subscriptions`, :attr:`newest` and :attr:`step_started`.
    """

    def start(self) -> RoboKudoImageListener:
        """
        Create the context and node, subscribe to the image topics there are and start
        spinning; a start that fails part-way releases what it created.

        :return: This listener.
        """
        rclpy.init(context=self.context)
        with ExitStack() as starting:
            starting.callback(self.stop)
            self.node = rclpy.create_node(NODE_NAME, context=self.context)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.discover()
            self.thread = threading.Thread(
                target=self.spin, name=NODE_NAME, daemon=True
            )
            self.thread.start()
            starting.pop_all()
        return self

    def spin(self) -> None:
        """
        Deliver images, and look for new image topics now and then, until the listener
        stops.
        """
        discovered = self.clock()
        while not self.stopping.is_set() and self.context.ok():
            self.executor.spin_once(timeout_sec=SPIN_TIMEOUT_SECONDS)
            if self.clock() - discovered >= DISCOVERY_SECONDS:
                self.discover()
                discovered = self.clock()

    def image_topics(self) -> List[str]:
        """
        :return: Every topic a RoboKudo visualizer publishes images on.
        """
        return sorted(
            name
            for name, types in self.node.get_topic_names_and_types()
            if IMAGE_TYPE in types and name.endswith(OUTPUT_IMAGE_SUFFIX)
        )

    def discover(self) -> None:
        """
        Subscribe to the image topics not subscribed yet.
        """
        for topic in self.image_topics():
            with self.lock:
                if topic in self.subscriptions:
                    continue
                self.subscriptions[topic] = self.node.create_subscription(
                    Image, topic, self.delivery(topic), QoSProfile(depth=QUEUE_DEPTH)
                )

    def delivery(self, topic: str) -> Callable[[Image], None]:
        """
        :param topic: A topic.
        :return: The callback keeping the topic's newest image.
        """

        def deliver(message: Image) -> None:
            image = RawImage(
                encoding=message.encoding,
                width=message.width,
                height=message.height,
                step=message.step,
                data=bytes(message.data),
            )
            with self.lock:
                self.newest[topic] = (self.clock(), image)

        return deliver

    def observe_started(self) -> None:
        with self.lock:
            self.step_started = self.clock()

    def images(
        self, world: World, robot: AbstractRobot, perceived: Sequence[Body]
    ) -> List[SeenImage]:
        with self.lock:
            arrived = [
                (topic, image)
                for topic, (when, image) in sorted(self.newest.items())
                if when >= self.step_started
            ]
        seen = []
        for topic, image in arrived:
            pixels = image.rgb()
            if pixels is not None:
                seen.append(
                    SeenImage(
                        image_title(topic),
                        PerceptionImageSource.ROBOKUDO,
                        PngImage.of_rgb(pixels),
                    )
                )
        return seen

    def stop(self) -> None:
        """
        Stop spinning, join the thread, destroy the subscriptions and the node, and shut
        down the listener's own context.
        """
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=JOIN_TIMEOUT_SECONDS)
            self.thread = None
        with self.lock:
            subscriptions = list(self.subscriptions.values())
            self.subscriptions.clear()
        if self.node is not None:
            for subscription in subscriptions:
                self.node.destroy_subscription(subscription)
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        self.context.try_shutdown()
