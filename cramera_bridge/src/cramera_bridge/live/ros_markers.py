"""
Reading debug markers from ROS.

Imported only where ROS is installed; the listener runs in a ROS context of its own, so
it neither depends on nor shuts down a context the demo owns.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from dataclasses import dataclass, field

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.subscription import Subscription
from typing_extensions import Callable, Dict, List, Optional
from visualization_msgs.msg import MarkerArray

from cramera_bridge.live.marker_topics import MarkerTopicSource
from cramera_bridge.live.markers import MarkerBoard

NODE_NAME = "cramera_markers"
"""
The name of the listener's ROS node.
"""

QUEUE_DEPTH = 100
"""
How many marker arrays a subscription buffers.
"""

SPIN_TIMEOUT_SECONDS = 0.1
"""
How long one spin waits for a message, which bounds how long stopping takes.
"""

JOIN_TIMEOUT_SECONDS = 2.0
"""
How long stopping waits for the spin thread.
"""

MARKER_ARRAY_TYPE = "visualization_msgs/msg/MarkerArray"
"""
The ROS type name of a marker array.
"""

# %% listener


@dataclass
class RosMarkerListener(MarkerTopicSource):
    """
    Subscribes to marker topics and hands every array to a :class:`MarkerBoard`.
    """

    board: MarkerBoard
    """
    The board receiving the markers.
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

    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    """
    Guards :attr:`subscriptions`.
    """

    def start(self, topics: List[str]) -> RosMarkerListener:
        """
        Create the context, node and subscriptions and start spinning; a start that
        fails part-way releases what it created.

        :param topics: The topics read from the start.
        :return: This listener.
        """
        rclpy.init(context=self.context)
        with ExitStack() as starting:
            starting.callback(self.stop)
            self.node = rclpy.create_node(NODE_NAME, context=self.context)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            for topic in topics:
                self.subscribe(topic)
            self.thread = threading.Thread(
                target=self.spin, name=NODE_NAME, daemon=True
            )
            self.thread.start()
            starting.pop_all()
        return self

    def spin(self) -> None:
        """
        Deliver messages until the listener stops.
        """
        while not self.stopping.is_set() and self.context.ok():
            self.executor.spin_once(timeout_sec=SPIN_TIMEOUT_SECONDS)

    def delivery(self, topic: str) -> Callable[[MarkerArray], None]:
        """
        :param topic: A topic.
        :return: The callback handing the topic's arrays to the board.
        """

        def deliver(array: MarkerArray) -> None:
            self.board.observe(topic, list(array.markers))

        return deliver

    def subscribe(self, topic: str) -> None:
        with self.lock:
            if topic in self.subscriptions:
                return
            self.subscriptions[topic] = self.node.create_subscription(
                MarkerArray,
                topic,
                self.delivery(topic),
                QoSProfile(
                    depth=QUEUE_DEPTH, durability=DurabilityPolicy.TRANSIENT_LOCAL
                ),
            )

    def unsubscribe(self, topic: str) -> None:
        with self.lock:
            subscription = self.subscriptions.pop(topic, None)
        if subscription is not None:
            self.node.destroy_subscription(subscription)
        self.board.drop(topic)

    def subscribed_topics(self) -> List[str]:
        with self.lock:
            return sorted(self.subscriptions)

    def available_topics(self) -> List[str]:
        return sorted(
            name
            for name, types in self.node.get_topic_names_and_types()
            if MARKER_ARRAY_TYPE in types
        )

    def stop(self) -> None:
        """
        Stop spinning, join the thread, destroy the subscriptions and the node, and shut
        down the listener's own context.
        """
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=JOIN_TIMEOUT_SECONDS)
            self.thread = None
        for topic in self.subscribed_topics():
            self.unsubscribe(topic)
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        self.context.try_shutdown()
