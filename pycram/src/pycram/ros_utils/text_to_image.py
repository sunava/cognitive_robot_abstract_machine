from typing import Optional
import logging
import rclpy
import time
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy

logger = logging.getLogger(__name__)


class TextToImagePublisher:
    def __init__(self, topic_name: str = "/head_display/text_to_image"):
        self.is_init = False
        self.topic_name = topic_name
        self.node: Optional[Node] = None
        self.publisher: Optional[Publisher] = None
        self._init_interface()


    def _init_interface(self):
        """
        Initializes the ROS node and publisher once.
        """
        if self.is_init:
            return

        if not rclpy.ok():
            rclpy.init()

        reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.node = rclpy.create_node("text_publisher_node")
        self.publisher = self.node.create_publisher(String, self.topic_name, reliable_qos)

        self.first_screen()
        self.is_init = True

        # wait a few seconds to ensure the screen is initialized
        time.sleep(3)

        logger.info("TextImagePublisher initialized")


    def publish_text(self, text: str):
        """
        Publishes a text message to the display of HSR.
        """
        self._init_interface()

        msg = String()
        msg.data = text
        for _ in range(3):
            self.publisher.publish(msg)
        time.sleep(1)

        logger.info("Published new text to image display")


    def first_screen(self):
        """
        Triggers the first screen update.
        The initial image takes a few seconds to appear.
        """

        if not self.publisher:
            return

        msg = String()
        msg.data = ""

        # first few images are not shown
        for _ in range(6):
            self.publisher.publish(msg)
        time.sleep(1)


    def shutdown(self):
        """
        Shuts down the ROS node.
        """
        if self.node:
            self.node.destroy_node()

        logger.info("TextImagePublisher shut down")


def example_use():
    # create new Publisher like this
    text_pub = TextToImagePublisher()

    # display Text like this
    # if you spam text, it will only stay on the display for a second
    # maybe add a time.sleep() before publishing the next image
    text_pub.publish_text("Hello my name is Toya")
    time.sleep(1)
    text_pub.publish_text("What is your name?")
    time.sleep(1)
    text_pub.publish_text("This is a test")

example_use()