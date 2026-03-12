import os
from xml.etree import ElementTree as ET

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)
DEFAULT_ENVIRONMENT_PATH = os.path.join(
    "/home/vee",
    "worlds",
    "tmc_gazebo",
    "tmc_gazebo_worlds",
    "worlds",
    "apartment-hsr.urdf",
)


def resolve_demo_environment_path():
    return os.environ.get("THESIS_NEW_ENVIRONMENT", DEFAULT_ENVIRONMENT_PATH)


def load_demo_world(environment_path):
    with open(environment_path, "r") as environment_file:
        xml_text = environment_file.read()

    root = ET.fromstring(xml_text)
    if root.tag.lower() == "sdf":
        raise ValueError(
            f"{environment_path} is an SDF world file, not a URDF. "
            "Convert it to URDF first or use a real URDF environment file."
        )

    return URDFParser(urdf=xml_text).parse()


# %% Environment Setup
environment_path = resolve_demo_environment_path()
world = load_demo_world(environment_path)


# %% Visualization
try:
    import rclpy

    rclpy.init()
    rclpy_node = rclpy.create_node("ros_node")
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)
    viz.with_tf_publisher()
except ImportError:
    pass

print(world.root)
while True:
    pass
