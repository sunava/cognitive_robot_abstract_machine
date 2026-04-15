import os

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.justin import Justin
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    OmniDrive,
)
from semantic_digital_twin.world_description.utils import world_with_urdf_factory

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)
DEFAULT_ROBOT_START_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(0.0, 0.0, 0.0)
ARMAR7_START_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(0.0, 0.0, 0.0)

ROBOT_SPECS = {
    "pr2": (
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro",
        PR2,
        OmniDrive,
        DEFAULT_ROBOT_START_POSE,
    ),
    "hsrb": (
        os.path.join(RESOURCES_DIR, "robots", "hsrb.urdf"),
        HSRB,
        OmniDrive,
        DEFAULT_ROBOT_START_POSE,
    ),
    "stretch": (
        os.path.join(RESOURCES_DIR, "robots", "stretch_description.urdf"),
        Stretch,
        DifferentialDrive,
        DEFAULT_ROBOT_START_POSE,
    ),
    "tiago": (
        "package://iai_tiago_description/urdf/tiago_from_our_robot.urdf",
        Tiago,
        DifferentialDrive,
        DEFAULT_ROBOT_START_POSE,
    ),
    "armar7": (
        "package://iai_kit_armar7/urdf/Armar7.urdf",
        Armar7,
        OmniDrive,
        ARMAR7_START_POSE,
    ),
    "justin": (
        "package://iai_dlr_rollin_justin/urdf/rollin_justin.urdf",
        Justin,
        OmniDrive,
        ARMAR7_START_POSE,
    ),
    "g1": (
        "package://iai_offis_g1_description/urdf/offis_unitree_g1.urdf",
        UnitreeG1,
        OmniDrive,
        ARMAR7_START_POSE,
        HomogeneousTransformationMatrix.from_xyz_rpy(z=0.8),
    ),
}

ROBOT_LINEUP = (
    ("armar7", 0.0, -1.1, 0.0, 90.0),
    ("hsrb", 0.0, 1.2, 0.0, 90.0),
    ("stretch", 3.4, 1.1, 0.0, 90.0),
    ("justin", -1.8, -1.0, 0.0, 90.0),
    ("tiago", -1.8, 0.9, 0.0, 90.0),
    ("g1", 1.6, 1.0, 0.8, 90.0),
    ("pr2", 1.6, -0.9, 0.0, 90.0),
)


def build_robot_world(robot_name: str) -> World:
    robot_spec = ROBOT_SPECS[robot_name]
    robot_urdf, robot_cls, drive_cls, robot_start_pose = robot_spec[:4]
    robot_localization_pose = robot_spec[4] if len(robot_spec) > 4 else None
    robot_world = world_with_urdf_factory(
        urdf_path=robot_urdf,
        robot_semantic_annotation=robot_cls,
        drive_connection_type=drive_cls,
        robot_starting_pose=robot_start_pose,
        robot_localization_pose=robot_localization_pose,
    )
    # Each robot world introduces generic "map" and "odom_combined" frames.
    # Namespace them before merging so TF/RViz does not collapse all robots onto
    # the same frame names.
    robot_world.root.name = PrefixedName("map", robot_name)
    robot_world.get_body_by_name("odom_combined").name = PrefixedName(
        "odom_combined", robot_name
    )
    return robot_world


def setup_world() -> World:
    base_robot_name, _, _, _, _ = ROBOT_LINEUP[0]
    world = build_robot_world(base_robot_name)

    for robot_name, x, y, z, yaw in ROBOT_LINEUP[1:]:

        robot_world = build_robot_world(robot_name)
        with robot_world.modify_world():
            robot_world.merge_world(world)

    return world


world = setup_world()
print(world.root)
try:
    import rclpy

    rclpy.init()
    rclpy_node = rclpy.create_node("ros_node")
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)
    viz.with_tf_publisher()
except ImportError:
    pass

while True:
    pass
