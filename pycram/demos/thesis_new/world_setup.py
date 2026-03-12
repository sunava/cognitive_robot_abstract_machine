import os

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import DiffDrive, OmniDrive
from semantic_digital_twin.world_description.utils import world_with_urdf_factory

THESIS_NEW_DEFAULT_ROBOT = "pr2"
THESIS_NEW_ROBOT_ENV = "THESIS_NEW_ROBOT"
ROBOT_START_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.0, 0.0)
RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)

ROBOT_SPECS = {
    "pr2": (
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro",
        PR2,
        OmniDrive,
    ),
    "hsrb": (
        os.path.join(RESOURCES_DIR, "robots", "hsrb.urdf"),
        HSRB,
        OmniDrive,
    ),
    "stretch": (
        os.path.join(RESOURCES_DIR, "robots", "stretch_description.urdf"),
        Stretch,
        DiffDrive,
    ),
    "tiago": (
        "package://iai_tiago_description/urdf/tiago_from_our_robot.urdf",
        Tiago,
        DiffDrive,
    ),
}


def resolve_robot_name(robot_name=None):
    selected = robot_name or os.environ.get(THESIS_NEW_ROBOT_ENV, THESIS_NEW_DEFAULT_ROBOT)
    normalized = str(selected).strip().lower()
    if normalized not in ROBOT_SPECS:
        supported = ", ".join(sorted(ROBOT_SPECS))
        raise ValueError(f"Unsupported thesis_new robot '{selected}'. Supported: {supported}")
    return normalized


def resolve_robot_name_from_annotation(robot):
    for robot_name, (_, robot_cls, _) in ROBOT_SPECS.items():
        if isinstance(robot, robot_cls):
            return robot_name
    raise ValueError(f"Unsupported semantic robot annotation type: {type(robot).__name__}")


def setup_thesis_world(robot_name=None):
    resolved_robot_name = resolve_robot_name(robot_name)
    robot_urdf, robot_cls, drive_cls = ROBOT_SPECS[resolved_robot_name]

    apartment_world = URDFParser.from_file(
        os.path.join(RESOURCES_DIR, "worlds", "apartment.urdf")
    ).parse()
    robot_world = world_with_urdf_factory(robot_urdf, None, drive_cls)
    apartment_world.merge_world_at_pose(robot_world, ROBOT_START_POSE)
    robot_cls.from_world(apartment_world)
    return apartment_world
