import os

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import DiffDrive, OmniDrive
from semantic_digital_twin.world_description.utils import world_with_urdf_factory

THESIS_NEW_DEFAULT_ROBOT = "pr2"
THESIS_NEW_ROBOT_ENV = "THESIS_NEW_ROBOT"
THESIS_NEW_DEFAULT_ENVIRONMENT = "apartment"
THESIS_NEW_ENVIRONMENT_ENV = "THESIS_NEW_ENVIRONMENT"
DEFAULT_ROBOT_START_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.0, 0.0)
ARMAR7_START_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(3.8, 8.40, 0.0)
RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)
WORLDS_DIR = os.path.join(RESOURCES_DIR, "worlds")

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
        DiffDrive,
        DEFAULT_ROBOT_START_POSE,
    ),
    "tiago": (
        "package://iai_tiago_description/urdf/tiago_from_our_robot.urdf",
        Tiago,
        DiffDrive,
        DEFAULT_ROBOT_START_POSE,
    ),
    "armar7": (
        "package://iai_kit_armar7/urdf/Armar7.urdf",
        Armar7,
        OmniDrive,
        ARMAR7_START_POSE,
    ),
}


def resolve_robot_name(robot_name=None):
    selected = robot_name or os.environ.get(
        THESIS_NEW_ROBOT_ENV, THESIS_NEW_DEFAULT_ROBOT
    )
    normalized = str(selected).strip().lower()
    if normalized not in ROBOT_SPECS:
        supported = ", ".join(sorted(ROBOT_SPECS))
        raise ValueError(
            f"Unsupported thesis_new robot '{selected}'. Supported: {supported}"
        )
    return normalized


def resolve_robot_name_from_annotation(robot):
    for robot_name, (_, robot_cls, _, _) in ROBOT_SPECS.items():
        if isinstance(robot, robot_cls):
            return robot_name
    raise ValueError(
        f"Unsupported semantic robot annotation type: {type(robot).__name__}"
    )


def _supported_environment_names():
    supported = []
    for filename in os.listdir(WORLDS_DIR):
        file_path = os.path.join(WORLDS_DIR, filename)
        if os.path.isfile(file_path) and filename.endswith(".urdf"):
            supported.append(os.path.splitext(filename)[0])
    return sorted(supported)


def resolve_environment_path(environment_name=None):
    print(environment_name)
    selected = environment_name or os.environ.get(
        THESIS_NEW_ENVIRONMENT_ENV, THESIS_NEW_DEFAULT_ENVIRONMENT
    )
    normalized = str(selected).strip()

    if normalized.startswith("package://") or os.path.isabs(normalized):
        return normalized

    candidate = normalized[:-5] if normalized.endswith(".urdf") else normalized
    environment_path = os.path.join(WORLDS_DIR, f"{candidate}.urdf")
    if os.path.isfile(environment_path):
        return environment_path

    supported = ", ".join(_supported_environment_names())
    raise ValueError(
        f"Unsupported thesis_new environment '{selected}'. Supported: {supported}"
    )


def setup_thesis_world(robot_name=None, environment_name=None):
    resolved_robot_name = resolve_robot_name(robot_name)
    robot_urdf, robot_cls, drive_cls, robot_start_pose = ROBOT_SPECS[
        resolved_robot_name
    ]
    environment_path = resolve_environment_path(environment_name)

    environment_world = URDFParser.from_file(environment_path).parse()

    robot_world = world_with_urdf_factory(robot_urdf, None, drive_cls)
    environment_world.merge_world_at_pose(robot_world, robot_start_pose)
    robot_cls.from_world(environment_world)

    print(environment_world.root)
    return environment_world
