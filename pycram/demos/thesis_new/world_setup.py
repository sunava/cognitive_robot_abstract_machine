import os
import warnings
from xml.etree import ElementTree as ET

from semantic_digital_twin.adapters.package_resolver import CompositePathResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import DiffDrive, OmniDrive
from semantic_digital_twin.world_description.utils import world_with_urdf_factory

THESIS_NEW_DEFAULT_ROBOT = "pr2"
THESIS_NEW_ROBOT_ENV = "THESIS_NEW_ROBOT"
THESIS_NEW_DEFAULT_ENVIRONMENT = "apartment"
THESIS_NEW_ENVIRONMENT_ENV = "THESIS_NEW_ENVIRONMENT"
DEFAULT_ROBOT_START_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.0, 0.0)
ARMAR7_START_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(3.8, 8.40, 0.0)
ISR_TESTBED_ALIASES = {"isr", "isr-testbed", "isr_testbed"}
RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)
WORLDS_DIR = os.path.join(RESOURCES_DIR, "worlds")
EXTRA_ENVIRONMENT_PATHS = {}

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
    supported.extend(sorted(EXTRA_ENVIRONMENT_PATHS))
    supported.extend(sorted(ISR_TESTBED_ALIASES))
    return sorted(supported)


def _resolve_isr_testbed_path():
    candidates = [
        "/home/vee/workspace/ros/src/isr_testbed/urdf/isr-testbed.urdf",
        "/home/vee/workspace/ros/install/isr_testbed/share/isr_testbed/urdf/isr-testbed.urdf",
        "package://isr_testbed/urdf/isr-testbed.urdf",
    ]
    for candidate in candidates:
        if os.path.isabs(candidate):
            if os.path.isfile(candidate):
                return candidate
            continue
        return candidate

    raise FileNotFoundError("Could not resolve an ISR testbed URDF path.")


def _resolve_robocanes_path():
    candidates = [
        "/home/vee/workspace/ros/src/robocane_manual/urdf/robocane.urdf",
        "/home/vee/workspace/ros/install/robocane_manual//share/robocane_manual/urdf/robocane.urdf",
        "package://robocane_manual/urdf/robocane.urdf",
    ]
    print(candidates)
    for candidate in candidates:
        if os.path.isabs(candidate):
            if os.path.isfile(candidate):
                return candidate
            continue

        return candidate

    raise FileNotFoundError("Could not resolve a Robocanes URDF path.")


def _remove_unresolved_meshes(urdf_path: str) -> str:
    path_resolver = CompositePathResolver()
    resolved_urdf_path = path_resolver.resolve(urdf_path)

    with open(resolved_urdf_path, "r") as urdf_file:
        root = ET.fromstring(urdf_file.read())

    missing_packages = set()
    for link in root.findall("link"):
        for tag_name in ("visual", "collision"):
            for geometry_parent in list(link.findall(tag_name)):
                mesh = geometry_parent.find("geometry/mesh")
                if mesh is None:
                    continue

                filename = mesh.get("filename")
                if not filename:
                    continue

                try:
                    path_resolver.resolve(filename)
                except ParsingError:
                    link.remove(geometry_parent)
                    if filename.startswith("package://"):
                        missing_packages.add(filename.split("/", 3)[2])

    if missing_packages:
        warnings.warn(
            "Skipping unresolved mesh assets from packages: "
            + ", ".join(sorted(missing_packages)),
            RuntimeWarning,
        )

    return ET.tostring(root, encoding="unicode")


def _parse_environment_world(environment_path):
    environment_path_lower = environment_path.lower()
    if any(alias in environment_path_lower for alias in ISR_TESTBED_ALIASES):
        return URDFParser(urdf=_remove_unresolved_meshes(environment_path)).parse()
    if (
        environment_path_lower.endswith("/robocane_manual/urdf/robocanes.urdf")
        or environment_path_lower.endswith("/robocane_manual/urdf/robocane.urdf")
        or environment_path_lower == "package://robocane_manual/urdf/robocane.urdf"
        or environment_path_lower == "package://robocane_manual/urdf/robocane.urdf"
    ):
        return URDFParser(urdf=_remove_unresolved_meshes(environment_path)).parse()
    return URDFParser.from_file(environment_path).parse()


def resolve_environment_path(environment_name=None):
    selected = environment_name or os.environ.get(
        THESIS_NEW_ENVIRONMENT_ENV, THESIS_NEW_DEFAULT_ENVIRONMENT
    )
    normalized = str(selected).strip()

    if normalized.startswith("package://") or os.path.isabs(normalized):
        return normalized

    if normalized.lower() in ISR_TESTBED_ALIASES:
        return _resolve_isr_testbed_path()
    if normalized.lower() == "robocane":
        return _resolve_robocanes_path()

    candidate = normalized[:-5] if normalized.endswith(".urdf") else normalized
    if candidate in EXTRA_ENVIRONMENT_PATHS:
        return EXTRA_ENVIRONMENT_PATHS[candidate]
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

    environment_world = _parse_environment_world(environment_path)

    robot_world = world_with_urdf_factory(robot_urdf, None, drive_cls)
    environment_world.merge_world_at_pose(robot_world, robot_start_pose)
    robot_cls.from_world(environment_world)

    print(environment_world.root)
    return environment_world
