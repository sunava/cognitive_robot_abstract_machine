"""
The recorded demo scene as an EQL (Entity Query Language) knowledge base.

EQL is krrood's pythonic relational query language. This module models the recorded
coraplex/giskardpy episode — bench objects, robot parts, action episodes, per-joint
motion — as plain dataclasses and exposes:

fresh_namespace()  -> dict for evaluating one EQL query (fresh variables)
run_query(code)    -> execute an EQL query string, return JSON-able result
graph_payload()    -> nodes/edges/details/presets for the UI knowledge graph
view_payload(name) -> one of the graph-panel tabs (knowledge / kinematics / plan /
chart)

Importing this module requires krrood; :mod:`cram_viz.server` guards that import so the
static viewer still works without it — only the EQL panel becomes unavailable. Scene
bundles are read from :func:`cram_viz.paths.scenes_dir`.
"""

from __future__ import annotations

import ast
import functools
import json
import logging
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum

from typing_extensions import Any, Protocol, runtime_checkable

from krrood.entity_query_language import factories as eql_factories

from cram_viz import paths


def scene_name() -> str | None:
    """
    The active scene: ``CRAM_VIZ_SCENE``, else the scenes-index default.
    """
    environment_override = os.environ.get("CRAM_VIZ_SCENE")
    if environment_override:
        return environment_override
    index_path = os.path.join(str(paths.scenes_dir()), "index.json")
    if not os.path.isfile(index_path):
        return None
    with open(index_path, encoding="utf-8") as index_file:
        index = json.load(index_file)
    return index.get("default")


def scene_dir() -> str | None:
    """
    Directory of the active scene bundle, or None without one.
    """
    name = scene_name()
    return os.path.join(str(paths.scenes_dir()), name) if name else None


def load_scene() -> tuple[dict[str, Any], dict[str, Any]]:
    """
    The active scene's (scene.json, trajectory.json), or ``({}, {})``.
    """
    directory = scene_dir()
    if not directory:
        return {}, {}
    scene_path = os.path.join(directory, "scene.json")
    if not os.path.isfile(scene_path):
        return {}, {}
    with open(scene_path, encoding="utf-8") as scene_file:
        scene = json.load(scene_file)
    trajectory_path = os.path.join(
        directory, scene.get("trajectory", "trajectory.json")
    )
    if not os.path.isfile(trajectory_path):
        return scene, {}
    with open(trajectory_path, encoding="utf-8") as trajectory_file:
        trajectory = json.load(trajectory_file)
    return scene, trajectory


def load_urdf() -> tuple[list[str], list[dict[str, str]]]:
    """
    Parse the active scene's robot URDF into (links, joints).

    Used by the kinematic-tree view; a regex parse suffices because the bundled URDFs
    are flat.
    """
    scene, _ = load_scene()
    robot_model = next(
        (model for model in scene.get("models", []) if model.get("robot")), None
    )
    directory = scene_dir()
    if not robot_model or not directory:
        return [], []
    urdf_path = os.path.join(directory, robot_model["urdf"])
    if not os.path.isfile(urdf_path):
        return [], []
    with open(urdf_path, encoding="utf-8", errors="replace") as urdf_file:
        text = urdf_file.read()
    links = re.findall(r'<link\s+name="([^"]+)"', text)
    joints = []
    for joint in re.finditer(
        r'<joint\s+name="([^"]+)"\s+type="([^"]+)">(.*?)</joint>', text, re.S
    ):
        body = joint.group(3)
        parent = re.search(r'<parent\s+link="([^"]+)"', body)
        child = re.search(r'<child\s+link="([^"]+)"', body)
        if parent and child:
            joints.append(
                {
                    "name": joint.group(1),
                    "type": joint.group(2),
                    "parent": parent.group(1),
                    "child": child.group(1),
                }
            )
    return links, joints


# %% enums shared by the entity model and the UI graph
class ObjectKind(Enum):
    """
    Whether a :class:`BenchObject` is a graspable thing or a named area.
    """

    OBJECT = "object"
    LOCATION = "location"


class BodySide(Enum):
    """
    Which side of the robot a part, arm or joint belongs to.
    """

    LEFT = "left"
    RIGHT = "right"
    NOT_APPLICABLE = "n/a"
    BODY = "body"
    ENVIRONMENT = "environment"


class GraphGroup(Enum):
    """
    Colour group of a UI knowledge-graph node.
    """

    ROBOT = "robot"
    OBJECT = "object"
    EVENT = "event"
    GOAL = "goal"
    CONCEPT = "concept"
    ROOT = "root"
    KLASS = "klass"
    PYCLASS = "pyclass"
    UPPER = "upper"
    IND = "ind"


class EdgeKind(Enum):
    """
    Visual style of a UI knowledge-graph edge: solid or dashed.
    """

    PROP = "prop"
    TYPE = "type"


# %% the entity model ---------------------------------------------------------
@dataclass(frozen=True)
class Position:
    """
    A world position in metres.
    """

    x: float
    """
    World x coordinate.
    """

    y: float
    """
    World y coordinate.
    """

    z: float
    """
    World z coordinate.
    """

    def __repr__(self) -> str:
        return "(%.2f, %.2f, %.2f)" % (self.x, self.y, self.z)


@dataclass(frozen=True)
class Gripper:
    """
    An end effector of the recorded robot.
    """

    name: str
    """
    Part name from the scene's robot annotation.
    """

    side: BodySide
    """
    Body side the gripper belongs to.
    """

    opening_m: float = 0.085
    """
    Maximum opening width in metres (Robotiq 2F-85 default).
    """

@dataclass(frozen=True)
class Arm:
    """
    A manipulator of the recorded robot.
    """

    name: str
    """
    Part name from the scene's robot annotation.
    """

    side: BodySide
    """
    Body side of the arm.
    """

    robot: str
    """
    Name of the robot this arm belongs to.
    """

    gripper: Gripper
    """
    The end effector mounted on this arm.
    """


@dataclass(frozen=True)
class Robot:
    """
    The robot that executed the recorded episode.
    """

    name: str
    """
    Robot name from the scene bundle.
    """

    arm_count: int
    """
    Number of annotated arms.
    """


@dataclass(frozen=True)
class BenchObject:
    """
    A loose object (or named location) in the scene.
    """

    name: str
    """
    Object identifier, e.g. ``milk``.
    """

    kind: ObjectKind
    """
    Whether this is a graspable thing or a named area.
    """

    label: str
    """
    Human-readable display name.
    """

    height_m: float
    """
    Approximate object height in metres.
    """

    position: Position
    """
    Spawn position recorded at frame 0 of the episode.
    """


@dataclass(frozen=True)
class ActionEpisode:
    """
    One executed plan segment of the recording.
    """

    name: str
    """
    Segment step name, e.g. ``transport_milk``.
    """

    index: int
    """
    Position of the episode in execution order.
    """

    start_frame: int
    """
    First trajectory frame of the episode.
    """

    end_frame: int
    """
    Frame after the last trajectory frame of the episode.
    """

    duration_s: float
    """
    Episode duration in seconds.
    """

    performed_by: Arm | None
    """
    The arm that performed the manipulation, if any.
    """

    picks: BenchObject | None
    """
    The object the episode picks up, if any.
    """

    places_at: BenchObject | None
    """
    The location the object is placed at, if any.
    """


@dataclass(frozen=True)
class JointMotion:
    """
    Per-joint motion statistics over the whole recorded trajectory.
    """

    name: str
    """
    Joint name (without the model prefix).
    """

    arm_side: BodySide
    """
    Body side the joint belongs to.
    """

    min_rad: float
    """
    Smallest recorded joint position (radians or metres).
    """

    max_rad: float
    """
    Largest recorded joint position (radians or metres).
    """

    range_rad: float
    """Travelled range, ``max_rad - min_rad``."""


# %% the CRAM architecture entities --------------------------------------------
@dataclass(frozen=True)
class Package:
    """
    A top-level package of the CRAM repository.
    """

    name: str
    """
    Directory name, e.g. ``coraplex``.
    """

    description: str
    """
    One-line description (curated, or the first README line).
    """

    module_count: int
    """
    Number of Python modules in the package.
    """

    class_count: int
    """
    Number of classes defined in the package.
    """


@dataclass(frozen=True)
class SubPackage:
    """
    A qualified subpackage, e.g. ``coraplex.plans``.
    """

    name: str
    """
    Qualified name, e.g. ``coraplex.plans``.
    """

    package: str
    """
    The top-level package this subpackage belongs to.
    """

    module_count: int
    """
    Number of modules in the subpackage.
    """

    class_count: int
    """
    Number of classes defined in the subpackage.
    """


@dataclass(frozen=True)
class PythonClass:
    """
    A class found by the static scan of the CRAM repository.
    """

    name: str
    """
    Class name.
    """

    package: str
    """
    Top-level package the class is defined in.
    """

    subpackage: str
    """
    Qualified subpackage (equal to ``package`` for top-level modules).
    """

    module: str
    """
    Repository-relative module path.
    """

    bases: tuple
    """
    Names of the direct base classes.
    """

    methods: int
    """
    Number of methods defined on the class.
    """

    doc: str
    """
    First docstring line, or ``''``.
    """


#: entity types that carry a ``name`` field usable as a graph-node/highlight id
_NAMED_ENTITY_TYPES = (
    Gripper,
    Arm,
    Robot,
    BenchObject,
    ActionEpisode,
    JointMotion,
    Package,
    SubPackage,
    PythonClass,
)


# %% scan the CRAM architecture --------------------------------------------------
def _cram_root() -> str:
    """
    The CRAM repository the architecture graph is scanned from.
    """
    return str(paths.architecture_root())


def _architecture_cache() -> str:
    """
    Path of the scan cache — always in the writable data directory, because the scenes
    checkout may be read-only.
    """
    return os.path.join(str(paths.data_dir()), "arch_cache.json")


#: directories never descended into during the architecture scan
SKIP_DIRS = {
    "__pycache__",
    "node_modules",
    "doc",
    "docs",
    "resources",
    "build",
    "dist",
    "plugins",
}

#: curated one-line descriptions for the well-known workspace packages
PKG_DESCRIPTIONS = {
    "krrood": "knowledge representation & reasoning through OO design (home of EQL)",
    "coraplex": "the plan executive: designators, plans, locations",
    "pycram": "legacy plan executive (resources/demos)",
    "giskardpy": "constraint-based motion planning and control",
    "robokudo": "perception framework",
    "semantic_digital_twin": "semantic world model / digital twin",
    "segmind": "segmentation / vision models",
    "probabilistic_model": "probabilistic models and inference",
    "random_events": "sigma-algebra & random events for probabilistic reasoning",
    "physics_simulators": "physics simulator bindings",
    "experiments": "experiment scripts (incl. EQL experiments)",
    "test": "the test suites of all packages",
    "scripts": "maintenance scripts",
    "root": "top-level demo scripts (sterility test, wind turbine…)",
}


def _first_readme_line(directory: str) -> str:
    """
    The first non-empty line of a directory's README, or ``''``.
    """
    for name in ("README.md", "readme.md"):
        readme_path = os.path.join(directory, name)
        if os.path.exists(readme_path):
            with open(readme_path, encoding="utf-8", errors="replace") as readme_file:
                for line in readme_file:
                    line = line.strip().lstrip("#").strip()
                    if line:
                        return line[:120]
    return ""


def scan_architecture() -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, str]]]
):
    """
    Statically scan the CRAM repository for its architecture graph.

    :return: (packages, classes, import edges between packages). A pure ``ast`` parse —
        nothing is imported.
    """
    packages: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    imports: dict[str, set] = {}
    cram_root = _cram_root()
    if not os.path.isdir(cram_root):
        return packages, classes, []

    package_dirs = {"root": cram_root}
    for entry in sorted(os.listdir(cram_root)):
        directory = os.path.join(cram_root, entry)
        if (
            os.path.isdir(directory)
            and not entry.startswith(".")
            and entry not in SKIP_DIRS
            and "egg-info" not in entry
        ):
            package_dirs[entry] = directory
    package_names = set(package_dirs)

    modules_per_package: dict[str, int] = {}
    for package, base in package_dirs.items():
        module_count = 0
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not name.startswith(".") and name not in SKIP_DIRS
            )
            if package == "root":
                dirnames[:] = []  # root package = top-level scripts only
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, encoding="utf-8", errors="replace") as module_file:
                    source = module_file.read()
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    # a module mid-edit in the scanned checkout; skip it, the scan
                    # itself is the only way to know whether a file parses
                    continue
                module_count += 1
                module = os.path.relpath(path, cram_root)[:-3].replace(os.sep, ".")
                _collect_classes_and_imports(
                    tree, package, module, package_names, classes, imports
                )
        modules_per_package[package] = module_count

    class_counts = Counter(entry["package"] for entry in classes)
    for package in package_dirs:
        description = PKG_DESCRIPTIONS.get(package) or _first_readme_line(
            package_dirs[package]
        )
        packages.append(
            dict(
                name=package,
                description=description,
                module_count=modules_per_package.get(package, 0),
                class_count=class_counts.get(package, 0),
            )
        )
    dependency_edges = sorted(
        (source, target) for source, targets in imports.items() for target in targets
    )
    return packages, classes, dependency_edges


def _collect_classes_and_imports(
    tree: ast.Module,
    package: str,
    module: str,
    package_names: set,
    classes: list[dict[str, Any]],
    imports: dict[str, set],
) -> None:
    """
    Collect class definitions and cross-package imports from one module.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = tuple(
                (
                    base.id
                    if isinstance(base, ast.Name)
                    else (base.attr if isinstance(base, ast.Attribute) else "?")
                )
                for base in node.bases
            )
            doc = (ast.get_docstring(node) or "").strip().split("\n")[0][:140]
            methods = sum(
                1
                for member in node.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            classes.append(
                dict(
                    name=node.name,
                    package=package,
                    module=module,
                    bases=list(bases),
                    methods=methods,
                    doc=doc,
                )
            )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif node.level == 0:
                roots = [(node.module or "").split(".")[0]]
            else:
                roots = []
            for root in roots:
                if root in package_names and root != package:
                    imports.setdefault(package, set()).add(root)


def _load_architecture_cache(cram_root: str, require_classes: bool) -> tuple | None:
    """
    The cached scan if it is usable, else None.

    A cache written for another repository root is not trusted (unless no repository
    exists at all, in which case any cache beats nothing). A malformed cache (e.g. from
    a crashed write) is treated the same as no cache — the caller falls back to a fresh
    scan rather than failing outright.
    """
    cache_path = _architecture_cache()
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
    except json.JSONDecodeError:
        return None
    if cached.get("version") != 2:
        return None
    if os.path.isdir(cram_root) and cached.get("cram_root") != cram_root:
        return None
    if require_classes and not cached.get("classes"):
        return None
    return (
        cached["packages"],
        cached["classes"],
        [tuple(edge) for edge in cached["deps"]],
    )


def load_architecture() -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, str]]]
):
    """
    :func:`scan_architecture` behind a JSON disk cache.

    A full scan takes seconds, so results are cached in the data directory, keyed by the
    scanned root; a cache from another root is rescanned.
    """
    cram_root = _cram_root()
    cached = _load_architecture_cache(cram_root, require_classes=False)
    if cached is not None:
        return cached
    if not os.path.isdir(cram_root):
        return [], [], []
    packages, classes, dependency_edges = scan_architecture()
    if not classes:
        # a checkout exists but yielded nothing (empty or partial clone) — fall
        # back to the cache rather than losing the architecture graph
        return _load_architecture_cache(cram_root, require_classes=True) or (
            packages,
            classes,
            dependency_edges,
        )
    os.makedirs(os.path.dirname(_architecture_cache()), exist_ok=True)
    with open(_architecture_cache(), "w", encoding="utf-8") as cache_file:
        json.dump(
            {
                "version": 2,
                "cram_root": cram_root,
                "packages": packages,
                "classes": classes,
                "deps": dependency_edges,
            },
            cache_file,
        )
    return packages, classes, dependency_edges


def _side_of_name(name: str) -> BodySide | None:
    """
    Body side encoded in a part/link name, or None if neither.
    """
    lowered = name.lower()
    if "left" in lowered or lowered.startswith("l_"):
        return BodySide.LEFT
    if "right" in lowered or lowered.startswith("r_"):
        return BodySide.RIGHT
    return None


@dataclass(frozen=True, eq=False)
class KB:
    """
    The recorded episode as EQL-queryable entities.

    Built once from the active scene bundle plus a static scan of the CRAM repository;
    every list attribute is a plain list of dataclass instances that EQL variables range
    over.
    """

    objects: list[BenchObject]
    """
    Scene objects (spawn poses recorded at frame 0) plus the place area.
    """

    grippers: list[Gripper]
    """
    End effectors of the recorded robot.
    """

    arms: list[Arm]
    """
    Manipulators of the recorded robot.
    """

    robot: Robot
    """
    The robot that executed the recorded episode.
    """

    episodes: list[ActionEpisode]
    """
    Executed plan segments of the recording.
    """

    joints: list[JointMotion]
    """
    Per-joint motion statistics over the whole recorded trajectory.
    """

    packages: list[Package]
    """
    Top-level packages of the scanned CRAM repository.
    """

    classes: list[PythonClass]
    """
    Classes found by the static scan of the CRAM repository.
    """

    package_deps: list[tuple[str, str]]
    """
    Import edges between packages, as (source, target) pairs.
    """

    subpackages: list[SubPackage]
    """
    Qualified subpackages aggregated from the scanned classes.
    """

    @classmethod
    def build(cls) -> KB:
        """
        Build every entity list from the active scene bundle and a static scan of the
        CRAM architecture.
        """
        scene, trajectory = load_scene()
        frames_per_second = scene.get("fps", 30)
        parts = (scene.get("robot") or {}).get("parts") or {}
        robot_name = (scene.get("robot") or {}).get("name", "robot")
        robot_prefix = (scene.get("robot") or {}).get("prefix", "")

        objects = cls._build_objects(scene)
        objects_by_id = {entity.name: entity for entity in objects}
        place_area = objects_by_id.get("place_area")

        grippers, arms = cls._build_arms(parts, robot_name)
        robot = Robot(robot_name, arm_count=len(arms))
        episodes = cls._build_episodes(
            arms, scene, frames_per_second, objects_by_id, place_area
        )
        joints = cls._build_joint_motions(trajectory, parts, robot_prefix)

        packages, classes, dependency_edges = load_architecture()
        built_packages = [Package(**entry) for entry in packages]
        built_classes = [
            PythonClass(
                name=entry["name"],
                package=entry["package"],
                subpackage=cls._subpackage_of(entry["package"], entry["module"]),
                module=entry["module"],
                bases=tuple(entry["bases"]),
                methods=entry["methods"],
                doc=entry["doc"],
            )
            for entry in classes
        ]
        return cls(
            objects=objects,
            grippers=grippers,
            arms=arms,
            robot=robot,
            episodes=episodes,
            joints=joints,
            packages=built_packages,
            classes=built_classes,
            package_deps=dependency_edges,
            subpackages=cls._build_subpackages(built_classes),
        )

    @staticmethod
    def _build_objects(scene: dict[str, Any]) -> list[BenchObject]:
        """
        Scene objects (spawn poses recorded at frame 0) plus the place area.
        """
        objects = []
        for entry in scene.get("objects") or []:
            objects.append(
                BenchObject(
                    name=entry["id"],
                    kind=ObjectKind.OBJECT,
                    label=entry["id"].replace("_", " ").title(),
                    height_m=0.1,
                    position=Position(
                        *[round(value, 3) for value in entry["spawn"][:3]]
                    ),
                )
            )
        if scene.get("placeTarget"):
            target = scene["placeTarget"]
            objects.append(
                BenchObject(
                    name="place_area",
                    kind=ObjectKind.LOCATION,
                    label="Place area",
                    height_m=0.0,
                    position=Position(
                        round(target["pos"][0], 3),
                        round(target["pos"][1], 3),
                        target.get("z", 0),
                    ),
                )
            )
        return objects

    @staticmethod
    def _build_arms(
        parts: dict[str, Any], robot_name: str
    ) -> tuple[list[Gripper], list[Arm]]:
        """
        Arms and grippers from the recorded robot annotation parts.

        Gripper keywords take precedence — robot names can contain 'arm' themselves, so
        'arm' alone must not decide.
        """
        gripper_parts = [
            part
            for part in parts
            if any(
                keyword in part.lower() for keyword in ("gripper", "hand", "effector")
            )
        ]
        arm_parts = [
            part
            for part in parts
            if part not in gripper_parts and "arm" in part.lower()
        ]
        grippers, arms = [], []
        for arm_part in sorted(arm_parts):
            side = _side_of_name(arm_part) or BodySide.NOT_APPLICABLE
            gripper_part = next(
                (part for part in gripper_parts if _side_of_name(part) == side), None
            )
            gripper = Gripper(gripper_part or (arm_part + "_ee"), side)
            grippers.append(gripper)
            arms.append(Arm(arm_part, side, robot_name, gripper))
        return grippers, arms

    @staticmethod
    def _build_episodes(
        arms: list[Arm],
        scene: dict[str, Any],
        frames_per_second: int,
        objects_by_id: dict[str, BenchObject],
        place_area: BenchObject | None,
    ) -> list[ActionEpisode]:
        """
        Action episodes from the recorded plan segments.
        """

        def arm_for(segment: dict[str, Any]) -> Arm | None:
            """
            The arm matching the segment's recorded side hint, falling back to the first
            arm if the segment picks something but names no side.
            """
            hint = (segment.get("arm") or "").lower()
            for arm in arms:
                if arm.side is not BodySide.NOT_APPLICABLE and arm.side.value in hint:
                    return arm
            return arms[0] if arms and segment.get("picks") else None

        episodes = []
        for index, segment in enumerate(scene.get("segments") or []):
            picks = objects_by_id.get(segment.get("picks"))
            episodes.append(
                ActionEpisode(
                    name=segment["step"],
                    index=index,
                    start_frame=segment["start"],
                    end_frame=segment["end"],
                    duration_s=round(
                        (segment["end"] - segment["start"]) / max(1, frames_per_second),
                        1,
                    ),
                    performed_by=arm_for(segment) if picks else None,
                    picks=picks,
                    places_at=place_area if picks else None,
                )
            )
        return episodes

    @staticmethod
    def _build_joint_motions(
        trajectory: dict[str, Any], parts: dict[str, Any], robot_prefix: str
    ) -> list[JointMotion]:
        """
        Per-joint motion statistics over the whole recorded trajectory.
        """
        minimum: dict[str, float] = {}
        maximum: dict[str, float] = {}
        for frame in trajectory.get("frames") or []:
            for joint, value in frame.items():
                if joint not in minimum or value < minimum[joint]:
                    minimum[joint] = value
                if joint not in maximum or value > maximum[joint]:
                    maximum[joint] = value

        link_to_part = {link: part for part, links in parts.items() for link in links}

        def side_of(key: str) -> BodySide:
            """
            Which arm side a prefixed joint key belongs to, or BODY/ENVIRONMENT when it
            isn't part of an arm.
            """
            prefix, _, joint_name = key.partition("/")
            if "/" not in key:
                prefix, joint_name = "", key
            if robot_prefix and prefix != robot_prefix:
                return BodySide.ENVIRONMENT
            part = link_to_part.get(joint_name.replace("_joint", "_link"))
            if part and _side_of_name(part):
                return _side_of_name(part)
            return _side_of_name(joint_name) or BodySide.BODY

        return [
            JointMotion(
                name=key.partition("/")[2] or key,
                arm_side=side_of(key),
                min_rad=round(minimum[key], 3),
                max_rad=round(maximum[key], 3),
                range_rad=round(maximum[key] - minimum[key], 3),
            )
            for key in sorted(minimum)
        ]

    @staticmethod
    def _subpackage_of(package: str, module: str) -> str:
        """
        Qualified subpackage of a module path.

        ``coraplex.src.coraplex.plans.designator`` → ``coraplex.plans``; top-level
        modules collapse onto the package itself.
        """
        segments = module.split(".")
        if segments and segments[0] == package:
            segments = segments[1:]
        while segments and segments[0] in ("src", package):
            segments = segments[1:]
        return package + "." + segments[0] if len(segments) >= 2 else package

    @staticmethod
    def _build_subpackages(classes: list[PythonClass]) -> list[SubPackage]:
        """
        Subpackage entities aggregated from the scanned classes.
        """
        modules = defaultdict(set)
        class_counts = defaultdict(int)
        for entry in classes:
            if entry.subpackage != entry.package:
                modules[(entry.package, entry.subpackage)].add(entry.module)
                class_counts[entry.subpackage] += 1
        return [
            SubPackage(
                name=subpackage,
                package=package,
                module_count=len(modules[(package, subpackage)]),
                class_count=class_counts[subpackage],
            )
            for (package, subpackage) in sorted(modules)
        ]


@functools.lru_cache(maxsize=1)
def get_kb() -> KB:
    """
    The process-wide knowledge base, built on first use.
    """
    return KB.build()


def reset_kb() -> None:
    """
    Drop the cached KB (tests point CRAM_VIZ_SCENES at fixtures).
    """
    get_kb.cache_clear()


# %% EQL session -----------------------------------------------------------------
def fresh_namespace() -> dict[str, Any]:
    """
    A namespace for evaluating one EQL query (fresh variables each time).
    """
    kb = get_kb()
    namespace = {
        "entity": eql_factories.entity,
        "set_of": eql_factories.set_of,
        "variable": eql_factories.variable,
        "an": eql_factories.an,
        "a": eql_factories.a,
        "the": eql_factories.the,
        "and_": eql_factories.and_,
        "or_": eql_factories.or_,
        "not_": eql_factories.not_,
        "contains": eql_factories.contains,
        "in_": eql_factories.in_,
        "exists": eql_factories.exists,
        "for_all": eql_factories.for_all,
        "count": eql_factories.count,
        "count_all": eql_factories.count_all,
        "average": eql_factories.average,
        "sum": eql_factories.sum,
        "min": eql_factories.min,
        "max": eql_factories.max,
        "mode": eql_factories.mode,
        "distinct": eql_factories.distinct,
        "flat_variable": eql_factories.flat_variable,
        "variable_from": eql_factories.variable_from,
    }
    namespace.update(
        Position=Position,
        Gripper=Gripper,
        Arm=Arm,
        Robot=Robot,
        BenchObject=BenchObject,
        ActionEpisode=ActionEpisode,
        JointMotion=JointMotion,
        Package=Package,
        SubPackage=SubPackage,
        PythonClass=PythonClass,
        objects=kb.objects,
        episodes=kb.episodes,
        arms=kb.arms,
        grippers=kb.grippers,
        joints=kb.joints,
        robots=[kb.robot],
        packages=kb.packages,
        subpackages=kb.subpackages,
        classes=kb.classes,
    )
    # ready-made query variables so one-liners stay short
    namespace["object"] = eql_factories.variable(BenchObject, domain=kb.objects)
    namespace["episode"] = eql_factories.variable(ActionEpisode, domain=kb.episodes)
    namespace["arm"] = eql_factories.variable(Arm, domain=kb.arms)
    namespace["joint"] = eql_factories.variable(JointMotion, domain=kb.joints)
    namespace["robot"] = eql_factories.variable(Robot, domain=[kb.robot])
    namespace["package"] = eql_factories.variable(Package, domain=kb.packages)
    namespace["subpackage"] = eql_factories.variable(SubPackage, domain=kb.subpackages)
    namespace["python_class"] = eql_factories.variable(PythonClass, domain=kb.classes)
    return namespace


def _entity_name(value: Any) -> str | None:
    """
    The entity's name attribute, or None for entity types without one (e.g. Position).
    """
    if isinstance(value, _NAMED_ENTITY_TYPES):
        return value.name
    return None


def _jsonable(value: Any) -> Any:
    """
    A JSON-serializable rendering of one query result value.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return _entity_name(value) or repr(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


class EmptyEqlQueryError(ValueError):
    """
    Raised when an EQL query string contains no statements.
    """


@runtime_checkable
class _Evaluable(Protocol):
    """
    An EQL query object with a lazy ``evaluate()`` step.
    """

    def evaluate(self) -> Any: ...


@runtime_checkable
class _ItemsLike(Protocol):
    """
    A mapping-like EQL result (e.g. ``UnificationDict`` from ``set_of()``).
    """

    def items(self) -> Any: ...


def run_query(code: str, limit: int = 200) -> dict[str, Any]:
    """
    Execute an EQL query string and return a JSON-able result payload.

    The last expression of ``code`` is the query; preceding statements are executed as
    setup. Like :func:`view_payload`/:func:`graph_payload`/:func:`expand_node`, a failure
    is reported as ``{"ok": False, "error": ...}`` rather than raised — ``code`` is
    untrusted text typed into the browser's EQL panel, a system boundary.

    :param code: The EQL query source.
    :param limit: Maximum number of result rows to return.
    """
    try:
        namespace = fresh_namespace()
        tree = ast.parse(code, mode="exec")
        if not tree.body:
            raise EmptyEqlQueryError("empty query")
        last = tree.body[-1]
        if isinstance(last, ast.Expr):
            if len(tree.body) > 1:
                preamble = ast.Module(body=tree.body[:-1], type_ignores=[])
                exec(compile(preamble, "<eql>", "exec"), namespace)
            result = eval(
                compile(ast.Expression(last.value), "<eql>", "eval"), namespace
            )
        else:
            exec(compile(tree, "<eql>", "exec"), namespace)
            result = namespace.get("result")
        if isinstance(result, _Evaluable):
            result = result.evaluate()
        rows, highlight, more = _result_rows(result, limit)
    except Exception as error:
        return {"ok": False, "error": "%s: %s" % (type(error).__name__, error)}
    kind = "rows" if rows and "__entity__" not in rows[0] else "entities"
    return {
        "ok": True,
        "kind": kind,
        "rows": rows,
        "count": len(rows),
        "more": more,
        "highlight": sorted(set(highlight)),
    }


def _result_rows(
    result: Any, limit: int
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """
    Render a query result into (answer rows, highlight ids, truncated).
    """
    rows: list[dict[str, Any]] = []
    highlight: list[str] = []
    if result is None:
        return rows, highlight, False
    if isinstance(result, (str, int, float, bool)):
        rows.append({"value": _jsonable(result)})
        return rows, highlight, False
    if is_dataclass(result) and not isinstance(result, type):
        rows.append(_entity_row(result, highlight))
        return rows, highlight, False
    try:
        iterator = iter(result)
    except TypeError:
        rows.append({"value": _jsonable(result)})
        return rows, highlight, False
    for item in iterator:
        if len(rows) >= limit:
            return rows, highlight, True
        rows.append(_item_row(item, highlight))
    return rows, highlight, False


def _entity_row(item: Any, highlight: list[str]) -> dict[str, Any]:
    """
    One entity as an answer row; collects the ids to highlight.
    """
    name = _entity_name(item)
    if isinstance(item, PythonClass):
        # classes aren't graph nodes — light up their subpackage + package instead
        highlight.append(item.subpackage)
        highlight.append(item.package)
    elif name:
        highlight.append(name)
    row = {"__entity__": name or repr(item), "__type__": type(item).__name__}
    for entity_field in fields(item):
        if entity_field.name != "name":
            row[entity_field.name] = _jsonable(item.__dict__[entity_field.name])
    return row


def _item_row(item: Any, highlight: list[str]) -> dict[str, Any]:
    """
    One arbitrary query result item as an answer row.
    """
    if is_dataclass(item) and not isinstance(item, type):
        return _entity_row(item, highlight)
    if isinstance(item, _ItemsLike):  # UnificationDict from set_of()
        row = {}
        for key, value in item.items():
            if (
                is_dataclass(value)
                and not isinstance(value, type)
                and _entity_name(value)
            ):
                highlight.append(_entity_name(value))
            row[str(key)] = _jsonable(value)
        return row
    return {"value": _jsonable(item)}


# %% the UI graph -----------------------------------------------------------------
NodeAdder = Callable[[str, str, GraphGroup, list[str]], None]


def _edge(source: str, target: str, kind: EdgeKind, label: str) -> dict[str, Any]:
    """
    One knowledge-graph edge, ready for JSON.
    """
    return {"from": source, "to": target, "kind": kind.value, "label": label}


def _add_robot_nodes(kb: KB, add: NodeAdder, edges: list[dict[str, Any]]) -> None:
    """
    Add the robot node plus its arms and grippers.
    """
    rob = kb.robot.name
    add(
        rob,
        rob,
        GraphGroup.ROBOT,
        [
            "a Robot",
            "%d arm%s" % (kb.robot.arm_count, "" if kb.robot.arm_count == 1 else "s"),
            "double-click: full URDF tree",
        ],
    )
    for arm in kb.arms:
        add(
            arm.name,
            arm.name.replace("_", " "),
            GraphGroup.ROBOT,
            ["an Arm", "side: " + arm.side.value, "gripper: " + arm.gripper.name],
        )
        edges.append(_edge(rob, arm.name, EdgeKind.PROP, "has part"))
        add(
            arm.gripper.name,
            arm.gripper.name.replace("_", " "),
            GraphGroup.ROBOT,
            [
                "a Gripper",
                "side: " + arm.gripper.side.value,
                "opening: %.3f m" % arm.gripper.opening_m,
            ],
        )
        edges.append(_edge(arm.name, arm.gripper.name, EdgeKind.PROP, "has part"))


def _add_object_nodes(kb: KB, add: NodeAdder) -> None:
    """
    Add the scene's bench objects.
    """
    for bench_object in kb.objects:
        add(
            bench_object.name,
            bench_object.label,
            GraphGroup.OBJECT,
            [
                "a BenchObject",
                "kind: " + bench_object.kind.value,
                "position: " + repr(bench_object.position),
                "height: %.2f m" % bench_object.height_m,
            ],
        )


def _add_episode_nodes(kb: KB, add: NodeAdder, edges: list[dict[str, Any]]) -> None:
    """
    Add the executed action episodes, chained in execution order.
    """
    previous = None
    for episode in kb.episodes:
        add(
            episode.name,
            episode.name,
            GraphGroup.EVENT,
            [
                "an ActionEpisode",
                "frames %d–%d" % (episode.start_frame, episode.end_frame),
                "duration: %.1f s" % episode.duration_s,
            ]
            + (["picks: " + episode.picks.name] if episode.picks else [])
            + (["places at: " + episode.places_at.name] if episode.places_at else []),
        )
        if previous:
            edges.append(_edge(previous, episode.name, EdgeKind.TYPE, "precedes"))
        previous = episode.name
        # the robot performs the episode (with its arm); don't wire the episode
        # straight to the arm — the arm hangs off the robot, so the chain reads
        # transport_milk → pr2 → left_arm → left_gripper
        if episode.performed_by:
            edges.append(
                _edge(episode.name, episode.performed_by.robot, EdgeKind.PROP, "performed by")
            )
        if episode.picks:
            edges.append(_edge(episode.name, episode.picks.name, EdgeKind.PROP, "picks"))
        if episode.places_at:
            edges.append(
                _edge(episode.name, episode.places_at.name, EdgeKind.PROP, "places at")
            )


def _add_architecture_nodes(
    kb: KB,
    add: NodeAdder,
    edges: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    rob: str,
) -> None:
    """
    Add the CRAM architecture cluster (repo root → packages → subpackages, plus import
    edges) and ground the demo in it.
    """
    if not kb.packages:
        return
    add(
        "cram",
        "CRAM architecture",
        GraphGroup.ROOT,
        [
            "~/cognitive_robot_abstract_machine",
            "%d packages · %d Python classes" % (len(kb.packages), len(kb.classes)),
        ],
    )
    for package in kb.packages:
        add(
            package.name,
            package.name,
            GraphGroup.CONCEPT,
            [
                "a Package",
                package.description,
                "%d modules · %d classes" % (package.module_count, package.class_count),
                "double-click to open",
            ],
        )
        edges.append(_edge("cram", package.name, EdgeKind.PROP, "contains"))
    for subpackage in kb.subpackages:
        add(
            subpackage.name,
            subpackage.name.split(".", 1)[1],
            GraphGroup.KLASS,
            [
                "a SubPackage of " + subpackage.package,
                "%d modules · %d classes"
                % (subpackage.module_count, subpackage.class_count),
                "double-click to open",
            ],
        )
        edges.append(_edge(subpackage.package, subpackage.name, EdgeKind.PROP, "contains"))
    for source, target in kb.package_deps:
        edges.append(_edge(source, target, EdgeKind.TYPE, "imports"))

    # ground the demo in the architecture at the *subpackage* that actually
    # realises each part (only wire to a node that exists in this view)
    def link(src: str, dst: str, label: str) -> None:
        """
        Add an edge, but only if dst is actually a node in this view.
        """
        if any(n["id"] == dst for n in nodes):
            edges.append(_edge(src, dst, EdgeKind.TYPE, label))

    # anchor one representative manipulation episode (they share the stack)
    anchor = next((episode.name for episode in kb.episodes if episode.picks), None)
    if anchor:
        link(anchor, "coraplex.plans", "planned by")  # plan / designator layer
        link(anchor, "giskardpy.motion_statechart", "motion by")  # motion execution
    # every physical thing in the scene is modelled in the semantic digital twin
    link(rob, "semantic_digital_twin", "modelled in")
    for bench_object in kb.objects:
        link(bench_object.name, "semantic_digital_twin", "modelled in")


def _add_plan_summary_node(
    kb: KB, add: NodeAdder, edges: list[dict[str, Any]], rob: str
) -> None:
    """
    Add the executed-plan summary node, if the scene recorded a plan tree.
    """
    scene, _ = load_scene()
    if not scene.get("planTrees"):
        return
    node_count = sum(_count_plan_nodes(tree) for tree in scene["planTrees"])
    add(
        "plan",
        "executed plan",
        GraphGroup.GOAL,
        [
            "the plan tree the demo actually executed",
            "%d nodes" % node_count,
            "double-click to open",
        ],
    )
    edges.append(_edge("plan", rob, EdgeKind.PROP, "executed by"))
    for episode in kb.episodes:
        edges.append(_edge("plan", episode.name, EdgeKind.TYPE, "spans"))


def graph_payload() -> dict[str, Any]:
    """
    The knowledge-graph overview: nodes, edges, details and presets.
    """
    kb = get_kb()
    nodes, edges, details = [], [], {}

    def add(node_id: str, label: str, group: GraphGroup, lines: list[str]) -> None:
        """
        Append one graph node and its detail-panel entry.
        """
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "group": group.value,
                "title": "\n".join([label] + lines),
            }
        )
        details[node_id] = {"label": label, "group": group.value, "lines": lines}

    rob = kb.robot.name
    _add_robot_nodes(kb, add, edges)
    _add_object_nodes(kb, add)
    _add_episode_nodes(kb, add, edges)
    _add_architecture_nodes(kb, add, edges, nodes, rob)
    _add_plan_summary_node(kb, add, edges, rob)

    status = "EQL ready · %d graph nodes · %d joints · %d CRAM classes" % (
        len(nodes),
        len(kb.joints),
        len(kb.classes),
    )
    return {
        "ok": True,
        "status": status,
        "nodes": nodes,
        "edges": edges,
        "details": details,
        "presets": get_presets(),
    }


# %% drill-down subgraphs -----------------------------------------------------
# Double-clicking a node in the UI asks for its inside view: package → its
# subpackages + top-level classes, subpackage → its classes (with inheritance
# edges), class → its base classes and every subclass in the repo.

#: at most this many classes are drawn in one drill-down view
CLASS_CAP = 150


def _view() -> tuple[list, list, dict, Callable]:
    """
    Fresh (nodes, edges, details, add) accumulators for one subgraph.
    """
    nodes, edges, details = [], [], {}

    def add(
        node_id: str, label: str, group: GraphGroup, lines: list[str], **extra: Any
    ) -> None:
        """
        Append one graph node (plus arbitrary extra fields) and its detail entry.
        """
        node = {
            "id": node_id,
            "label": label,
            "group": group.value,
            "title": "\n".join([label] + lines),
        }
        node.update(extra)
        nodes.append(node)
        details[node_id] = {"label": label, "group": group.value, "lines": lines}

    return nodes, edges, details, add


# %% the graph-panel tabs ---------------------------------------------------------
def view_payload(name: str) -> dict[str, Any]:
    """
    One tab of the graph panel.

    ``knowledge`` is the entity graph (the default, with drill-down); the others are
    structural views of the same demo that the UI can overlay with live status from the
    bridge (see :mod:`cram_viz.live.http`, ``/plan`` and ``/chart``).
    """
    kb = get_kb()
    if name == "knowledge":
        return graph_payload()
    if name == "kinematics":
        return _urdf_view(kb)
    if name == "plan":
        return _plan_view(kb)
    if name == "chart":
        return _chart_view(kb)
    return {"ok": False, "error": "unknown view: %s" % name}


def _chart_view(kb: KB) -> dict[str, Any]:
    """
    The (live-only) statechart tab.

    Motion statecharts only exist while giskardpy executes them: one is
    compiled per merged motion group and thrown away afterwards, and nothing
    of it is recorded into the bundle — the UI fills this view from the
    bridge's ``/chart`` while attached.
    """
    return {
        "ok": True,
        "crumb": "motion statechart",
        "nodes": [],
        "edges": [],
        "details": {},
        "layout": "hier",
        "live": "chart",
        "empty": "Motion statecharts are built and ticked at execution time. "
        "Start the demo with cram-viz-live and press ◉ Live — "
        "the statechart of the running motion group appears here, "
        "coloured by its node life cycle.",
    }


def _class_id(python_class: PythonClass) -> str:
    """
    Graph node id of a scanned class (module-qualified).
    """
    return python_class.module + "." + python_class.name


def _class_lines(python_class: PythonClass, drill_hint: bool = True) -> list[str]:
    """
    Detail lines shown for a class node.
    """
    lines = [
        "a PythonClass",
        "package: " + python_class.package,
        "module: " + python_class.module,
        "methods: %d" % python_class.methods,
    ]
    if python_class.bases:
        lines.append("bases: " + ", ".join(python_class.bases))
    if python_class.doc:
        lines.append(python_class.doc)
    if drill_hint:
        lines.append("double-click: inheritance view")
    return lines


def _add_classes(
    add: Callable,
    edges: list[dict[str, Any]],
    parent_id: str,
    shown: list[PythonClass],
    total: int,
) -> list[str]:
    """
    Add class nodes plus their on-screen inheritance edges to a view.

    :return: Extra detail lines for the parent (a truncation notice, if any).
    """
    name_to_id: dict[str, str] = {}
    for python_class in shown:
        class_id = _class_id(python_class)
        add(class_id, python_class.name, GraphGroup.PYCLASS, _class_lines(python_class))
        edges.append(_edge(parent_id, class_id, EdgeKind.PROP, "defines"))
        name_to_id.setdefault(python_class.name, class_id)
    for python_class in shown:
        for base in python_class.bases:
            if base in name_to_id and name_to_id[base] != _class_id(python_class):
                edges.append(
                    _edge(_class_id(python_class), name_to_id[base], EdgeKind.TYPE, "inherits")
                )
    if total > len(shown):
        return [
            "showing the %d largest of %d classes (by method count)"
            % (len(shown), total)
        ]
    return []


def _count_plan_nodes(tree: dict[str, Any]) -> int:
    """
    Number of nodes in a serialized plan tree.
    """
    return 1 + sum(_count_plan_nodes(child) for child in tree.get("children", []))


def expand_node(node_id: str) -> dict[str, Any] | None:
    """
    The inside view of a double-clicked node, or None if not drillable.
    """
    kb = get_kb()
    if node_id == kb.robot.name:  # robot → full URDF kinematic tree
        return _urdf_view(kb)
    if node_id == "plan":  # → the executed plan tree
        return _plan_view(kb)
    package = next((entry for entry in kb.packages if entry.name == node_id), None)
    if package:
        return _package_view(kb, package)
    subpackage = next(
        (entry for entry in kb.subpackages if entry.name == node_id), None
    )
    if subpackage:
        return _subpackage_view(kb, subpackage)
    python_class = next(
        (entry for entry in kb.classes if _class_id(entry) == node_id), None
    )
    if python_class:
        return _class_view(kb, python_class)
    return None


#: plan-node kind → node colour group of the graph panel
PLAN_GROUPS = {
    "ActionNode": GraphGroup.EVENT,
    "MotionNode": GraphGroup.ROBOT,
    "ConditionNode": GraphGroup.GOAL,
    "AttachmentNode": GraphGroup.OBJECT,
    "DetachmentNode": GraphGroup.OBJECT,
}

#: legend rows of the plan view
PLAN_LEGEND = [
    {"group": GraphGroup.EVENT.value, "label": "Action"},
    {"group": GraphGroup.ROBOT.value, "label": "Motion"},
    {"group": GraphGroup.GOAL.value, "label": "Condition"},
    {"group": GraphGroup.OBJECT.value, "label": "Attach / detach"},
    {"group": GraphGroup.IND.value, "label": "Other plan node"},
]


def _plan_view(kb: KB) -> dict[str, Any]:
    """
    The executed plan as a tree, one node per plan node the demo ran.

    The recorded statuses are thin on purpose: coraplex performs only the
    plan *root* (``Plan.perform`` → ``root.perform``), while
    ``ActionNode.notify`` merely expands its children into the merged motion
    statechart. So every inner node of a recorded tree reads ``CREATED``, and
    real per-step progress only shows up while the live bridge is attached
    (it derives it from the statechart life cycle).
    """
    scene, _ = load_scene()
    trees = scene.get("planTrees") or []
    nodes, edges, details, add = _view()
    counter = [0]

    def walk(tree: dict[str, Any], parent: str | None) -> None:
        """
        Add this plan node (with a freshly assigned id) and recurse into its children.
        """
        node_id = "pn%d" % counter[0]
        counter[0] += 1
        status = tree.get("status") or "CREATED"
        lines = ["a " + tree.get("kind", "PlanNode"), "status: " + status]
        if tree.get("arm"):
            lines.append("arm: " + tree["arm"])
        if tree.get("target"):
            lines.append("target: " + tree["target"])
        label = tree.get("label", "?")
        if label.endswith("Action"):
            label = label[: -len("Action")]
        add(
            node_id,
            label,
            PLAN_GROUPS.get(tree.get("kind"), GraphGroup.IND),
            lines,
            status=status,
        )
        if parent:
            edges.append(_edge(parent, node_id, EdgeKind.PROP, "has step"))
        for child in tree.get("children", []):
            walk(child, node_id)

    for tree in trees:
        walk(tree, None)
    return {
        "ok": True,
        "crumb": "executed plan",
        "nodes": nodes,
        "edges": edges,
        "details": details,
        "legend": PLAN_LEGEND,
        "layout": "hier",
        "live": "plan",
        "statusLegend": True,
        "empty": "No plan tree in this bundle — re-run cram-viz-onboard.",
    }


def _urdf_view(kb: KB) -> dict[str, Any]:
    """
    The scene robot's URDF as a kinematic tree.

    Every link is a node, every joint an edge (parent → child); movable joints are solid
    edges, fixed ones dashed. Links are coloured by robot part from the recorded
    annotation.
    """
    links, joints = load_urdf()
    nodes, edges, details, add = _view()
    if not links:
        return {
            "ok": True,
            "crumb": kb.robot.name + " · URDF (not found)",
            "nodes": [],
            "edges": [],
            "details": {},
        }

    scene, _ = load_scene()
    parts = (scene.get("robot") or {}).get("parts") or {}
    link_to_part = {
        link: part for part, part_links in parts.items() for link in part_links
    }

    def chain_group(link_name: str) -> GraphGroup:
        """
        The visual group (colour) a kinematic-chain link is bucketed into.
        """
        part = link_to_part.get(link_name, "").lower()
        if "gripper" in part or "hand" in part or "effector" in part:
            return GraphGroup.OBJECT  # grippers (teal)
        if "left" in part:
            return GraphGroup.ROBOT  # left arm (pink)
        if "right" in part:
            return GraphGroup.EVENT  # right arm (purple)
        lowered = link_name.lower()
        if any(
            keyword in lowered
            for keyword in ("head", "stereo", "sensor", "kinect", "camera", "laser")
        ):
            return GraphGroup.GOAL  # head / sensors (amber)
        return GraphGroup.CONCEPT  # base, torso, casters (green)

    # which joint drives each link (child link → its parent joint), for tooltips
    parent_joint = {joint["child"]: joint for joint in joints}
    for link in links:
        joint = parent_joint.get(link)
        lines = ["a URDF Link"]
        if joint:
            lines.append("joint: %s (%s)" % (joint["name"], joint["type"]))
            lines.append("parent link: " + joint["parent"])
        else:
            lines.append("root link")
        add("urdf:" + link, link, chain_group(link), lines)
    for joint in joints:
        if ("urdf:" + joint["parent"]) in details and (
            "urdf:" + joint["child"]
        ) in details:
            movable = joint["type"] not in ("fixed",)
            edges.append(
                _edge(
                    "urdf:" + joint["parent"],
                    "urdf:" + joint["child"],
                    EdgeKind.PROP if movable else EdgeKind.TYPE,
                    "%s (%s)" % (joint["name"], joint["type"]),
                )
            )
    revolute_count = sum(1 for joint in joints if joint["type"] == "revolute")
    details["urdf:" + links[0]]["lines"].append(
        "%d links · %d joints (%d movable)" % (len(links), len(joints), revolute_count)
    )
    legend = [
        {"group": GraphGroup.CONCEPT.value, "label": "Base / torso"},
        {"group": GraphGroup.ROBOT.value, "label": "Left arm"},
        {"group": GraphGroup.EVENT.value, "label": "Right arm"},
        {"group": GraphGroup.OBJECT.value, "label": "Grippers"},
        {"group": GraphGroup.GOAL.value, "label": "Head / sensors"},
    ]
    # force-directed, not hierarchical: the chains read better when the arms and
    # the sensor head spread out around the base than as one wide LR tree
    return {
        "ok": True,
        "crumb": kb.robot.name + " · URDF",
        "nodes": nodes,
        "edges": edges,
        "details": details,
        "legend": legend,
    }


def _package_view(kb: KB, package: Package) -> dict[str, Any]:
    """
    Inside view of a package: its subpackages and top-level classes.
    """
    nodes, edges, details, add = _view()
    subpackages = [entry for entry in kb.subpackages if entry.package == package.name]
    top_level = sorted(
        (
            entry
            for entry in kb.classes
            if entry.package == package.name and entry.subpackage == package.name
        ),
        key=lambda entry: -entry.methods,
    )
    add(
        package.name,
        package.name,
        GraphGroup.CONCEPT,
        [
            "a Package",
            package.description,
            "%d modules · %d classes" % (package.module_count, package.class_count),
        ],
    )
    for subpackage in subpackages:
        add(
            subpackage.name,
            subpackage.name.split(".", 1)[1],
            GraphGroup.KLASS,
            [
                "a SubPackage of " + subpackage.package,
                "%d modules · %d classes"
                % (subpackage.module_count, subpackage.class_count),
                "double-click to open",
            ],
        )
        edges.append(_edge(package.name, subpackage.name, EdgeKind.PROP, "contains"))
    note = _add_classes(add, edges, package.name, top_level[:CLASS_CAP], len(top_level))
    if note:
        details[package.name]["lines"] += note
    return {
        "ok": True,
        "crumb": package.name,
        "nodes": nodes,
        "edges": edges,
        "details": details,
    }


def _subpackage_view(kb: KB, subpackage: SubPackage) -> dict[str, Any]:
    """
    Inside view of a subpackage: its classes with inheritance edges.
    """
    nodes, edges, details, add = _view()
    classes = sorted(
        (entry for entry in kb.classes if entry.subpackage == subpackage.name),
        key=lambda entry: -entry.methods,
    )
    add(
        subpackage.name,
        subpackage.name.split(".", 1)[1],
        GraphGroup.KLASS,
        [
            "a SubPackage of " + subpackage.package,
            "%d modules · %d classes"
            % (subpackage.module_count, subpackage.class_count),
        ],
    )
    note = _add_classes(add, edges, subpackage.name, classes[:CLASS_CAP], len(classes))
    if note:
        details[subpackage.name]["lines"] += note
    return {
        "ok": True,
        "crumb": subpackage.name.split(".", 1)[1],
        "nodes": nodes,
        "edges": edges,
        "details": details,
    }


#: at most this many subclasses are drawn in a class inheritance view
SUBCLASS_CAP = 80


def _class_view(kb: KB, python_class: PythonClass) -> dict[str, Any]:
    """
    Inheritance view of one class: bases above, repo subclasses below.
    """
    nodes, edges, details, add = _view()
    class_id = _class_id(python_class)
    add(
        class_id,
        python_class.name,
        GraphGroup.PYCLASS,
        _class_lines(python_class, drill_hint=False),
    )
    # direct base classes: resolve inside the repo (same package preferred),
    # otherwise show them as external
    for base in python_class.bases:
        candidates = [entry for entry in kb.classes if entry.name == base]
        pick = next(
            (entry for entry in candidates if entry.package == python_class.package),
            candidates[0] if candidates else None,
        )
        if pick:
            base_id = _class_id(pick)
            if base_id not in details:
                add(base_id, pick.name, GraphGroup.PYCLASS, _class_lines(pick))
        else:
            base_id = "ext:" + base
            if base_id not in details:
                add(
                    base_id,
                    base,
                    GraphGroup.UPPER,
                    ["external base class (outside the repo)"],
                )
        edges.append(_edge(class_id, base_id, EdgeKind.TYPE, "inherits"))
    # every subclass in the repo (matched by base name)
    subclasses = [
        entry
        for entry in kb.classes
        if python_class.name in entry.bases and _class_id(entry) != class_id
    ]
    for subclass in subclasses[:SUBCLASS_CAP]:
        subclass_id = _class_id(subclass)
        if subclass_id not in details:
            add(subclass_id, subclass.name, GraphGroup.PYCLASS, _class_lines(subclass))
        edges.append(_edge(subclass_id, class_id, EdgeKind.TYPE, "inherits"))
    if len(subclasses) > SUBCLASS_CAP:
        details[class_id]["lines"].append(
            "showing %d of %d subclasses" % (SUBCLASS_CAP, len(subclasses))
        )
    return {
        "ok": True,
        "crumb": python_class.name,
        "nodes": nodes,
        "edges": edges,
        "details": details,
    }


#: static presets for the architecture side of the graph
ARCH_PRESETS = [
    {
        "text": "CRAM packages by size",
        "code": "set_of(package.name, package.class_count).ordered_by(package.class_count, descending=True)",
    },
    {
        "text": "all Designator classes",
        "code": "an(entity(python_class).where(python_class.name.endswith('Designator')))",
    },
    {
        "text": "where does EQL live?",
        "code": "set_of(python_class.name, python_class.module).where(in_('entity_query_language', python_class.module)).limit(15)",
    },
    {
        "text": "subclasses of Symbol",
        "code": "an(entity(python_class).where(in_('Symbol', python_class.bases)))",
    },
    {
        "text": "inside coraplex",
        "code": "an(entity(subpackage).where(subpackage.package == 'coraplex'))",
    },
]


def get_presets() -> list[dict[str, str]]:
    """
    Ready-made queries for the EQL panel.

    Scene presets are generated from the loaded scene, so they stay valid for any
    onboarded robot/environment; the architecture presets are static.
    """
    kb = get_kb()
    presets = [
        {"text": "which robot is this?", "code": "the(entity(robot))"},
        {"text": "which arms does it have?", "code": "an(entity(arm))"},
        {"text": "each arm and its gripper", "code": "set_of(arm.side, arm.gripper)"},
        {"text": "what is in the scene?", "code": "an(entity(object))"},
        {
            "text": "what gets moved?",
            "code": "an(entity(episode.picks).where(episode.picks != None))",
        },
    ]
    first_object = next((entry for entry in kb.objects if entry.kind == ObjectKind.OBJECT), None)
    if first_object:
        presets.append(
            {
                "text": "the %s" % first_object.label.lower(),
                "code": "the(entity(object).where(object.name == '%s'))" % first_object.name,
            }
        )
    manipulation = next((episode for episode in kb.episodes if episode.picks), None)
    if manipulation:
        if manipulation.places_at:
            presets.append(
                {
                    "text": "where does it place them?",
                    "code": "the(entity(episode.places_at).where(episode.name == '%s'))"
                    % manipulation.name,
                }
            )
        if manipulation.performed_by:
            presets.append(
                {
                    "text": "which arm does '%s'?" % manipulation.name,
                    "code": "the(entity(episode.performed_by).where(episode.name == '%s'))"
                    % manipulation.name,
                }
            )
    return presets + ARCH_PRESETS


if __name__ == "__main__":
    # smoke test: run every preset against the active scene
    logging.basicConfig(level=logging.INFO)
    for preset in get_presets():
        result = run_query(preset["code"])
        if result["ok"]:
            logging.info("OK   %-32s -> %d rows", preset["text"], result["count"])
        else:
            logging.error("FAIL %-32s -> %s", preset["text"], result["error"])
