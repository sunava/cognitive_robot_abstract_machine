"""
Bundles a live world's *current* model into a throwaway scene.

Serializes the world object the bridge is attached to — every robot's subtree as its own
model, everything else as the environment model — through the same
:class:`~cramera.onboard.world_to_urdf.UrdfDocument` serializer every recorded scene
goes through. Whatever a demo built its world from, and however it built it, is what
the viewer shows; no model source tracking or manual onboarding is involved.

A build is idempotent: while the world model is unchanged, the existing bundle is left
untouched — a viewer may be downloading its files at any moment, and deleting them
mid-flight breaks the page that requested the rebuild in the first place.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional, Sequence

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from cramera import paths
from cramera.generated_json import GeneratedJson
from cramera.live.bridge import Bridge
from cramera.streamed_bodies import is_streamed
from cramera.onboard.bundle_urdf import BundleReport
from cramera.onboard.world_to_urdf import UrdfDocument
from cramera.robot_parts import (
    RobotPartAnnotation,
    robot_base_link,
    robot_model_names,
    robot_prefix,
)

BUILD_LOCK = threading.Lock()
"""
Serializes bundling across this process.

The output directory is one fixed path, and a build clears it before writing it, so two
overlapping builds delete and overwrite each other's files. The bridge answers on a
threading HTTP server and the viewer keeps asking for the live scene while a demo starts
up, which makes that overlap routine rather than rare.
"""

ENVIRONMENT_MODEL_NAME = "environment"
"""
Name of the model holding every world body outside the robot and the object overlay.
"""

MESH_SUBDIRECTORY = "live"
"""
Directory bundled meshes nest under inside the scene's ``meshes/`` tree.
"""


@dataclass(frozen=True)
class WorldModelsBundle:
    """
    A live world's geometry, serialized into a scene bundle's ``models``/``robot``
    fields.
    """

    models: List[Dict[str, Any]]
    """
    The scene's ``models`` entries: the environment (if any bodies remain outside the
    robots and the object overlay), followed by one entry per bound robot, each carrying
    the world-instance prefix its live base pose is streamed under.
    """

    robot: Optional[Dict[str, Any]]
    """
    The scene's ``robot`` field — the *first* robot, for a viewer that knows only one —
    or None without a bound robot.
    """

    robots: List[Dict[str, Any]]
    """
    The scene's ``robots`` field: one payload per bound robot, in the same order as the
    robot entries of :attr:`models`.
    """

    missing_assets: List[str]
    """
    Every asset a model referenced but could not resolve, across all models.
    """


def bundle_world_models(
    world: World,
    robots: Sequence[AbstractRobot],
    output_directory: Path,
    mesh_subdirectory: str,
) -> WorldModelsBundle:
    """
    Serialize a world's robots and environment bodies into URDF models on disk.

    Every robot becomes its own model, named after its class and grafted at the origin,
    because the viewer applies each robot's streamed base pose on top of its model —
    baking a second robot into the environment is what used to leave it standing still
    while its joints moved.

    Shared between the throwaway ``__live__`` bundle and a finalized live recording
    (:mod:`cramera.live.recording_bundle`) — both bundle the same live world through the
    same :class:`~cramera.onboard.world_to_urdf.UrdfDocument` serializer, and would
    otherwise duplicate the environment/robot split and the ``UrdfDocument.of_bodies``
    wiring.

    :param world: The world whose bodies are serialized.
    :param robots: The world's robot annotations, in publication order; empty to bundle
        everything as environment.
    :param output_directory: Directory the URDF and mesh files are written into.
    :param mesh_subdirectory: Directory the meshes nest under inside
        ``output_directory``.
    """
    names = robot_model_names(robots)
    subtrees = [_robot_bodies(world, robot) for robot in robots]
    claimed = {body for subtree in subtrees for body in subtree}
    models: List[Dict[str, Any]] = []
    environment_bodies = [
        body
        for body in world.bodies_topologically_sorted
        if body not in claimed and not is_streamed(body)
    ]
    if environment_bodies:
        report = UrdfDocument.of_bodies(
            environment_bodies,
            ENVIRONMENT_MODEL_NAME,
            str(output_directory),
            mesh_subdirectory,
        )
        models.append(_model_payload(report, is_robot=False))
    for robot, name, robot_bodies in zip(robots, names, subtrees):
        if not robot_bodies:
            continue
        report = UrdfDocument.of_bodies(
            robot_bodies,
            name,
            str(output_directory),
            mesh_subdirectory,
            identity_root=robot.root,
        )
        models.append(_model_payload(report, is_robot=True, prefix=robot_prefix(robot)))
    missing_assets = sorted(
        {missing for model in models for missing in model.pop("missing")}
    )
    payloads = [_robot_payload(robot, name) for robot, name in zip(robots, names)]
    return WorldModelsBundle(
        models=models,
        robot=payloads[0] if payloads else None,
        robots=payloads,
        missing_assets=missing_assets,
    )


def build_live_scene(bridge: Bridge) -> Optional[str]:
    """
    Bundle the live world's current model into a throwaway scene.

    :param bridge: The live bridge whose current world is bundled.
    :return: :data:`cramera.paths.LIVE_SCENE_NAME`, the scene name to navigate the
        viewer to, or None while no world is attached yet.
    """
    if bridge.world is None:
        return None
    output_directory = paths.local_scenes_directory() / paths.LIVE_SCENE_NAME
    with BUILD_LOCK:
        signature = bridge.bundle_signature()
        if _existing_signature(output_directory) == signature:
            return paths.LIVE_SCENE_NAME
        return _write_bundle(bridge, output_directory, signature)


def _existing_signature(output_directory: Path) -> Optional[str]:
    """
    The signature the existing bundle was built from, or None without a readable one.

    :param output_directory: Directory the previous bundle was written to.
    """
    scene = GeneratedJson(output_directory / "scene.json").read()
    if not isinstance(scene, dict):
        return None
    return scene.get("bundleSignature")


def _write_bundle(bridge: Bridge, output_directory: Path, signature: str) -> str:
    """
    Clear the throwaway scene's directory and serialize the world into it.

    Only ever called while :data:`BUILD_LOCK` is held.

    :param bridge: The live bridge whose current world is bundled.
    :param output_directory: Directory the bundle is written to, cleared first.
    :param signature: The digest of the world model this bundle is built from, recorded
        so the next build can tell whether anything changed.
    """
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True)
    geometry = bundle_world_models(
        bridge.world, bridge.robots, output_directory, MESH_SUBDIRECTORY
    )
    scene = {
        "name": paths.LIVE_SCENE_NAME,
        "models": geometry.models,
        "robot": geometry.robot,
        "robots": geometry.robots,
        "objects": [],
        "segments": [],
        "missingAssets": geometry.missing_assets,
        # the world was attached before this bundle was built; kept for the viewer,
        # which rebuilds a bundle that reports False
        "worldBound": True,
        # what this bundle was built from; while it is unchanged, /live_scene leaves
        # the bundle untouched instead of deleting files a viewer may be downloading
        "bundleSignature": signature,
    }
    (output_directory / "scene.json").write_text(json.dumps(scene, indent=1))
    return paths.LIVE_SCENE_NAME


def _robot_bodies(world: World, robot: Optional[AbstractRobot]) -> List[Body]:
    """
    One robot's subtree in serialization order, or an empty list without a robot.

    Overlay objects are left out here just as they are left out of the environment: a
    grasped one is inside the subtree, and the model is not rebuilt when it is put down
    again, so baking it in would leave a copy of it stuck to the hand.

    :param world: The world the robot lives in.
    :param robot: The robot annotation, or None.
    """
    if robot is None:
        return []
    subtree = set(world.get_kinematic_structure_entities_of_branch(robot.root))
    return [
        body
        for body in world.bodies_topologically_sorted
        if body in subtree and not is_streamed(body)
    ]


def _model_payload(
    report: BundleReport, is_robot: bool, prefix: str = ""
) -> Dict[str, Any]:
    """
    One entry of the scene's ``models`` list.

    :param report: What the serializer wrote for this model.
    :param is_robot: Whether the entry is a robot rather than the environment.
    :param prefix: The model's world-instance prefix, under which its live base pose is
        streamed; empty for the environment, whose bodies belong to no robot.
    """
    return {
        "name": report.name,
        "urdf": "%s.urdf" % report.name,
        "prefix": prefix,
        "robot": is_robot,
        "links": len(report.links),
        "movableJoints": report.movable_joints,
        "missing": report.missing,
    }


def _robot_payload(robot: AbstractRobot, name: str) -> Dict[str, Any]:
    """
    One entry of the scene's ``robots`` list; the first of them doubles as the scene's
    single ``robot`` field, for a viewer that knows only one robot.

    :param robot: The robot's semantic annotation.
    :param name: The robot's model name, unique within the bundle (see
        :func:`~cramera.robot_parts.robot_model_names`).
    """
    part_annotations = RobotPartAnnotation.of_robot(robot)
    return {
        "name": name,
        "prefix": robot_prefix(robot),
        "class": type(robot).__name__,
        "baseBody": robot_base_link(robot),
        "parts": {annotation.name: annotation.links for annotation in part_annotations},
        "partAnnotations": [annotation.to_payload() for annotation in part_annotations],
    }
