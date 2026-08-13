"""
Bundles a live world's *current* state into the reserved live scene.

Reuses exactly the bundler every recorded scene already goes through (see
:func:`~cramera.onboard.bundle_urdf.bundle_model`) instead of serving geometry from
scratch, so the viewer always shows what the running demo actually looks like right now,
with no manual onboarding step. Each world configuration is bundled once, into a cache
keyed by its signature, and the reserved live scene is only a pointer at the current
world's entry — attaching to a world seen before is instant, and switching worlds never
deletes files a viewer may still be downloading.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional

from semantic_digital_twin.robots.robot_parts import AbstractRobot

from cramera import paths
from cramera.generated_json import GeneratedJson
from cramera.live.bridge import Bridge, ModelBundleContext
from cramera.onboard.bundle_urdf import BundledModel, bundle_model
from cramera.robot_parts import RobotPartAnnotation

BUILD_LOCK = threading.Lock()
"""
Serializes bundling and live-scene repointing across this process.

Two overlapping builds of the same configuration would clear and overwrite each other's
cache entry. The bridge answers on a threading HTTP server and the viewer keeps asking
for the live scene while a demo starts up, which makes that overlap routine rather than
rare.
"""


def build_live_scene(bridge: Bridge) -> Optional[str]:
    """
    Bundle every model source the live world was built from into a throwaway scene.

    :param bridge: The live bridge whose current world is bundled.
    :return: :data:`cramera.paths.LIVE_SCENE_NAME`, the scene name to navigate the
        viewer to, or None while the demo has neither parsed a model source nor attached
        a world yet. A fully procedural world builds a scene without models; its bodies
        reach the viewer through the object overlay instead.
    """
    context = bridge.model_bundle_context()
    if not context.sources and not context.world_body_names:
        return None
    live_path = paths.scenes_directory() / paths.LIVE_SCENE_NAME
    with BUILD_LOCK:
        signature = context.signature()
        if _existing_signature(live_path) == signature:
            return paths.LIVE_SCENE_NAME
        cache_entry = _cache_entry_path(signature)
        if _existing_signature(cache_entry) != signature:
            _write_bundle(context, cache_entry, signature)
        _point_live_scene_at(cache_entry, live_path)
        return paths.LIVE_SCENE_NAME


def _cache_entry_path(signature: str) -> Path:
    """
    Where the bundle of one world configuration lives in the cache.

    Inside the scenes directory, so the server's path-traversal guard (which resolves
    symlinks) keeps serving the live scene.

    :param signature: The digest of what the bundle is built from.
    """
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    return paths.scenes_directory() / paths.LIVE_BUNDLE_CACHE_NAME / digest


def _point_live_scene_at(cache_entry: Path, live_path: Path) -> None:
    """
    Make the reserved live scene a pointer at one cached bundle.

    Switching worlds only redirects the pointer — another world's bundle files stay in
    the cache, so a viewer still downloading them is never broken and the next attach to
    that world is instant.

    :param cache_entry: The cached bundle the live scene should serve.
    :param live_path: The reserved live scene path.
    """
    if live_path.is_symlink():
        live_path.unlink()
    elif live_path.exists():
        # a bundle written in place by an older version
        shutil.rmtree(live_path)
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.symlink_to(cache_entry, target_is_directory=True)


def _existing_signature(output_directory: Path) -> Optional[str]:
    """
    The signature the existing bundle was built from, or None without a readable one.

    :param output_directory: Directory the previous bundle was written to.
    """
    scene = GeneratedJson(output_directory / "scene.json").read()
    if not isinstance(scene, dict):
        return None
    return scene.get("bundleSignature")


def _write_bundle(
    context: ModelBundleContext, output_directory: Path, signature: str
) -> str:
    """
    Write this world configuration's bundle into its cache entry.

    Only ever called while :data:`BUILD_LOCK` is held.

    :param context: What the bridge reported about its current world.
    :param output_directory: Cache entry the bundle is written to, cleared first when a
        stale or partial one is in the way.
    :param signature: The digest of what this bundle is built from, recorded so a later
        build can tell whether this entry still matches.
    """
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True)
    models = [
        bundle_model(
            tracked.path,
            tracked.bundler,
            context.world_body_names,
            context.base_body,
            str(output_directory),
        )
        for tracked in context.sources
    ]
    if context.world_body_names:
        # a long-running process may parse one world after another; the scene shows
        # only the models the currently executing world contains. Without a world
        # there is nothing to check against, and the early bundle keeps everything.
        models = [
            model
            for model in models
            if _model_in_world(model.report.links, context.world_body_names)
        ]
    scene = {
        "name": paths.LIVE_SCENE_NAME,
        "models": [model.to_payload() for model in models],
        "robot": _robot_payload(context.robot, context.base_body),
        "objects": [],
        "segments": [],
        "missingAssets": _missing_assets(models),
        # what this bundle was built from: reused unchanged, rebuilt on any change,
        # and compared by the viewer against /info to notice its page went stale
        "bundleSignature": signature,
    }
    (output_directory / "scene.json").write_text(json.dumps(scene, indent=1))
    return paths.LIVE_SCENE_NAME


def _model_in_world(links: List[str], world_body_names: List[str]) -> bool:
    """
    Whether any of a model's links exists in the composed world.

    Matches both prefixed (``pr2_1/base_link``) and unprefixed world instances of a
    link.

    :param links: The model's own link names, in document order.
    :param world_body_names: Every body name in the composed world.
    """
    link_set = set(links)
    return any(name.rpartition("/")[2] in link_set for name in world_body_names)


def _robot_payload(
    robot: Optional[AbstractRobot], base_body: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    The scene's ``robot`` field, or None if no robot is bound.

    :param robot: The robot's semantic annotation, or None if no robot is bound.
    :param base_body: The robot's unprefixed base link name, or None.
    """
    if robot is None:
        return None
    root_name = str(robot.root.name)
    prefix = root_name.split("/", 1)[0] if "/" in root_name else ""
    part_annotations = RobotPartAnnotation.of_robot(robot)
    return {
        "name": type(robot).__name__.lower(),
        "prefix": prefix,
        "baseBody": base_body,
        "parts": {annotation.name: annotation.links for annotation in part_annotations},
        "partAnnotations": [annotation.to_payload() for annotation in part_annotations],
    }


def _missing_assets(models: List[BundledModel]) -> List[str]:
    """
    Every unresolved mesh reference across all bundled models, sorted and deduplicated.

    :param models: The models just bundled.
    """
    return sorted({missing for model in models for missing in model.report.missing})
