"""
The scenes index (``index.json``) the viewer's pickers read.

Kept free of the heavy onboarding imports (URDF/Gazebo/MJCF parsers, ``runpy``, the
monkey-patched :class:`~cramera.onboard.demo.Recorder`) so the always-on static file
server (:mod:`cramera.server`) and the live bridge's recording finalizer
(:mod:`cramera.live.recording_bundle`) can register a scene without pulling in the
offline onboarding pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional

from cramera import paths
from cramera.generated_json import GeneratedJson, write_json_atomically

RESERVED_SCENE_NAMES = (paths.LIVE_SCENE_NAME, paths.RECORDING_SCENE_NAME)
"""
Throwaway bundle names that are never something a user onboarded or saved, and must
never show up as a robot choice in the real picker.
"""

SCENE_PATH_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
"""
What one segment of a scene's identity (robot, environment, task or version) may look
like: safe as a single path component, without resorting to escaping or length limits a
filesystem might reject.
"""

NO_ENVIRONMENT_SEGMENT = "_none"
"""
Directory segment recorded in place of the environment when a scene has none (a robot
recorded alone, with no environment model) — keeps a scene's path at a fixed four
segments, so the viewer's picker can always narrow robot -> environment -> task ->
version instead of falling back to a flat list of names.
"""

SCENE_IDENTITY_SEGMENT_COUNT = 4
"""
How many path segments a scene name is made of: robot, environment, task, version.
"""


class InvalidSceneName(Exception):
    """
    Raised by :func:`validate_scene_name` when a user-given name is not a safe, non-
    reserved scene name.
    """


def validate_scene_path_segment(value: str, field: str) -> str:
    """
    Check that one segment of a scene's identity is safe to use as a directory name.

    :param value: The segment to validate.
    :param field: Name of the parameter it validates, used in the error message.
    :return:``value`` unchanged, for chaining.
    :raises InvalidSceneName: If ``value`` is not exactly letters, digits, ``_`` or
        ``-`` (1-64 characters).
    """
    if not SCENE_PATH_SEGMENT_PATTERN.match(value):
        raise InvalidSceneName(
            "%s must be 1-64 characters of letters, digits, '_' or '-'" % field
        )
    return value


def validate_scene_name(name: str) -> str:
    """
    Check that a user-given name is a safe scene identity to save or look up.

    A scene name is exactly four segments — robot, environment, task, version — joined
    by ``/``, matching the four-level directory a scene bundle is stored under.

    :param name: The name to validate.
    :return:``name`` unchanged, for chaining.
    :raises InvalidSceneName: If ``name`` is not exactly four safe segments joined by
        ``/``, or its robot segment is one of :data:`RESERVED_SCENE_NAMES`.
    """
    segments = name.split("/")
    if len(segments) != SCENE_IDENTITY_SEGMENT_COUNT:
        raise InvalidSceneName(
            "a scene name must be exactly robot/environment/task/version"
        )
    for segment in segments:
        validate_scene_path_segment(segment, "each segment of a scene name")
    if segments[0] in RESERVED_SCENE_NAMES:
        raise InvalidSceneName("'%s' is a reserved robot name" % segments[0])
    return name


def compose_scene_name(
    robot: str, environment: Optional[str], task: str, version: str
) -> str:
    """
    Build the ``/``-joined scene name a (robot, environment, task, version) identity is
    saved and looked up under.

    :param robot: Name of the robot the scene was recorded with.
    :param environment: Name of the scene's environment, or None for a bench-only scene.
    :param task: Name of the task the recording demonstrates.
    :param version: Which recorded version of that task this is.
    :raises InvalidSceneName: If any segment is unsafe, or ``robot`` is a reserved name.
    """
    environment_segment = (
        environment if environment is not None else NO_ENVIRONMENT_SEGMENT
    )
    return validate_scene_name("/".join([robot, environment_segment, task, version]))


@dataclass
class SceneIndexEntry:
    """
    One onboarded (or saved) scene bundle, as ``index.json`` advertises it to the
    viewer.

    The viewer's header offers a robot, an environment, a task and a version separately,
    narrowing each in turn to the bundle actually recorded for that combination — these
    entries are what it looks that up in.
    """

    name: str
    """
    The bundle's ``robot/environment/task/version`` directory path, which is also its
    ``?scene=`` value.
    """

    robot: str
    """
    Name of the robot the scene was recorded with.
    """

    environment: Optional[str]
    """
    Name of the scene's environment, or None for a bench-only scene.
    """

    task: str
    """
    Name of the task the recording demonstrates.
    """

    version: str
    """
    Which recorded version of that task this is.
    """

    @classmethod
    def of_directory(cls, scenes_directory: Path) -> List[SceneIndexEntry]:
        """
        Every onboarded bundle under a scenes directory, in name order.

        Only a ``scene.json`` exactly four directory levels down counts as a bundle — a
        throwaway bundle (see :data:`RESERVED_SCENE_NAMES`) is written one level down
        and so is never matched here, needing no separate exclusion.

        :param scenes_directory: Directory holding the scene bundles.
        """
        entries = []
        for scene_path in sorted(scenes_directory.glob("*/*/*/*/scene.json")):
            robot, environment_segment, task, version = scene_path.parent.relative_to(
                scenes_directory
            ).parts
            entries.append(
                cls(
                    name="/".join((robot, environment_segment, task, version)),
                    robot=robot,
                    environment=(
                        None
                        if environment_segment == NO_ENVIRONMENT_SEGMENT
                        else environment_segment
                    ),
                    task=task,
                    version=version,
                )
            )
        return entries

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape ``index.json`` carries.
        """
        return {
            "name": self.name,
            "robot": self.robot,
            "environment": self.environment,
            "task": self.task,
            "version": self.version,
        }


def write_scene_index(path: Path, name: str) -> None:
    """
    Register a freshly written scene in the index the viewer reads.

    The ``scenes`` list is rebuilt from the bundles actually on disk, each carrying its
    robot/environment/task/version identity for the viewer's pickers, so a bundle that
    was removed or renamed since it was indexed cannot leave a stale entry behind.
    ``default`` is filled in on the first scene onboarded and left alone after that.

    :param path: Path of the scene index file.
    :param name: Name of the scene to register.
    """
    index: Dict[str, Any] = {}
    if path.is_file():
        index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        index = {}
    index["scenes"] = [
        entry.to_payload() for entry in SceneIndexEntry.of_directory(path.parent)
    ]
    index.setdefault("default", name)
    write_json_atomically(path, index, indent=1)


def merged_scene_index() -> Dict[str, Any]:
    """
    The ``index.json`` the frontend fetches: the shared scenes plus local recordings
    saved under :func:`cramera.paths.local_scenes_directory`, with a local scene
    shadowing a shared one of the same name.

    Both roots are read straight from disk rather than from a persisted merged file, so
    a recording that was just saved (or discarded) shows up immediately.
    """
    shared_directory = paths.scenes_directory()
    local_directory = paths.local_scenes_directory()
    by_name: Dict[str, SceneIndexEntry] = {}
    if shared_directory.is_dir():
        by_name.update(
            {
                entry.name: entry
                for entry in SceneIndexEntry.of_directory(shared_directory)
            }
        )
    if local_directory != shared_directory and local_directory.is_dir():
        by_name.update(
            {
                entry.name: entry
                for entry in SceneIndexEntry.of_directory(local_directory)
            }
        )
    shared_index = GeneratedJson(shared_directory / "index.json").read()
    default = shared_index.get("default") if isinstance(shared_index, dict) else None
    return {
        "default": default,
        "scenes": [by_name[name].to_payload() for name in sorted(by_name)],
    }
