"""
Scan on-disk scene bundles and maintain the ``index.json`` the web viewer reads.

Kept free of the CRAM stack's own dependencies, so a scenes directory can be indexed
by a plain Python interpreter - a static-hosting deploy pipeline has no reason to
install the full simulation stack just to rebuild this one file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional


# %% scanning bundles on disk
def scene_environment(models: List[Dict[str, Any]]) -> Optional[str]:
    """
    The name of a scene's environment, or ``None`` for a bench-only scene.

    A scene's models are either its robot or the environment it stands in; the viewer's
    header picker shows this alongside the robot name to pick out one onboarded scene.
    """
    environment_models = [model["name"] for model in models if not model["robot"]]
    return "+".join(environment_models) if environment_models else None


def scan_scenes(scenes_dir: Path) -> List[Dict[str, Any]]:
    """
    Every onboarded scene bundle under ``scenes_dir``, with its robot/environment
    identity.

    Read straight off the bundles on disk rather than accumulated incrementally, so a
    scene folder that was removed or renamed since it was indexed cannot leave a stale
    entry behind.
    """
    entries = []
    for bundle_dir in sorted(scenes_dir.iterdir()):
        scene_path = bundle_dir / "scene.json"
        if not scene_path.is_file():
            continue
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "name": bundle_dir.name,
                "robot": scene["robot"]["name"],
                "environment": scene_environment(scene["models"]),
            }
        )
    return entries


# %% writing the index
def write_json(path: Path, payload: Any, indent: Optional[int] = None) -> None:
    """
    Write a bundle file, replacing it only once it is complete.

    A bundle is the artifact of a long recording, so a failure part-way through a write
    must not leave a truncated file behind.
    """
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    temporary.replace(path)


def update_scene_index(path: Path, name: str) -> None:
    """
    Register a freshly written scene in the index the viewer reads.

    The ``scenes`` list is rebuilt from every bundle actually on disk, each carrying its
    robot/environment identity for the viewer's robot/environment picker. ``default`` is
    filled in on the first scene onboarded and left alone after that.
    """
    index: Dict[str, Any] = {}
    if path.is_file():
        index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        index = {}
    index["scenes"] = scan_scenes(path.parent)
    index.setdefault("default", name)
    write_json(path, index, indent=1)


def repair_default(path: Path) -> None:
    """
    Rescan ``path``'s scenes and point ``default`` at a real bundle if it is not one.

    A hand-maintained scenes index can drift from the bundles actually shipped
    alongside it - for example a renamed or removed bundle its ``default`` still
    names. Restores the index to something the viewer can load without a ``?scene=``
    query param, without touching an already-valid ``default``.

    :param path: the ``index.json`` path, directly inside the scenes directory
    """
    index = json.loads(path.read_text(encoding="utf-8"))
    scenes = scan_scenes(path.parent)
    index["scenes"] = scenes
    names = [entry["name"] for entry in scenes]
    if names and index.get("default") not in names:
        index["default"] = names[0]
    write_json(path, index, indent=1)


# %% command line entry point
def main() -> None:
    """
    ``python -m cram_viz.onboard.scene_index <scenes-dir>`` - repair its index in place.
    """
    repair_default(Path(sys.argv[1]) / "index.json")


if __name__ == "__main__":
    main()
