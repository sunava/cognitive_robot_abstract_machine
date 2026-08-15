"""
Recording a set of coraplex demos into scene bundles with one command.

Each demo is recorded by :mod:`cramera.onboard.demo` in a process of its own: the
onboarder patches the parsers and the executor of the interpreter it runs in and ends
that process once the bundle is written, so a second demo cannot be recorded after it in
the same process.

Usage (the interpreter needs the CRAM stack on it)::

    cramera-onboard-demos path/to/demo_cutting.py path/to/demo_pouring.py

Every demo becomes a scene named after its file, with a leading ``demo_`` dropped, so
the command above yields the scenes ``cutting`` and ``pouring``.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import List, Sequence

from cramera import paths
from cramera.logging_setup import get_logger
from cramera.onboard.scene_index import validate_scene_name

logger = get_logger(__name__)

ONBOARDER_MODULE = "cramera.onboard.demo"
"""
The module recording one demo, run as ``python -m`` in a process of its own.
"""

DEMO_FILE_PREFIX = "demo_"
"""
Conventional prefix of a demo file's name, dropped from the scene name it yields.
"""

SUCCESSFUL_EXIT_CODE = 0
"""
Exit code the onboarder reports a written bundle with.
"""

RECORDING_FAILED_EXIT_CODE = 1
"""
Exit code :func:`main` reports when at least one demo failed to record.
"""


class MissingDemoFile(Exception):
    """
    Raised by :meth:`DemoScene.of_demo_file` when a given path is not a file.
    """


class DuplicateSceneName(Exception):
    """
    Raised by :meth:`DemoScene.of_demo_files` when two of the given demos yield the same
    scene name, which would record the second one over the first.
    """


# %% one demo, one scene
@dataclass
class DemoScene:
    """
    A demo file and the scene bundle its recording is written to.
    """

    demo_file: Path
    """
    The demo to run under the onboarder.
    """

    name: str
    """
    Name of the scene bundle, which is also its ``?scene=`` value in the viewer.
    """

    @classmethod
    def of_demo_file(cls, demo_file: Path) -> DemoScene:
        """
        The scene a demo file is recorded as, named after the file itself.

        :param demo_file: Path of the demo to record.
        :raises MissingDemoFile: If ``demo_file`` is not a file.
        :raises cramera.onboard.scene_index.InvalidSceneName: If the file's name does
            not yield a safe scene name.
        """
        if not demo_file.is_file():
            raise MissingDemoFile("'%s' is not a demo file" % demo_file)
        stem = demo_file.stem
        name = (
            stem[len(DEMO_FILE_PREFIX) :] if stem.startswith(DEMO_FILE_PREFIX) else stem
        )
        return cls(demo_file=demo_file.resolve(), name=validate_scene_name(name))

    @classmethod
    def of_demo_files(cls, demo_files: Sequence[Path]) -> List[DemoScene]:
        """
        The scenes a set of demo files is recorded as, in the given order.

        :param demo_files: Paths of the demos to record.
        :raises DuplicateSceneName: If two of the demos are named alike.
        """
        scenes = [cls.of_demo_file(demo_file) for demo_file in demo_files]
        repeated = [
            name
            for name, count in Counter(scene.name for scene in scenes).items()
            if count > 1
        ]
        if repeated:
            raise DuplicateSceneName(
                "these demos would record over each other: %s"
                % ", ".join(sorted(repeated))
            )
        return scenes

    def onboard_command(self, scenes_directory: Path) -> List[str]:
        """
        The command recording this demo, run in a process of its own.

        :param scenes_directory: Directory the scene bundle is written under.
        """
        return [
            sys.executable,
            "-m",
            ONBOARDER_MODULE,
            str(self.demo_file),
            "--name",
            self.name,
            "--out",
            str(scenes_directory),
        ]


@dataclass
class SceneRecording:
    """
    What recording one demo produced.
    """

    scene: DemoScene
    """
    The demo that was run and the scene it was recorded as.
    """

    exit_code: int
    """
    Exit code of the onboarder process.
    """

    @property
    def succeeded(self) -> bool:
        """
        Whether the onboarder wrote the bundle.
        """
        return self.exit_code == SUCCESSFUL_EXIT_CODE


# %% recording a set of demos
def record_scenes(
    scenes: Sequence[DemoScene], scenes_directory: Path
) -> List[SceneRecording]:
    """
    Record every demo, each in a process of its own.

    A demo that fails does not stop the ones after it: a set of demos takes long enough
    that the recordings which do succeed are worth keeping, and the failure is reported
    in the returned results.

    :param scenes: The demos to record and the scenes they are written to.
    :param scenes_directory: Directory the scene bundles are written under.
    """
    recordings = []
    for position, scene in enumerate(scenes, start=1):
        logger.info(
            "recording %d/%d: %s -> scene '%s'",
            position,
            len(scenes),
            scene.demo_file,
            scene.name,
        )
        exit_code = subprocess.run(scene.onboard_command(scenes_directory)).returncode
        recordings.append(SceneRecording(scene=scene, exit_code=exit_code))
    return recordings


def main() -> int:
    """
    ``cramera-onboard-demos`` — record several demos into scene bundles.

    :return: The process' exit code, non-zero if any demo failed to record.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "demos", nargs="+", type=Path, help="paths to the coraplex demo .py files"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.scenes_directory(),
        help="scenes directory (default: CRAMERA_SCENES or ~/.cramera/scenes)",
    )
    arguments = parser.parse_args()

    recordings = record_scenes(DemoScene.of_demo_files(arguments.demos), arguments.out)
    for recording in recordings:
        logger.info(
            "  %s: %s",
            "recorded" if recording.succeeded else "failed",
            recording.scene.name,
        )
    failed = [recording for recording in recordings if not recording.succeeded]
    if failed:
        return RECORDING_FAILED_EXIT_CODE
    logger.info("scenes written to %s", arguments.out)
    return SUCCESSFUL_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
