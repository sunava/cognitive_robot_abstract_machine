"""
Tests for recording a set of demos into scene bundles.

Running a demo needs the whole CRAM stack, so what is covered here is everything around
that: which scene a demo file is recorded as, the command each recording is run with,
and that one failing demo neither stops the others nor is reported as written.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typing_extensions import List

from cramera import paths
from cramera.onboard import demo_scenes
from cramera.onboard.demo_scenes import (
    DemoScene,
    DuplicateSceneName,
    MissingDemoFile,
    SceneRecording,
)
from cramera.onboard.scene_index import InvalidSceneName

DEMO_SOURCE = "def main() -> None:\n    pass\n"
"""
Body of the demo files the tests point at; never run, only found on disk.
"""


@dataclass
class FinishedProcess:
    """
    The part of :class:`subprocess.CompletedProcess` a recording reads back.
    """

    returncode: int
    """
    Exit code the onboarder process ended with.
    """


@dataclass
class ProcessLauncher:
    """
    Stands in for :func:`subprocess.run`, remembering every command it was asked to run
    and answering each with the next queued exit code.
    """

    exit_codes: List[int] = field(default_factory=list)
    """
    Exit code to answer with, one per expected command; a command beyond them ends
    successfully.
    """

    commands: List[List[str]] = field(default_factory=list)
    """
    The commands launched so far, in order.
    """

    def __call__(self, command: List[str]) -> FinishedProcess:
        self.commands.append(command)
        if len(self.commands) > len(self.exit_codes):
            return FinishedProcess(returncode=demo_scenes.SUCCESSFUL_EXIT_CODE)
        return FinishedProcess(returncode=self.exit_codes[len(self.commands) - 1])


def write_demo(directory: Path, file_name: str) -> Path:
    """
    A demo file on disk, so a scene can be derived from it.

    :param directory: Directory to write the demo into.
    :param file_name: Name of the demo file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    demo_file = directory / file_name
    demo_file.write_text(DEMO_SOURCE, encoding="utf-8")
    return demo_file


# %% naming a demo's scene
class TestDemoSceneNaming:
    def test_the_demo_prefix_is_dropped(self, tmp_path):
        scene = DemoScene.of_demo_file(write_demo(tmp_path, "demo_cutting.py"))

        assert scene.name == "cutting"

    def test_a_file_without_the_prefix_keeps_its_name(self, tmp_path):
        scene = DemoScene.of_demo_file(write_demo(tmp_path, "kitchen.py"))

        assert scene.name == "kitchen"

    def test_the_demo_file_is_resolved_to_an_absolute_path(self, tmp_path, monkeypatch):
        write_demo(tmp_path, "demo_wiping.py")
        monkeypatch.chdir(tmp_path)

        scene = DemoScene.of_demo_file(Path("demo_wiping.py"))

        assert scene.demo_file == (tmp_path / "demo_wiping.py").resolve()

    def test_a_missing_demo_is_rejected(self, tmp_path):
        with pytest.raises(MissingDemoFile):
            DemoScene.of_demo_file(tmp_path / "demo_absent.py")

    def test_a_file_name_that_is_no_scene_name_is_rejected(self, tmp_path):
        with pytest.raises(InvalidSceneName):
            DemoScene.of_demo_file(write_demo(tmp_path, "demo_pouring milk.py"))

    def test_a_demo_named_after_a_reserved_scene_is_rejected(self, tmp_path):
        reserved = "%s%s.py" % (demo_scenes.DEMO_FILE_PREFIX, paths.LIVE_SCENE_NAME)

        with pytest.raises(InvalidSceneName):
            DemoScene.of_demo_file(write_demo(tmp_path, reserved))


class TestDemoSceneSet:
    def test_the_scenes_keep_the_order_the_demos_were_given_in(self, tmp_path):
        demos = [
            write_demo(tmp_path, "demo_cutting.py"),
            write_demo(tmp_path, "demo_pouring.py"),
            write_demo(tmp_path, "demo_mixing.py"),
            write_demo(tmp_path, "demo_wiping.py"),
        ]

        scenes = DemoScene.of_demo_files(demos)

        assert [scene.name for scene in scenes] == [
            "cutting",
            "pouring",
            "mixing",
            "wiping",
        ]

    def test_demos_recording_over_each_other_are_rejected(self, tmp_path):
        first = write_demo(tmp_path / "a", "demo_cutting.py")
        second = write_demo(tmp_path / "b", "demo_cutting.py")

        with pytest.raises(DuplicateSceneName):
            DemoScene.of_demo_files([first, second])


# %% running the onboarder
class TestOnboardCommand:
    def test_the_demo_is_recorded_by_the_onboarder_module(self, tmp_path):
        scene = DemoScene.of_demo_file(write_demo(tmp_path, "demo_mixing.py"))

        assert scene.onboard_command(tmp_path / "scenes") == [
            sys.executable,
            "-m",
            demo_scenes.ONBOARDER_MODULE,
            str(tmp_path / "demo_mixing.py"),
            "--name",
            "mixing",
            "--out",
            str(tmp_path / "scenes"),
        ]


class TestRecordScenes:
    def test_every_demo_is_run_in_a_process_of_its_own(self, tmp_path, monkeypatch):
        launcher = ProcessLauncher()
        monkeypatch.setattr(subprocess, "run", launcher)
        scenes = DemoScene.of_demo_files(
            [
                write_demo(tmp_path, "demo_cutting.py"),
                write_demo(tmp_path, "demo_pouring.py"),
            ]
        )
        scenes_directory = tmp_path / "scenes"

        recordings = demo_scenes.record_scenes(scenes, scenes_directory)

        assert launcher.commands == [
            scene.onboard_command(scenes_directory) for scene in scenes
        ]
        assert recordings == [
            SceneRecording(scene=scene, exit_code=demo_scenes.SUCCESSFUL_EXIT_CODE)
            for scene in scenes
        ]

    def test_a_failing_demo_does_not_stop_the_ones_after_it(
        self, tmp_path, monkeypatch
    ):
        failure_code = 3
        launcher = ProcessLauncher(exit_codes=[failure_code])
        monkeypatch.setattr(subprocess, "run", launcher)
        scenes = DemoScene.of_demo_files(
            [
                write_demo(tmp_path, "demo_cutting.py"),
                write_demo(tmp_path, "demo_pouring.py"),
            ]
        )

        recordings = demo_scenes.record_scenes(scenes, tmp_path / "scenes")

        assert len(launcher.commands) == len(scenes)
        assert [recording.exit_code for recording in recordings] == [
            failure_code,
            demo_scenes.SUCCESSFUL_EXIT_CODE,
        ]
        assert [recording.succeeded for recording in recordings] == [False, True]


# %% the command line
class TestMain:
    def test_the_demos_are_recorded_into_the_given_scenes_directory(
        self, tmp_path, monkeypatch
    ):
        launcher = ProcessLauncher()
        monkeypatch.setattr(subprocess, "run", launcher)
        demo_file = write_demo(tmp_path, "demo_wiping.py")
        scenes_directory = tmp_path / "my-scenes"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cramera-onboard-demos",
                str(demo_file),
                "--out",
                str(scenes_directory),
            ],
        )

        exit_code = demo_scenes.main()

        assert exit_code == demo_scenes.SUCCESSFUL_EXIT_CODE
        assert launcher.commands == [
            DemoScene.of_demo_file(demo_file).onboard_command(scenes_directory)
        ]

    def test_the_scenes_directory_defaults_to_the_configured_one(
        self, tmp_path, monkeypatch
    ):
        launcher = ProcessLauncher()
        monkeypatch.setattr(subprocess, "run", launcher)
        monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "configured"))
        demo_file = write_demo(tmp_path, "demo_cutting.py")
        monkeypatch.setattr(sys, "argv", ["cramera-onboard-demos", str(demo_file)])

        demo_scenes.main()

        assert launcher.commands == [
            DemoScene.of_demo_file(demo_file).onboard_command(paths.scenes_directory())
        ]

    def test_a_failed_recording_is_reported_as_the_exit_code(
        self, tmp_path, monkeypatch
    ):
        launcher = ProcessLauncher(exit_codes=[1])
        monkeypatch.setattr(subprocess, "run", launcher)
        demo_file = write_demo(tmp_path, "demo_mixing.py")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cramera-onboard-demos",
                str(demo_file),
                "--out",
                str(tmp_path / "scenes"),
            ],
        )

        assert demo_scenes.main() == demo_scenes.RECORDING_FAILED_EXIT_CODE
