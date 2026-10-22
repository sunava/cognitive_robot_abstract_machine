"""
Tests for the scenes index (``index.json``): what gets listed, and how the shared and
local-only recording roots are merged into one index the viewer reads.
"""

from __future__ import annotations

import json

import pytest

from cramera import paths
from cramera.onboard.scene_index import (
    InvalidSceneName,
    SceneIndexEntry,
    compose_scene_name,
    merged_scene_index,
    validate_scene_name,
    write_scene_index,
)


def _write_bundle(
    directory, robot="pr2", environment="kitchen", task="pouring", version="v1"
):
    """
    A minimal scene bundle on disk under its four-segment identity path.
    """
    bundle = directory / compose_scene_name(robot, environment, task, version)
    bundle.mkdir(parents=True)
    (bundle / "scene.json").write_text("{}")
    return bundle


class TestSceneIndexEntry:
    def test_a_bundle_is_indexed_with_its_full_identity(self, tmp_path):
        """
        The viewer's pickers resolve a (robot, environment, task, version) combination
        back to a bundle, so the index has to carry all four per scene.
        """
        _write_bundle(
            tmp_path, robot="pr2", environment="kitchen", task="pouring", version="v2"
        )

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.to_payload() == {
            "name": "pr2/kitchen/pouring/v2",
            "robot": "pr2",
            "environment": "kitchen",
            "task": "pouring",
            "version": "v2",
        }

    def test_a_bench_only_bundle_has_no_environment(self, tmp_path):
        _write_bundle(tmp_path, robot="tracy", environment=None)

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.environment is None

    def test_a_directory_without_a_scene_file_is_skipped(self, tmp_path):
        (tmp_path / "robot" / "env" / "task" / "v1").mkdir(parents=True)

        assert SceneIndexEntry.of_directory(tmp_path) == []

    def test_the_reserved_live_scene_name_is_skipped(self, tmp_path):
        """
        A live-attach snapshot (:mod:`cramera.live.live_bundle`) is a throwaway bundle
        rebuilt on every attach, never something a user onboarded — it lives one
        directory level down, so it can never match the four levels a real bundle is
        indexed from.
        """
        bundle = tmp_path / paths.LIVE_SCENE_NAME
        bundle.mkdir()
        (bundle / "scene.json").write_text("{}")

        assert SceneIndexEntry.of_directory(tmp_path) == []

    def test_the_reserved_recording_scene_name_is_skipped(self, tmp_path):
        """
        An unsaved live recording (:mod:`cramera.live.recording_bundle`) is likewise a
        throwaway bundle, never something a user onboarded.
        """
        bundle = tmp_path / paths.RECORDING_SCENE_NAME
        bundle.mkdir()
        (bundle / "scene.json").write_text("{}")

        assert SceneIndexEntry.of_directory(tmp_path) == []


class TestWriteSceneIndex:
    def test_a_scene_is_registered_with_its_full_identity(self, tmp_path):
        _write_bundle(tmp_path)
        name = compose_scene_name("pr2", "kitchen", "pouring", "v1")

        write_scene_index(tmp_path / "index.json", name)

        index = json.loads((tmp_path / "index.json").read_text())
        assert index["scenes"] == [
            {
                "name": name,
                "robot": "pr2",
                "environment": "kitchen",
                "task": "pouring",
                "version": "v1",
            }
        ]

    def test_the_first_scene_becomes_the_default(self, tmp_path):
        _write_bundle(tmp_path)
        name = compose_scene_name("pr2", "kitchen", "pouring", "v1")

        write_scene_index(tmp_path / "index.json", name)

        assert json.loads((tmp_path / "index.json").read_text())["default"] == name

    def test_a_later_default_is_left_alone(self, tmp_path):
        first = compose_scene_name("pr2", "kitchen", "pouring", "v1")
        second = compose_scene_name("pr2", "kitchen", "pouring", "v2")
        _write_bundle(tmp_path, version="v1")
        _write_bundle(tmp_path, version="v2")
        write_scene_index(tmp_path / "index.json", first)

        write_scene_index(tmp_path / "index.json", second)

        assert json.loads((tmp_path / "index.json").read_text())["default"] == first

    def test_a_removed_bundle_leaves_no_stale_entry(self, tmp_path):
        first = compose_scene_name("pr2", "kitchen", "pouring", "v1")
        second = compose_scene_name("pr2", "kitchen", "pouring", "v2")
        _write_bundle(tmp_path, version="v1")
        write_scene_index(tmp_path / "index.json", first)
        _write_bundle(tmp_path, version="v2")
        write_scene_index(tmp_path / "index.json", second)

        import shutil

        shutil.rmtree(tmp_path / "pr2" / "kitchen" / "pouring" / "v1")
        write_scene_index(tmp_path / "index.json", second)

        names = [
            e["name"]
            for e in json.loads((tmp_path / "index.json").read_text())["scenes"]
        ]
        assert names == [second]


class TestMergedSceneIndex:
    def test_shared_only(self, monkeypatch, tmp_path):
        shared = tmp_path / "shared"
        name = compose_scene_name("pr2", "kitchen", "pouring", "v1")
        _write_bundle(shared)
        (shared / "index.json").write_text(json.dumps({"default": name, "scenes": []}))
        monkeypatch.setenv("CRAMERA_SCENES", str(shared))
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))

        index = merged_scene_index()

        assert index == {
            "default": name,
            "scenes": [
                {
                    "name": name,
                    "robot": "pr2",
                    "environment": "kitchen",
                    "task": "pouring",
                    "version": "v1",
                }
            ],
        }

    def test_local_only(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        _write_bundle(tmp_path / "scenes", robot="tracy", task="my_run")

        index = merged_scene_index()

        assert index["scenes"] == [
            {
                "name": compose_scene_name("tracy", "kitchen", "my_run", "v1"),
                "robot": "tracy",
                "environment": "kitchen",
                "task": "my_run",
                "version": "v1",
            }
        ]
        assert index["default"] is None

    def test_a_local_recording_shadows_a_shared_scene_of_the_same_name(
        self, monkeypatch, tmp_path
    ):
        """
        A scene's name is entirely composed from its identity, so "the same name" and
        "the same identity" are one and the same now — a local bundle only ever shadows
        a shared one by matching every one of the four segments exactly, and the merged
        index must still list that identity once, not twice.
        """
        name = compose_scene_name("pr2", "kitchen", "pouring", "v1")
        shared = tmp_path / "shared"
        _write_bundle(
            shared, robot="pr2", environment="kitchen", task="pouring", version="v1"
        )
        (shared / "index.json").write_text(json.dumps({"default": name, "scenes": []}))
        monkeypatch.setenv("CRAMERA_SCENES", str(shared))
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        _write_bundle(
            tmp_path / "data" / "scenes",
            robot="pr2",
            environment="kitchen",
            task="pouring",
            version="v1",
        )

        index = merged_scene_index()

        assert [entry["name"] for entry in index["scenes"]] == [name]

    def test_both_roots_absent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "nowhere"))
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))

        assert merged_scene_index() == {"default": None, "scenes": []}


class TestValidateSceneName:
    def test_a_plain_four_segment_name_is_accepted(self):
        name = "pr2/kitchen/my_run-2/v1"
        assert validate_scene_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "just_one_segment",
            "only/two",
            "a/b/c",
            "a/b/c/d/e",
            "/kitchen/task/v1",
            "pr2/kitchen/../v1",
            "pr2/has space/task/v1",
            "pr2/kitchen/task/" + "a" * 65,
        ],
    )
    def test_an_unsafe_name_is_rejected(self, name):
        with pytest.raises(InvalidSceneName):
            validate_scene_name(name)

    @pytest.mark.parametrize(
        "robot", [paths.LIVE_SCENE_NAME, paths.RECORDING_SCENE_NAME]
    )
    def test_a_reserved_robot_segment_is_rejected(self, robot):
        with pytest.raises(InvalidSceneName):
            validate_scene_name(robot + "/kitchen/task/v1")


class TestComposeSceneName:
    def test_joins_all_four_segments(self):
        assert (
            compose_scene_name("pr2", "kitchen", "pouring", "v1")
            == "pr2/kitchen/pouring/v1"
        )

    def test_a_missing_environment_becomes_the_bench_only_sentinel(self):
        from cramera.onboard.scene_index import NO_ENVIRONMENT_SEGMENT

        assert (
            compose_scene_name("tracy", None, "bench_test", "v1")
            == "tracy/%s/bench_test/v1" % NO_ENVIRONMENT_SEGMENT
        )

    def test_an_unsafe_segment_is_rejected(self):
        with pytest.raises(InvalidSceneName):
            compose_scene_name("pr2", "has space", "pouring", "v1")
