"""
Tests of bundling the live world's current model into the throwaway scene.
"""

from __future__ import annotations

import json
import threading
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import List, Optional

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cramera import paths
from cramera.live.bridge import Bridge
from cramera.live.live_bundle import build_live_scene

from .test_robot_parts import ArmPart, NamedBody

# %% fixtures


@dataclass
class RobotWithSubtree:
    """
    A robot mimic exposing what :func:`build_live_scene` reads off it: its root body (a
    real world body, so the robot's subtree can be walked) and one arm (for
    :meth:`~cramera.robot_parts.RobotPartAnnotation.of_robot`).
    """

    root: Body
    arm: ArmPart = field(
        default_factory=lambda: ArmPart(bodies=[NamedBody("robot/arm_link")])
    )

    def get_arms(self):
        return [self.arm]

    def get_left_arm_if_specified(self):
        return None

    def get_right_arm_if_specified(self):
        return None


def shaped(prefix: str, name: str) -> Body:
    """
    A body carrying one visual box shape.

    :param prefix: The body's namespace prefix.
    :param name: The body's local name.
    """
    return Body(
        name=PrefixedName(name, prefix=prefix),
        visual=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
    )


def laboratory_world() -> World:
    """
    A real world with an environment body, a robot subtree and one overlay object.
    """
    world = World()
    root = Body(name=PrefixedName("root", prefix="world"))
    bench = shaped("laboratory", "bench")
    robot_base = shaped("robot", "base_link")
    robot_arm = shaped("robot", "arm_link")
    milk = shaped("world", "milk.stl")
    with world.modify_world():
        world.add_body(root)
        world.add_connection(FixedConnection(parent=root, child=bench))
        world.add_connection(
            FixedConnection(
                parent=root,
                child=robot_base,
                parent_T_connection_expression=(
                    HomogeneousTransformationMatrix.from_xyz_rpy(1.0, 2.0, 0.0)
                ),
            )
        )
        world.add_connection(FixedConnection(parent=robot_base, child=robot_arm))
        world.add_connection(FixedConnection(parent=root, child=milk))
    return world


def carry_overlay_object(bridge: Bridge) -> None:
    """
    Move the overlay object into the robot's subtree, as picking it up does.

    :param bridge: The bridge whose world the object is carried in.
    """
    world = bridge.world
    milk = world.get_body_by_name(PrefixedName("milk.stl", prefix="world"))
    arm = world.get_body_by_name(PrefixedName("arm_link", prefix="robot"))
    with world.modify_world():
        world.move_branch(milk, arm, True)


def attached_bridge(with_robot: bool = False) -> Bridge:
    """
    A bridge attached to :func:`laboratory_world`.

    :param with_robot: Whether the robot mimic is bound, so the robot subtree is bundled
        as its own model.
    """
    bridge = Bridge()
    bridge.attach(laboratory_world())
    if with_robot:
        robot_base = next(
            body for body in bridge.world.bodies if str(body.name) == "robot/base_link"
        )
        bridge.robot = RobotWithSubtree(root=robot_base)
    return bridge


def multi_robot_bridge() -> Bridge:
    """
    A bridge attached to a laboratory holding two robots of the same class.

    The second robot is a base/arm subtree of its own, spawned under its own prefix the
    way a world specification spawns a second robot, and both are discovered the way the
    bridge discovers real sem_dt annotations.
    """
    world = laboratory_world()
    second_base = shaped("uMe", "base_link")
    second_arm = shaped("uMe", "arm_link")
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=second_base,
                parent_T_connection_expression=(
                    HomogeneousTransformationMatrix.from_xyz_rpy(3.0, 0.0, 0.0)
                ),
            )
        )
        world.add_connection(FixedConnection(parent=second_base, child=second_arm))
    first_base = world.get_body_by_name(PrefixedName("base_link", prefix="robot"))
    robots = [
        RobotWithSubtree(root=first_base),
        RobotWithSubtree(
            root=second_base, arm=ArmPart(bodies=[NamedBody("uMe/arm_link")])
        ),
    ]
    # the mimics are no sem_dt annotations, so stand in for the world's own discovery
    world.get_semantic_annotations_by_type = lambda annotation_type: list(robots)
    bridge = Bridge()
    bridge.attach(world)
    return bridge


def use_scratch_scenes_directory(monkeypatch, tmp_path) -> Path:
    """
    Point both scenes roots at throwaway directories.

    :param monkeypatch: The active monkeypatch fixture.
    :param tmp_path: The test's own scratch directory.
    :return: The local root the live bundle is written into.
    """
    monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "shared"))
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
    return paths.local_scenes_directory()


def scene_payload(scenes: Path) -> dict:
    """
    The scene.json the last build wrote.

    :param scenes: The scratch scenes directory.
    """
    return json.loads((scenes / paths.LIVE_SCENE_NAME / "scene.json").read_text())


# %% nothing attached


class TestNothingAttachedYet:
    def test_a_fresh_bridge_bundles_nothing(self, monkeypatch, tmp_path):
        use_scratch_scenes_directory(monkeypatch, tmp_path)

        assert build_live_scene(Bridge()) is None


# %% bundling the attached world


class TestBuildLiveScene:
    def test_the_world_is_bundled_under_the_reserved_name(self, monkeypatch, tmp_path):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
        bridge = attached_bridge()

        assert build_live_scene(bridge) == paths.LIVE_SCENE_NAME

        scene = scene_payload(scenes)
        assert scene["name"] == paths.LIVE_SCENE_NAME
        assert scene["worldBound"] is True
        assert scene["bundleSignature"] == bridge.bundle_signature()
        assert [model["name"] for model in scene["models"]] == ["environment"]
        assert (scenes / paths.LIVE_SCENE_NAME / "environment.urdf").is_file()

    def test_overlay_objects_stay_out_of_the_environment_model(
        self, monkeypatch, tmp_path
    ):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
        build_live_scene(attached_bridge())

        urdf = (scenes / paths.LIVE_SCENE_NAME / "environment.urdf").read_text()

        assert "laboratory/bench" in urdf
        assert "milk.stl" not in urdf

    def test_a_carried_overlay_object_stays_out_of_the_robot_model(
        self, monkeypatch, tmp_path
    ):
        """
        The overlay streams a carried object's pose, so baking it into the robot model
        as well would draw it twice -- and the model is not rebuilt when the object is
        put down again, which would leave the copy stuck to the hand.
        """
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
        bridge = attached_bridge(with_robot=True)
        carry_overlay_object(bridge)

        build_live_scene(bridge)

        urdf = (scenes / paths.LIVE_SCENE_NAME / "robotwithsubtree.urdf").read_text()
        assert "robot/arm_link" in urdf
        assert "milk.stl" not in urdf

    def test_the_robot_subtree_becomes_its_own_model(self, monkeypatch, tmp_path):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
        bridge = attached_bridge(with_robot=True)

        build_live_scene(bridge)

        scene = scene_payload(scenes)
        robot_models = [model for model in scene["models"] if model["robot"]]
        assert [model["name"] for model in robot_models] == ["robotwithsubtree"]
        robot_urdf = (
            scenes / paths.LIVE_SCENE_NAME / "robotwithsubtree.urdf"
        ).read_text()
        assert "robot/base_link" in robot_urdf
        assert "robot/arm_link" in robot_urdf
        environment_urdf = (
            scenes / paths.LIVE_SCENE_NAME / "environment.urdf"
        ).read_text()
        assert "robot/base_link" not in environment_urdf

    def test_the_robot_root_is_grafted_at_the_origin(self, monkeypatch, tmp_path):
        """
        The viewer applies the robot's live base pose on top of the model, so the model
        itself must not bake the spawn pose in.
        """
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
        build_live_scene(attached_bridge(with_robot=True))

        robot_urdf = (
            scenes / paths.LIVE_SCENE_NAME / "robotwithsubtree.urdf"
        ).read_text()

        document = ElementTree.fromstring(robot_urdf)
        graft = next(
            joint
            for joint in document.iter("joint")
            if joint.find("child").attrib["link"] == "robot/base_link"
        )
        assert graft.find("origin").attrib["xyz"] == "0.0 0.0 0.0"

    def test_the_scene_names_the_robot(self, monkeypatch, tmp_path):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
        build_live_scene(attached_bridge(with_robot=True))

        robot = scene_payload(scenes)["robot"]

        assert robot["name"] == "robotwithsubtree"
        assert robot["baseBody"] == "base_link"

    def test_an_unchanged_world_does_not_rebuild_the_bundle(
        self, monkeypatch, tmp_path
    ):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
        bridge = attached_bridge()
        build_live_scene(bridge)
        marker = scenes / paths.LIVE_SCENE_NAME / "marker"
        marker.touch()

        build_live_scene(bridge)

        assert marker.exists()

    def test_a_model_change_rebuilds_the_bundle(self, monkeypatch, tmp_path):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
        bridge = attached_bridge()
        build_live_scene(bridge)
        marker = scenes / paths.LIVE_SCENE_NAME / "marker"
        marker.touch()

        world = bridge.world
        with world.modify_world():
            world.add_connection(
                FixedConnection(parent=world.root, child=shaped("laboratory", "shelf"))
            )
        bridge.observe_model_change()
        build_live_scene(bridge)

        assert not marker.exists()


# %% concurrent builds


class TestConcurrentBuilds:
    """
    The bridge serves on a threading HTTP server, and the viewer keeps polling
    ``/live_scene`` while a demo starts up, so two builds of the same throwaway scene
    overlap routinely.

    Each one clears the output directory before writing it -- the directory the other
    one is deleting and writing at the same time.
    """

    VIEWERS_ASKING_AT_ONCE = 8
    """
    How many builds are started together.

    More than two, so the window between one build's delete and its write is actually
    hit rather than only theoretically open.
    """

    def test_overlapping_builds_all_produce_a_complete_bundle(
        self, monkeypatch, tmp_path
    ):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
        bridge = attached_bridge()

        names: List[Optional[str]] = []
        failures: List[BaseException] = []
        start = threading.Barrier(self.VIEWERS_ASKING_AT_ONCE)

        def build() -> None:
            start.wait()
            try:
                names.append(build_live_scene(bridge))
            except BaseException as failure:
                failures.append(failure)

        builders = [
            threading.Thread(target=build, name="viewer-%d" % index)
            for index in range(self.VIEWERS_ASKING_AT_ONCE)
        ]
        for builder in builders:
            builder.start()
        for builder in builders:
            builder.join(timeout=60)

        assert [type(failure).__name__ for failure in failures] == []
        assert names == [paths.LIVE_SCENE_NAME] * self.VIEWERS_ASKING_AT_ONCE
        scene = scene_payload(scenes)
        assert [model["name"] for model in scene["models"]] == ["environment"]
        assert (scenes / paths.LIVE_SCENE_NAME / "environment.urdf").exists()


# %% keeping the throwaway bundle out of the shared checkout


class TestLiveBundleLocation:
    def test_the_bundle_is_written_into_the_local_root(self, monkeypatch, tmp_path):
        """
        The shared root is a git checkout, and a scene there is shadowed by a local one
        of the same name, so the throwaway bundle belongs in the local root only.
        """
        local = use_scratch_scenes_directory(monkeypatch, tmp_path)

        build_live_scene(attached_bridge())

        assert (local / paths.LIVE_SCENE_NAME / "scene.json").is_file()
        assert not (paths.scenes_directory() / paths.LIVE_SCENE_NAME).exists()


# %% a world holding several robots


class TestSeveralRobots:
    """
    Every robot of the world is bundled as its own model, because the viewer drives a
    model by its streamed base pose -- a robot baked into the environment stands still
    while its joints move.
    """

    def test_every_robot_becomes_its_own_model(self, monkeypatch, tmp_path):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)

        build_live_scene(multi_robot_bridge())

        models = scene_payload(scenes)["models"]
        assert [model["name"] for model in models] == [
            "environment",
            "robotwithsubtree",
            "robotwithsubtree_2",
        ]
        assert [model["prefix"] for model in models] == ["", "robot", "uMe"]
        assert [model["robot"] for model in models] == [False, True, True]
        assert (scenes / paths.LIVE_SCENE_NAME / "robotwithsubtree.urdf").is_file()
        assert (scenes / paths.LIVE_SCENE_NAME / "robotwithsubtree_2.urdf").is_file()

    def test_no_robots_bodies_are_baked_into_the_environment(
        self, monkeypatch, tmp_path
    ):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)

        build_live_scene(multi_robot_bridge())

        environment = (scenes / paths.LIVE_SCENE_NAME / "environment.urdf").read_text()
        assert "laboratory/bench" in environment
        assert "robot/base_link" not in environment
        assert "uMe/base_link" not in environment
        second = (
            scenes / paths.LIVE_SCENE_NAME / "robotwithsubtree_2.urdf"
        ).read_text()
        assert "uMe/arm_link" in second
        assert "robot/arm_link" not in second

    def test_every_robot_model_is_grafted_at_the_origin(self, monkeypatch, tmp_path):
        """
        The second robot's spawn pose reaches the viewer as its streamed base pose, so
        its model must not bake it in any more than the first robot's does.
        """
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)

        build_live_scene(multi_robot_bridge())

        document = ElementTree.fromstring(
            (scenes / paths.LIVE_SCENE_NAME / "robotwithsubtree_2.urdf").read_text()
        )
        graft = next(
            joint
            for joint in document.iter("joint")
            if joint.find("child").attrib["link"] == "uMe/base_link"
        )
        assert graft.find("origin").attrib["xyz"] == "0.0 0.0 0.0"

    def test_the_scene_names_every_robot(self, monkeypatch, tmp_path):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)

        build_live_scene(multi_robot_bridge())

        robots = scene_payload(scenes)["robots"]
        assert [robot["name"] for robot in robots] == [
            "robotwithsubtree",
            "robotwithsubtree_2",
        ]
        assert [robot["prefix"] for robot in robots] == ["robot", "uMe"]
        assert [robot["class"] for robot in robots] == [
            "RobotWithSubtree",
            "RobotWithSubtree",
        ]
        assert [robot["baseBody"] for robot in robots] == ["base_link", "base_link"]

    def test_each_robot_payload_matches_its_model_entry(self, monkeypatch, tmp_path):
        """
        The viewer pairs a robot payload with the model it drives by prefix, falling
        back to the name, and builds that robot's part table from the payload's own
        ``parts`` -- so the two have to agree, entry for entry.
        """
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)

        build_live_scene(multi_robot_bridge())

        scene = scene_payload(scenes)
        robot_models = [model for model in scene["models"] if model["robot"]]
        assert [(model["name"], model["prefix"]) for model in robot_models] == [
            (robot["name"], robot["prefix"]) for robot in scene["robots"]
        ]
        assert [robot["parts"] for robot in scene["robots"]] == [
            {"ArmPart": ["arm_link"]},
            {"ArmPart": ["arm_link"]},
        ]

    def test_each_robot_carries_its_own_part_annotations(self, monkeypatch, tmp_path):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)

        build_live_scene(multi_robot_bridge())

        robots = scene_payload(scenes)["robots"]
        assert [
            [annotation["links"] for annotation in robot["partAnnotations"]]
            for robot in robots
        ] == [[["arm_link"]], [["arm_link"]]]
        assert [
            [annotation["robot"] for annotation in robot["partAnnotations"]]
            for robot in robots
        ] == [["robot"], ["uMe"]]

    def test_the_single_robot_field_stays_the_first_robot(self, monkeypatch, tmp_path):
        """
        A viewer that only ever knew one robot keeps reading ``robot``; ``robots`` is
        what the multi-robot viewer reads.
        """
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)

        build_live_scene(multi_robot_bridge())

        scene = scene_payload(scenes)
        assert scene["robot"] == scene["robots"][0]
        assert scene["robot"]["prefix"] == "robot"

    def test_one_robot_is_still_bundled_as_one_model(self, monkeypatch, tmp_path):
        """
        The single-robot bundle is what every existing scene and viewer reads; a world
        with one robot must come out exactly as it did before.
        """
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)

        build_live_scene(attached_bridge(with_robot=True))

        scene = scene_payload(scenes)
        assert [model["name"] for model in scene["models"]] == [
            "environment",
            "robotwithsubtree",
        ]
        assert [robot["prefix"] for robot in scene["robots"]] == ["robot"]
