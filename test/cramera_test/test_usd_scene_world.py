"""
A scene bundle converted from a USD stage loads into CRAM worlds whose bodies carry the
stage's semantic attributes, queryable with EQL through the live bridge.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ElementTree
from copy import deepcopy
from pathlib import Path

import numpy
import pytest
import trimesh

from cramera.examples.usd_scene_world import ExampleArgument, LoadedUsdScene, main
from cramera.knowledge.queryable_knowledge import QueryScope
from cramera.live.bridge import Bridge
from cramera.live.world_query import WorldQuerySource
from cramera.onboard.usd_scene.bundle import BundleKey
from cramera.onboard.usd_scene.bundle_layout import BundleFile
from cramera.onboard.usd_scene.exceptions import (
    SceneBundleNotFound,
    SceneNotFound,
    UnknownPrimLink,
)
from cramera.onboard.usd_scene.scene_description import (
    OntologyAttributeName,
    SemanticAttributeName,
    SemanticNamespace,
)
from cramera.scene_presentation import PresentationField, ScenePresentation
from cramera.usd_scene_world import (
    PrimAttribute,
    SemanticClassMapping,
    UsdPrimAnnotation,
    UsdSceneDescription,
    UsdSceneEnvironment,
    UsdSceneQueryName,
    UsdSceneQuerySource,
    UsdSceneQuestion,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Fridge,
    Shelf,
    WallDecor,
)
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box
from semantic_digital_twin.world_description.world_entity import Body

DATASET = Path(__file__).parent / "dataset" / "usd_scene"
"""
Small stages and bundles, one per behaviour.
"""

ANNOTATED_BUNDLE = DATASET / "annotated_bundle"
"""
A bundle in the converter's format with a replay robot and a free object, its meshes
left out.
"""


# %% bundle fixture
def write_box(path: Path) -> None:
    """
    Write a unit box as a GLB mesh.

    :param path: The mesh file to create.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box().export(path)


def read_json(bundle: Path, file: BundleFile) -> dict:
    """
    One JSON file of a bundle.

    :param bundle: The bundle directory.
    :param file: The file.
    """
    return json.loads((bundle / file).read_text(encoding="utf-8"))


@pytest.fixture
def annotated_bundle(tmp_path: Path) -> Path:
    """
    The annotated bundle with a box GLB for every mesh its descriptions and objects name.
    """
    bundle = Path(shutil.copytree(ANNOTATED_BUNDLE, tmp_path / ANNOTATED_BUNDLE.name))
    for description in bundle.glob("*.urdf"):
        for mesh in ElementTree.parse(description).iter("mesh"):
            write_box(bundle / mesh.get("filename"))
    for item in read_json(bundle, BundleFile.SCENE)[PresentationField.OBJECTS]:
        write_box(bundle / item[BundleKey.MESH])
    return bundle


@pytest.fixture
def prims(annotated_bundle: Path) -> dict:
    """
    The prims ``semantics.json`` of the annotated bundle describes, by path.
    """
    return read_json(annotated_bundle, BundleFile.SEMANTICS)[BundleKey.PRIMS]


@pytest.fixture
def environment(annotated_bundle: Path) -> UsdSceneEnvironment:
    """
    The environment of the annotated bundle.
    """
    return UsdSceneEnvironment(bundle_directory=annotated_bundle)


@pytest.fixture
def world(environment: UsdSceneEnvironment) -> World:
    """
    A world holding only the annotated bundle.
    """
    return environment.create_world()


def labels_by_path(world: World) -> dict[str, UsdPrimAnnotation]:
    """
    The prim labels of a world, by prim path.

    :param world: The world.
    """
    return {
        label.prim_path: label
        for label in world.get_semantic_annotations_by_type(UsdPrimAnnotation)
    }


def annotations_on(world: World, body: Body) -> list[HasRootBody]:
    """
    The annotations of a world rooted at one body.

    :param world: The world.
    :param body: The body.
    """
    return [
        annotation
        for annotation in world.get_semantic_annotations_by_type(HasRootBody)
        if annotation.root is body
    ]


def environment_link_names(bundle: Path) -> set[str]:
    """
    The link names of a bundle's environment description.

    :param bundle: The bundle directory.
    """
    description = ElementTree.parse(bundle / BundleFile.ENVIRONMENT)
    return {link.get("name") for link in description.iter("link")}


# %% bundle description
def test_description_reads_models_objects_and_prims(annotated_bundle: Path) -> None:
    """
    The description names every model and object of the scene and every annotated prim.
    """
    scene = read_json(annotated_bundle, BundleFile.SCENE)
    description = UsdSceneDescription.from_bundle(annotated_bundle)
    assert [model.description for model in description.models] == [
        Path(model[BundleKey.URDF]) for model in scene[PresentationField.MODELS]
    ]
    assert [model.robot for model in description.models] == [
        model[PresentationField.ROBOT] for model in scene[PresentationField.MODELS]
    ]
    assert [item.key for item in description.objects] == [
        item[PresentationField.KEY] for item in scene[PresentationField.OBJECTS]
    ]
    assert [prim.path for prim in description.prims] == list(
        read_json(annotated_bundle, BundleFile.SEMANTICS)[BundleKey.PRIMS]
    )


def test_description_takes_initial_joint_values_from_the_first_frame(
    annotated_bundle: Path,
) -> None:
    """
    The first trajectory frame holds the joint values the stage was authored with.
    """
    [first_frame] = read_json(annotated_bundle, BundleFile.TRAJECTORY)[BundleKey.FRAMES]
    description = UsdSceneDescription.from_bundle(annotated_bundle)
    assert description.initial_joint_values == first_frame


def test_explicit_missing_bundle_reports_its_directory(tmp_path: Path) -> None:
    """
    A directory without a scene file is reported instead of loading something else.
    """
    with pytest.raises(SceneBundleNotFound) as error:
        UsdSceneEnvironment(bundle_directory=tmp_path)
    assert error.value.directory == tmp_path


def test_scene_name_resolves_through_the_scene_roots(
    annotated_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A bundle in the shared scenes directory is found by its name.
    """
    monkeypatch.setenv("CRAMERA_SCENES", str(annotated_bundle.parent))
    environment = UsdSceneEnvironment.from_scene_name(annotated_bundle.name)
    assert environment.bundle_directory == annotated_bundle


def test_unknown_scene_name_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A scene name no scene root holds raises instead of returning an empty world.
    """
    monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path))
    with pytest.raises(SceneNotFound) as error:
        UsdSceneEnvironment.from_scene_name("no_such_scene")
    assert error.value.name == "no_such_scene"


# %% world construction
def test_every_environment_link_and_object_is_a_body(
    annotated_bundle: Path, world: World
) -> None:
    """
    The world holds its root, every environment link and every free object.
    """
    objects = {
        item[PresentationField.KEY]
        for item in read_json(annotated_bundle, BundleFile.SCENE)[
            PresentationField.OBJECTS
        ]
    }
    assert {body.name.name for body in world.bodies} == (
        environment_link_names(annotated_bundle) | objects | {world.root.name.name}
    )


def test_movable_joints_become_the_one_degree_of_freedom_connections(
    annotated_bundle: Path, world: World
) -> None:
    """
    Exactly the environment's movable joints can be actuated one value at a time.
    """
    [environment_model, _] = read_json(annotated_bundle, BundleFile.SCENE)[
        PresentationField.MODELS
    ]
    assert {
        str(connection.name)
        for connection in world.connections
        if isinstance(connection, ActiveConnection1DOF)
    } == set(environment_model[BundleKey.MOVABLE_JOINTS])


def test_replay_robot_is_left_to_the_caller(
    annotated_bundle: Path, world: World
) -> None:
    """
    The recorded robot's links stay out; a planning robot comes from the caller.
    """
    arm = ElementTree.parse(annotated_bundle / "arm.urdf")
    arm_links = {link.get("name") for link in arm.iter("link")}
    assert arm_links.isdisjoint(body.name.name for body in world.bodies)


def test_environment_hangs_rigidly_from_the_world_root(
    annotated_bundle: Path, world: World
) -> None:
    """
    The converted root link sits at the world origin.
    """
    description = ElementTree.parse(annotated_bundle / BundleFile.ENVIRONMENT)
    root_link = description.getroot().find("link").get("name")
    environment_root = world.get_body_by_name(root_link)
    assert isinstance(environment_root.parent_connection, FixedConnection)
    assert environment_root.parent_connection.parent is world.root
    numpy.testing.assert_allclose(environment_root.global_pose.to_np(), numpy.eye(4))


def test_joints_start_at_the_authored_values(
    annotated_bundle: Path, world: World
) -> None:
    """
    Every environment joint starts where the first trajectory frame puts it.
    """
    [first_frame] = read_json(annotated_bundle, BundleFile.TRAJECTORY)[BundleKey.FRAMES]
    positions = {
        str(connection.name): connection.position
        for connection in world.connections
        if isinstance(connection, ActiveConnection1DOF)
    }
    assert positions == pytest.approx({name: first_frame[name] for name in positions})


def test_joint_missing_from_the_first_frame_stays_at_zero(
    annotated_bundle: Path,
) -> None:
    """
    A joint the trajectory does not mention keeps the position its description gives.
    """
    trajectory = read_json(annotated_bundle, BundleFile.TRAJECTORY)
    del trajectory[BundleKey.FRAMES][0]["Storage__Fridge__DoorHinge"]
    (annotated_bundle / BundleFile.TRAJECTORY).write_text(json.dumps(trajectory))
    world = UsdSceneEnvironment(bundle_directory=annotated_bundle).create_world()
    assert world.get_connection_by_name("Storage__Fridge__DoorHinge").position == 0.0


def test_objects_are_free_bodies_at_their_spawn_pose(
    annotated_bundle: Path, world: World
) -> None:
    """
    A free object can be picked up: it hangs from the root by six degrees of freedom.
    """
    [item] = read_json(annotated_bundle, BundleFile.SCENE)[PresentationField.OBJECTS]
    body = world.get_body_by_name(item[PresentationField.KEY])
    assert isinstance(body.parent_connection, Connection6DoF)
    assert body.parent_connection.parent is world.root
    numpy.testing.assert_allclose(
        body.global_pose.to_np(),
        Pose.from_xyz_quaternion(*item[BundleKey.SPAWN]).to_np(),
        atol=1e-6,
    )
    assert [shape.filename for shape in body.visual.shapes] == [
        str(annotated_bundle / item[BundleKey.MESH])
    ]


def test_collision_shapes_come_from_the_description(world: World) -> None:
    """
    Links declaring collision geometry keep it; the others collide with nothing.
    """
    colliding = {
        body.name.name: [type(shape) for shape in body.collision.shapes]
        for body in world.bodies
        if body.collision.shapes
    }
    assert colliding == {"Storage__Fridge": [Box]}


def test_visuals_are_the_bundled_meshes(annotated_bundle: Path, world: World) -> None:
    """
    Each link shows the GLB the bundle stores for it.
    """
    fridge = world.get_body_by_name("Storage__Fridge")
    assert [shape.filename for shape in fridge.visual.shapes] == [
        str(annotated_bundle / BundleFile.MESHES / "Storage__Fridge.glb")
    ]


def test_populate_keeps_an_existing_robot(
    environment: UsdSceneEnvironment, pr2_world_copy: World
) -> None:
    """
    Loading the scene next to a robot keeps the robot instance and its pose.
    """
    [robot] = pr2_world_copy.get_semantic_annotations_by_type(PR2)
    initial_pose = robot.root.global_pose.to_np().copy()
    environment.populate(pr2_world_copy)
    assert pr2_world_copy.get_semantic_annotations_by_type(PR2) == [robot]
    numpy.testing.assert_allclose(robot.root.global_pose.to_np(), initial_pose)
    assert labels_by_path(pr2_world_copy).keys() == {
        prim.path for prim in environment.description.prims
    }


def test_repeated_creation_gives_independent_worlds(
    environment: UsdSceneEnvironment,
) -> None:
    """
    Opening a door in one world leaves a separately created world untouched.
    """
    first, second = environment.create_world(), environment.create_world()
    first_hinge = first.get_connection_by_name("Storage__Fridge__DoorHinge")
    second_hinge = second.get_connection_by_name("Storage__Fridge__DoorHinge")
    initial_position = second_hinge.position
    first_hinge.position = 1.0
    assert second_hinge.position == initial_position


# %% semantic annotations
def test_every_annotated_prim_labels_the_body_of_its_link(
    prims: dict, world: World
) -> None:
    """
    Each prim of ``semantics.json`` becomes one label on its link's body.
    """
    assert {
        path: label.root.name.name for path, label in labels_by_path(world).items()
    } == {path: entry[BundleKey.LINK] for path, entry in prims.items()}


def test_prims_sharing_a_body_keep_separate_labels(prims: dict, world: World) -> None:
    """
    Two prims merged into one link are two labels that do not count as the same.
    """
    root_link_paths = sorted(
        path for path, entry in prims.items() if entry[BundleKey.LINK] == "World"
    )
    labels = annotations_on(world, world.get_body_by_name("World"))
    assert sorted(label.prim_path for label in labels) == root_link_paths
    [first, second] = labels
    assert first != second


def test_label_carries_the_identity_attributes(prims: dict, world: World) -> None:
    """
    Class, name, component and capabilities are fields an EQL query can compare.
    """
    station_path = "/World/Stations/L09_PlateHotel"
    authored = prims[station_path][SemanticNamespace.SEMANTIC_DIGITAL_TWIN]
    label = labels_by_path(world)[station_path]
    assert label.semantic_class is None
    assert label.canonical_name == authored[SemanticAttributeName.CANONICAL_NAME]
    assert label.component_id == authored[SemanticAttributeName.COMPONENT_ID]
    assert label.capabilities == authored[SemanticAttributeName.CAPABILITIES]


def test_label_carries_the_ontology_types(prims: dict, world: World) -> None:
    """
    Asserted and inferred ontology types are kept apart.
    """
    fridge_path = "/World/Storage/Fridge"
    authored = prims[fridge_path][SemanticNamespace.ONTOLOGY]
    label = labels_by_path(world)[fridge_path]
    assert label.semantic_class == (
        prims[fridge_path][SemanticNamespace.SEMANTIC_DIGITAL_TWIN][
            SemanticAttributeName.CLASS
        ]
    )
    assert label.ontology_types == authored[OntologyAttributeName.TYPES]
    assert label.inferred_ontology_types == (
        authored[OntologyAttributeName.INFERRED_TYPES]
    )


def test_label_keeps_every_authored_attribute(prims: dict, world: World) -> None:
    """
    Attributes without a field of their own stay available, with their JSON value.
    """
    tag_path = "/World/Stations/L09_PlateHotel/Gate/Tag"
    authored = prims[tag_path][SemanticNamespace.SEMANTIC_DIGITAL_TWIN]
    assert labels_by_path(world)[tag_path].attributes == [
        PrimAttribute(name=name, value=value) for name, value in authored.items()
    ]


def test_label_is_named_after_its_prim(world: World) -> None:
    """
    A label is found by its prim path.
    """
    label = world.get_semantic_annotation_by_name("/World/Storage/Shelf")
    assert isinstance(label, UsdPrimAnnotation)
    assert label.prim_path == "/World/Storage/Shelf"


def test_mapped_classes_add_the_matching_annotation(prims: dict, world: World) -> None:
    """
    A prim whose class has a counterpart is annotated with it on the same body.
    """
    mapping = SemanticClassMapping()
    expected = {
        (
            mapping.annotation_types[
                entry[SemanticNamespace.SEMANTIC_DIGITAL_TWIN][
                    SemanticAttributeName.CLASS
                ]
            ],
            entry[BundleKey.LINK],
        )
        for entry in prims.values()
        if entry[SemanticNamespace.SEMANTIC_DIGITAL_TWIN].get(
            SemanticAttributeName.CLASS
        )
        in mapping.annotation_types
    }
    found = {
        (type(annotation), annotation.root.name.name)
        for annotation in world.semantic_annotations
        if not isinstance(annotation, UsdPrimAnnotation)
    }
    assert found == expected
    assert {link for _, link in expected} == {"Storage__Fridge", "Storage__Shelf"}


def test_mapped_annotation_is_named_after_its_body(world: World) -> None:
    """
    The added annotation shares its body's name, the way annotation factories name it.
    """
    [fridge] = world.get_semantic_annotations_by_type(Fridge)
    assert fridge.name == fridge.root.name


def test_unmapped_class_keeps_only_the_label(prims: dict, world: World) -> None:
    """
    A class without a counterpart is still queryable through its label.
    """
    authored = prims["/World/Storage/Whiteboard"][
        SemanticNamespace.SEMANTIC_DIGITAL_TWIN
    ]
    assert authored[SemanticAttributeName.CLASS] not in (
        SemanticClassMapping().annotation_types
    )
    whiteboard = world.get_body_by_name("Storage__Whiteboard")
    assert [type(annotation) for annotation in annotations_on(world, whiteboard)] == [
        UsdPrimAnnotation
    ]


def test_caller_mapping_adds_classes(annotated_bundle: Path) -> None:
    """
    A caller maps further classes without changing the module.
    """
    mapping = SemanticClassMapping(annotation_types={"whiteboard": WallDecor})
    world = UsdSceneEnvironment(
        bundle_directory=annotated_bundle, class_mapping=mapping
    ).create_world()
    [decor] = world.get_semantic_annotations_by_type(WallDecor)
    assert decor.root is world.get_body_by_name("Storage__Whiteboard")
    assert world.get_semantic_annotations_by_type(Fridge) == []


def test_prim_on_a_missing_link_is_rejected(annotated_bundle: Path) -> None:
    """
    A semantics entry naming a link the description lacks is an inconsistent bundle.
    """
    semantics = read_json(annotated_bundle, BundleFile.SEMANTICS)
    semantics[BundleKey.PRIMS]["/World/Storage/Shelf"][BundleKey.LINK] = "Missing"
    (annotated_bundle / BundleFile.SEMANTICS).write_text(json.dumps(semantics))
    with pytest.raises(UnknownPrimLink) as error:
        UsdSceneEnvironment(bundle_directory=annotated_bundle).create_world()
    assert (error.value.prim_path, error.value.link) == (
        "/World/Storage/Shelf",
        "Missing",
    )


def test_labels_survive_a_world_copy(world: World) -> None:
    """
    Planning in a copied world keeps every label with all of its attributes.
    """
    copied = labels_by_path(deepcopy(world))
    original = labels_by_path(world)
    assert copied.keys() == original.keys()
    for path, label in original.items():
        assert copied[path].root.name == label.root.name
        assert copied[path].attributes == label.attributes
        assert copied[path].capabilities == label.capabilities
        assert copied[path].ontology_types == label.ontology_types


# %% live queries
@pytest.fixture
def bridge(environment: UsdSceneEnvironment, world: World) -> Bridge:
    """
    A bridge answering questions about the annotated world.
    """
    live_bridge = Bridge()
    live_bridge.register_query_source(environment.query_source(world))
    return live_bridge


def test_query_source_adds_prims_to_the_world_vocabulary(
    bridge: Bridge, world: World
) -> None:
    """
    Every standard world variable stays available next to the prim labels.
    """
    world_bridge = Bridge()
    world_bridge.register_query_source(WorldQuerySource(world))
    assert bridge.query_variables() == world_bridge.query_variables() + [
        UsdSceneQueryName.PRIM
    ]


def test_query_source_is_titled_after_the_scene(
    annotated_bundle: Path, bridge: Bridge
) -> None:
    """
    The query panel names the converted environment.
    """
    scene = read_json(annotated_bundle, BundleFile.SCENE)
    assert bridge.query_title() == scene["environmentName"]


def test_eql_finds_the_bodies_offering_a_capability(bridge: Bridge) -> None:
    """
    A capability names the station bodies offering it.
    """
    result = bridge.run_query(
        "an(entity(prim.root).where(contains(prim.capabilities, 'store_plate')))"
    )
    assert result.ok
    assert [row["__entity__"] for row in result.rows] == ["Stations__L09_PlateHotel"]


def test_eql_finds_prims_by_semantic_class(bridge: Bridge) -> None:
    """
    The authored class selects a label and its body.
    """
    result = bridge.run_query(
        "an(entity(prim).where(prim.semantic_class == 'lab_refrigerator'))"
    )
    assert result.ok
    assert [(row["__entity__"], row["body_name"]) for row in result.rows] == [
        ("/World/Storage/Fridge", "Storage__Fridge")
    ]


def test_eql_finds_prims_by_any_authored_attribute(bridge: Bridge) -> None:
    """
    An attribute without a field of its own is matched by name and value.
    """
    result = bridge.run_query(
        "an(entity(prim).where(contains(prim.attributes, "
        "PrimAttribute(name='zone', value='StorageRoom'))))"
    )
    assert result.ok
    assert [row["__entity__"] for row in result.rows] == ["/World/Storage"]


def test_prim_presets_run(bridge: Bridge, world: World) -> None:
    """
    The questions about prims follow the world's own and all answer.
    """
    world_questions = {preset.text for preset in WorldQuerySource(world).presets()}
    prim_presets = [
        preset
        for preset in bridge.query_presets()
        if preset.text not in world_questions
    ]
    assert [preset.text for preset in prim_presets] == list(UsdSceneQuestion)
    for preset in prim_presets:
        assert bridge.run_query(preset.code, preset.scope).ok, preset.text


def test_capability_preset_lists_the_station(bridge: Bridge) -> None:
    """
    Only prims with authored capabilities answer the capability question.
    """
    [preset] = [
        preset
        for preset in bridge.query_presets()
        if preset.text == UsdSceneQuestion.CAPABILITIES
    ]
    result = bridge.run_query(preset.code, preset.scope)
    assert [row["__entity__"] for row in result.rows] == [
        "/World/Stations/L09_PlateHotel"
    ]


def test_query_source_reads_the_world_afresh(
    environment: UsdSceneEnvironment, world: World
) -> None:
    """
    A label added after the source was made is part of the next answer.
    """
    source = environment.query_source(world)
    shelf = world.get_body_by_name("Storage__Shelf")
    with world.modify_world():
        world.add_semantic_annotation(
            UsdPrimAnnotation(root=shelf, prim_path="/World/Storage/Shelf/Board")
        )
    [knowledge] = [
        item for item in source.knowledge() if item.scope is QueryScope.CURRENT_STATE
    ]
    [prim_domain] = [
        domain for domain in knowledge.domains if domain.name == UsdSceneQueryName.PRIM
    ]
    assert "/World/Storage/Shelf/Board" in [
        label.prim_path for label in prim_domain.objects
    ]


def test_visualization_carries_presentation_and_queries(
    annotated_bundle: Path, environment: UsdSceneEnvironment, world: World
) -> None:
    """
    The live viewer shows the scene as authored and answers questions about it.
    """
    visualization = environment.create_visualization(world, port=0)
    assert visualization.world is world
    assert visualization.port == 0
    assert visualization.bridge.presentation == ScenePresentation.from_bundle(
        annotated_bundle
    )
    assert isinstance(visualization.bridge.query_source, UsdSceneQuerySource)
    assert visualization.bridge.query_source.world is world


# %% headless example
def test_example_counts_joints_and_labels(annotated_bundle: Path, prims: dict) -> None:
    """
    The example reports the movable joints and prim labels the world was given.
    """
    [environment_model, _] = read_json(annotated_bundle, BundleFile.SCENE)[
        PresentationField.MODELS
    ]
    scene = LoadedUsdScene.load(UsdSceneEnvironment(bundle_directory=annotated_bundle))
    assert scene.movable_joint_count == len(environment_model[BundleKey.MOVABLE_JOINTS])
    assert scene.label_count == len(prims)


def test_example_prints_the_summary_of_a_named_scene(
    annotated_bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Run by scene name, the example prints what the world holds and stops.
    """
    monkeypatch.setenv("CRAMERA_SCENES", str(annotated_bundle.parent))
    main([ExampleArgument.SCENE, annotated_bundle.name])
    printed = capsys.readouterr().out.splitlines()
    expected = LoadedUsdScene.load(
        UsdSceneEnvironment(bundle_directory=annotated_bundle)
    ).summary()
    timing_line = -1
    assert printed[:timing_line] == expected.splitlines()[:timing_line]
