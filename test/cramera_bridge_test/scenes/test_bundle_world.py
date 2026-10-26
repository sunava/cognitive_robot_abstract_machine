"""
CRAM worlds built from scene bundles.
"""

from __future__ import annotations

import shutil

import pytest

from cramera.scenes import SceneCatalog, SceneDirectory, SceneSource
from cramera_bridge.documents import DocumentFile
from cramera_bridge.exceptions import BundleNotFound, UnknownPrimLink
from cramera_bridge.export.world_models import ObjectFiles, WorldModels
from cramera_bridge.geometry import NumericPose
from cramera_bridge.scenes.bundle_world import (
    BundleContents,
    BundleEnvironment,
    ColorText,
    WorldSummary,
)
from cramera_bridge.scenes.semantics import (
    AuthoredClass,
    SemanticAttributeName,
    UsdPrimAnnotation,
)
from cramera_bridge.world_objects import WorldObjects
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.bundle.scene import SceneDescription
from cramera_protocol.bundle.trajectory import Trajectory
from semantic_digital_twin.semantic_annotations.semantic_annotations import Fridge
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Mesh

from ..dataset.bundles import SEMANTICS_WITH_MISSING_LINK

# %% converted bundles


@pytest.fixture
def scene(annotated_bundle):
    return DocumentFile(annotated_bundle / BundleFile.SCENE).read(SceneDescription)


@pytest.fixture
def world(annotated_bundle):
    return BundleEnvironment(directory=annotated_bundle).create_world()


def test_the_environment_hangs_rigidly_from_the_root(world):
    environment_root = world.get_body_by_name("World")
    assert isinstance(environment_root.parent_connection, FixedConnection)
    assert environment_root.parent_connection.parent is world.root


def test_the_robot_of_a_bundle_is_left_to_the_caller(world, scene):
    [robot_model] = [model for model in scene.models if model.robot]
    assert all(
        not str(body.name).startswith(robot_model.prefix) for body in world.bodies
    )


def test_joints_start_at_the_first_recorded_values(world, annotated_bundle, scene):
    first = DocumentFile(annotated_bundle / scene.trajectory).read(Trajectory).frames[0]
    joints = {
        str(connection.name): connection.position
        for connection in world.connections
        if isinstance(connection, ActiveConnection1DOF)
    }
    assert joints == {name: first[name] for name in joints}


def test_objects_are_free_bodies_at_their_spawn_pose(world, scene):
    [description] = scene.objects
    body = world.get_body_by_name(description.identifier)
    assert isinstance(body.parent_connection, Connection6DoF)
    assert NumericPose.of_entity(world, body).published(4) == type(description.spawn)(
        *(round(value, 4) for value in description.spawn.to_json())
    )
    assert isinstance(body.visual.shapes[0], Mesh)


def test_every_prim_annotates_the_body_of_its_link(world, annotated_bundle):
    prims = BundleContents.of_directory(annotated_bundle).semantics.prims
    annotations = {
        annotation.prim_path: annotation
        for annotation in world.get_semantic_annotations_by_type(UsdPrimAnnotation)
    }
    assert annotations.keys() == prims.keys()
    for path, entry in prims.items():
        assert annotations[path].root.name.name == entry.link
        assert {attribute.name for attribute in annotations[path].attributes} == set(
            entry.semantic_attributes
        )


def test_an_authored_class_adds_its_annotation(world, annotated_bundle):
    prims = BundleContents.of_directory(annotated_bundle).semantics.prims
    [fridge_link] = [
        entry.link
        for entry in prims.values()
        if entry.semantic_attributes.get(SemanticAttributeName.CLASS)
        == AuthoredClass.LAB_REFRIGERATOR
    ]
    [fridge] = world.get_semantic_annotations_by_type(Fridge)
    assert fridge.root.name.name == fridge_link


def test_a_prim_on_a_missing_link_is_refused(annotated_bundle):
    shutil.copy(SEMANTICS_WITH_MISSING_LINK, annotated_bundle / BundleFile.SEMANTICS)
    with pytest.raises(UnknownPrimLink):
        BundleEnvironment(directory=annotated_bundle).create_world()


def test_a_directory_without_a_scene_is_refused(tmp_path):
    with pytest.raises(BundleNotFound):
        BundleEnvironment(directory=tmp_path)


def test_a_bundle_is_found_by_name_in_the_catalog(annotated_bundle):
    catalog = SceneCatalog(
        [SceneDirectory(annotated_bundle.parent, SceneSource.SHARED)]
    )
    environment = BundleEnvironment.of_scene(catalog, annotated_bundle.name)
    assert environment.directory == annotated_bundle.resolve()


def test_the_summary_counts_joints_and_annotations(world):
    text = WorldSummary(world).text()
    prims = len(world.get_semantic_annotations_by_type(UsdPrimAnnotation))
    assert f"Movable joints: {WorldSummary(world).movable_joint_count()}" in text
    assert f"Prim annotations: {prims}" in text


# %% bundles written from a world


def test_a_written_world_builds_again(kitchen, tmp_path):
    objects = WorldObjects(kitchen.world).objects()
    models = WorldModels(kitchen.world, tmp_path, "kitchen").write(objects, None)
    files = ObjectFiles(tmp_path)
    scene = SceneDescription(
        name="kitchen",
        models=models.models,
        objects=[
            files.describe(
                body,
                str(body.id),
                NumericPose.of_entity(kitchen.world, body).published(),
            )
            for body in objects
        ],
    )
    DocumentFile(tmp_path / BundleFile.SCENE).write(scene)
    rebuilt = BundleEnvironment(directory=tmp_path).create_world()
    names = {str(body.name) for body in rebuilt.bodies}
    assert {str(body.name) for body in kitchen.world.bodies} - {"map"} <= names
    by_name = {str(body.name): body for body in rebuilt.bodies}
    assert isinstance(by_name[str(kitchen.table.name)].visual.shapes[0], Box)
    assert by_name[str(kitchen.frame.name)].visual.shapes == []


def test_a_css_colour_becomes_a_native_colour():
    assert ColorText("#ff8000").native().to_rgba() == (1.0, 128 / 255, 0.0, 1.0)
