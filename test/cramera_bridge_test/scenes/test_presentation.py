"""
The authored appearance of a scene carried into written bundles.
"""

from __future__ import annotations

import shutil

import pytest

from cramera.exceptions import PathOutsideDirectory

from cramera_bridge.documents import DocumentFile
from cramera_bridge.exceptions import BundleNotFound
from cramera_bridge.scenes.presentation import ScenePresentation
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.bundle.scene import (
    ModelDescription,
    ObjectDescription,
    SceneDescription,
)
from cramera_protocol.geometry import Pose

OUTSIDE_LIGHTING = "outside_lighting.json"
"""
A lighting file beside a bundle rather than inside it.
"""

# %% presentation


@pytest.fixture
def source(annotated_bundle):
    return DocumentFile(annotated_bundle / BundleFile.SCENE).read(SceneDescription)


def written_scene(source: SceneDescription) -> SceneDescription:
    return SceneDescription(
        name="written",
        models=[
            ModelDescription(name="environment", urdf="e.urdf", prefix="", robot=False),
            ModelDescription(name="robot", urdf="r.urdf", prefix="", robot=True),
        ],
        objects=[
            ObjectDescription(
                identifier=entry.identifier, key=entry.key, spawn=entry.spawn
            )
            for entry in source.objects
        ]
        + [ObjectDescription(identifier="cup", key="cup", spawn=Pose(*[0.0] * 6, 1.0))],
    )


def test_rendering_camera_and_lighting_are_carried_over(
    annotated_bundle, source, tmp_path
):
    scene = written_scene(source)
    ScenePresentation.of_bundle(annotated_bundle).apply(scene, tmp_path)
    assert scene.rendering == source.rendering
    assert scene.camera == source.camera
    lighting = tmp_path / source.rendering.environment
    assert (
        lighting.read_bytes()
        == (annotated_bundle / source.rendering.environment).read_bytes()
    )


def test_a_lighting_file_outside_the_bundle_is_refused(
    annotated_bundle, source, tmp_path
):
    outside = annotated_bundle.parent / OUTSIDE_LIGHTING
    shutil.copy(annotated_bundle / source.rendering.environment, outside)
    source.rendering.environment = f"../{OUTSIDE_LIGHTING}"
    DocumentFile(annotated_bundle / BundleFile.SCENE).write(source)
    destination = tmp_path / "written" / "bundle"
    destination.mkdir(parents=True)
    with pytest.raises(PathOutsideDirectory):
        ScenePresentation.of_bundle(annotated_bundle).apply(
            written_scene(source), destination
        )
    assert not (destination.parent / OUTSIDE_LIGHTING).exists()


def test_materials_are_kept_only_where_the_source_keeps_them(
    annotated_bundle, source, tmp_path
):
    scene = written_scene(source)
    ScenePresentation.of_bundle(annotated_bundle).apply(scene, tmp_path)
    assert [model.preserve_materials for model in scene.models] == [True, None]
    assert [entry.preserve_materials for entry in scene.objects] == [True, None]


def test_the_signature_follows_the_settings(annotated_bundle):
    authored = ScenePresentation.of_bundle(annotated_bundle)
    plain = ScenePresentation(source_directory=annotated_bundle)
    assert authored.signature() != plain.signature()
    assert (
        authored.signature()
        == ScenePresentation.of_bundle(annotated_bundle).signature()
    )


def test_a_scene_without_settings_stays_as_it_is(annotated_bundle, source, tmp_path):
    scene = written_scene(source)
    ScenePresentation(source_directory=annotated_bundle).apply(scene, tmp_path)
    assert scene == written_scene(source)


def test_a_directory_without_a_scene_has_no_presentation(tmp_path):
    with pytest.raises(BundleNotFound):
        ScenePresentation.of_bundle(tmp_path)
