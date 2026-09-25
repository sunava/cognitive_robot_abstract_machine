"""
Resource boundaries and non-destructive scene layout persistence.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from urllib.parse import urljoin

import pytest
from scipy.spatial.transform import Rotation

from cramera.scene_builder import InvalidSceneBuilderRequest, SceneBuilderService

from .conftest import DATASET_DIR, SCENE, TRAJECTORY


# %% bounded resource catalog
@pytest.fixture
def builder_service(fixture_scene, tmp_path):
    """
    Expose the existing miniature scene and one package with relative meshes.
    """
    resources = tmp_path / "resources"
    package = resources / "example_description"
    (package / "urdf").mkdir(parents=True)
    (package / "meshes").mkdir()
    shutil.copyfile(
        DATASET_DIR / "scene_builder" / "package_model.urdf",
        package / "urdf" / "model.urdf",
    )
    shutil.copyfile(
        fixture_scene / "scenes" / "fixture" / "milk.stl",
        package / "meshes" / "tool.stl",
    )
    (resources / "NOTICE.md").write_text(
        "Existing resource authors retain their rights."
    )
    return SceneBuilderService(
        roots={"scenes": fixture_scene / "scenes", "resources": resources},
        drafts_directory=tmp_path / "drafts",
        url_prefix="/workshop/api/scene-builder/assets/",
    )


def test_catalog_exposes_stable_resource_ids_and_notices(builder_service):
    """
    The catalog exposes relative IDs without disclosing local filesystem paths.
    """
    catalog = builder_service.catalog()
    assets = {asset["id"]: asset for asset in catalog["assets"]}
    model = assets["resources/example_description/urdf/model.urdf"]
    assert (
        model["url"]
        == "/workshop/api/scene-builder/assets/resources/example_description/urdf/model.urdf"
    )
    assert model["kind"] == "urdf"
    assert model["notices"] == [
        {
            "name": "NOTICE.md",
            "url": "/workshop/api/scene-builder/assets/resources/NOTICE.md",
        }
    ]
    assert catalog["scenes"] == [
        {"id": "scenes/fixture/scene.json", "name": SCENE["name"]}
    ]


@pytest.mark.parametrize(
    "identifier",
    [
        "scenes/../secret.stl",
        "scenes/%2e%2e/secret.stl",
        "/etc/passwd",
        "unknown/file.stl",
        "scenes/fixture/robot.urdf/../milk.stl",
        "scenes/fixture/../../outside.stl",
        "scenes/fixture\\robot.urdf",
    ],
)
def test_asset_paths_reject_traversal_and_unknown_roots(builder_service, identifier):
    """
    Request paths cannot select filesystem locations beyond configured roots.
    """
    with pytest.raises(InvalidSceneBuilderRequest):
        builder_service.resolve_asset(identifier)


def test_symlink_escape_is_neither_listed_nor_served(builder_service, tmp_path):
    """
    A model-shaped symlink cannot expose a file outside an allowed resource root.
    """
    outside = tmp_path / "outside.stl"
    outside.write_bytes(b"private")
    (builder_service.roots["resources"] / "escape.stl").symlink_to(outside)
    with pytest.raises(InvalidSceneBuilderRequest):
        builder_service.resolve_asset("resources/escape.stl")
    assert "resources/escape.stl" not in {
        item["id"] for item in builder_service.catalog()["assets"]
    }


def test_unrelated_source_files_cannot_be_served(builder_service):
    """
    Only renderable model dependencies and attribution documents are assets.
    """
    source = builder_service.roots["resources"] / "configuration.py"
    source.write_text("private configuration")
    with pytest.raises(InvalidSceneBuilderRequest):
        builder_service.resolve_asset("resources/configuration.py")


def test_package_and_relative_urdf_meshes_use_protected_asset_urls(builder_service):
    """
    URDF dependencies stay under the same authenticated mount.
    """
    body, media_type = builder_service.render_asset(
        "resources/example_description/urdf/model.urdf"
    )
    document = ElementTree.fromstring(body)
    expected = "/workshop/api/scene-builder/assets/resources/example_description/meshes/tool.stl"
    source = "/workshop/api/scene-builder/assets/resources/example_description/urdf/model.urdf"
    assert [
        urljoin(source, mesh.attrib["filename"]) for mesh in document.iter("mesh")
    ] == [expected, expected]
    assert all(
        not mesh.attrib["filename"].startswith("/") for mesh in document.iter("mesh")
    )
    assert media_type == "application/xml"


# %% existing scene import
def test_scene_import_preserves_robot_spawn_and_initial_joints(builder_service):
    """
    The first recorded state becomes the editable scene's initial state.
    """
    document = builder_service.load_scene("scenes/fixture/scene.json")
    robot = document["instances"][0]
    assert robot["assetId"] == "scenes/fixture/robot.urdf"
    assert robot["position"] == TRAJECTORY["base"][0][:3]
    assert robot["jointPositions"] == {
        key.removeprefix("pr2/"): value
        for key, value in TRAJECTORY["frames"][0].items()
    }
    assert document["instances"][1]["position"] == SCENE["objects"][0]["spawn"]


# %% editable drafts
def test_draft_round_trip_keeps_link_edits_separate_from_source_assets(builder_service):
    """
    Draft writes retain articulated furniture offsets without changing the bundle.
    """
    original = builder_service.resolve_asset("scenes/fixture/robot.urdf").read_bytes()
    document = builder_service.load_scene("scenes/fixture/scene.json")
    document["instances"][0]["linkOffsets"] = {
        "base_link": {"position": [1, 2, 3], "rotation": [0, 0, 0.5]}
    }
    saved = builder_service.save_draft(None, document)
    assert builder_service.load_draft(saved["id"]) == saved
    assert (
        builder_service.resolve_asset("scenes/fixture/robot.urdf").read_bytes()
        == original
    )
    assert builder_service.list_drafts()[0]["id"] == saved["id"]


def test_existing_draft_updates_preserve_identifier(builder_service):
    """
    Saving an existing draft replaces only that draft's document.
    """
    document = builder_service.load_scene("scenes/fixture/scene.json")
    first = builder_service.save_draft(None, document)
    document["name"] = "Rearranged workbench"
    second = builder_service.save_draft(first["id"], document)
    assert second["id"] == first["id"]
    assert (
        builder_service.load_draft(first["id"])["document"]["name"] == document["name"]
    )


@pytest.mark.parametrize(
    "mutation", ["nonfinite", "duplicate", "unknown_asset", "invalid_scale"]
)
def test_invalid_layout_never_creates_a_draft(builder_service, mutation):
    """
    Malformed layouts cannot enter the draft directory.
    """
    document = builder_service.load_scene("scenes/fixture/scene.json")
    instance = document["instances"][0]
    if mutation == "nonfinite":
        instance["position"][0] = float("nan")
    elif mutation == "duplicate":
        document["instances"].append(instance)
    elif mutation == "unknown_asset":
        instance["assetId"] = "scenes/fixture/missing.glb"
    else:
        instance["scale"] = [0, 1, 1]
    with pytest.raises(InvalidSceneBuilderRequest):
        builder_service.save_draft(None, document)
    assert builder_service.list_drafts() == []


def test_draft_identifiers_cannot_escape_storage(builder_service):
    """
    A draft name cannot overwrite a scene bundle or any other file.
    """
    document = builder_service.load_scene("scenes/fixture/scene.json")
    with pytest.raises(InvalidSceneBuilderRequest):
        builder_service.save_draft("../scene", document)


def test_asset_notices_are_preserved_when_saving(builder_service):
    """
    Attribution is recovered from source assets even when a draft omits it.
    """
    document = builder_service.load_scene("scenes/fixture/scene.json")
    document["instances"][0][
        "assetId"
    ] = "resources/example_description/urdf/model.urdf"
    document["notices"] = []
    saved = builder_service.save_draft(None, document)
    assert saved["document"]["notices"] == [
        {
            "name": "NOTICE.md",
            "url": "/workshop/api/scene-builder/assets/resources/NOTICE.md",
        }
    ]


def test_external_gltf_dependencies_are_rejected(builder_service):
    """
    An imported model cannot instruct the browser to fetch a remote buffer.
    """
    asset = builder_service.roots["resources"] / "remote.gltf"
    asset.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "https://outside.invalid/model.bin"}],
            }
        )
    )
    with pytest.raises(InvalidSceneBuilderRequest):
        builder_service.render_asset("resources/remote.gltf")


def test_missing_dependency_is_an_explicit_error(builder_service):
    """
    Unresolved package assets fail instead of falling back to arbitrary paths.
    """
    (
        builder_service.roots["resources"]
        / "example_description"
        / "meshes"
        / "tool.stl"
    ).unlink()
    with pytest.raises(InvalidSceneBuilderRequest):
        builder_service.render_asset("resources/example_description/urdf/model.urdf")


def test_scene_import_preserves_each_models_base_pose(builder_service):
    """
    Environment and robot model bases are independent initial placements.
    """
    trajectory_path = builder_service.roots["scenes"] / "fixture" / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text())
    trajectory["modelBases"] = [{"pr2": [4, 3, 0, 0, 0, 0, 1]}]
    trajectory_path.write_text(json.dumps(trajectory))
    document = builder_service.load_scene("scenes/fixture/scene.json")
    assert (
        document["instances"][0]["position"] == trajectory["modelBases"][0]["pr2"][:3]
    )


def test_scene_import_preserves_identity_at_gimbal_lock(builder_service):
    """
    A quarter-turn around Y remains finite and matches the recorded orientation.
    """
    trajectory_path = builder_service.roots["scenes"] / "fixture" / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text())
    trajectory["base"][0] = [0, 0, 0, 0, 2**-0.5, 0, 2**-0.5]
    trajectory_path.write_text(json.dumps(trajectory))
    document = builder_service.load_scene("scenes/fixture/scene.json")
    assert document["instances"][0]["rotation"] == pytest.approx(
        [0, 1.5707963267948966, 0]
    )


def test_scene_import_uses_urdf_roll_pitch_yaw_for_combined_rotation(builder_service):
    """
    Recorded quaternions follow the viewport's fixed-axis roll, pitch and yaw.
    """
    roll_pitch_yaw = [0.4, -0.3, 0.7]
    quaternion = Rotation.from_euler("xyz", roll_pitch_yaw).as_quat().tolist()
    trajectory_path = builder_service.roots["scenes"] / "fixture" / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text())
    trajectory["base"][0] = [0, 0, 0, *quaternion]
    trajectory_path.write_text(json.dumps(trajectory))
    document = builder_service.load_scene("scenes/fixture/scene.json")
    assert document["instances"][0]["rotation"] == pytest.approx(roll_pitch_yaw)


@pytest.mark.parametrize("pitch", [1.5707963267948966, -1.5707963267948966])
def test_scene_import_keeps_combined_orientation_at_both_gimbal_poles(
    builder_service, pitch
):
    """
    Singular RPY decompositions preserve the rotation with the viewport's zero roll.
    """
    expected = Rotation.from_euler("xyz", [0.4, pitch, -0.7])
    trajectory_path = builder_service.roots["scenes"] / "fixture" / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text())
    trajectory["base"][0] = [0, 0, 0, *expected.as_quat().tolist()]
    trajectory_path.write_text(json.dumps(trajectory))
    document = builder_service.load_scene("scenes/fixture/scene.json")
    rotation = document["instances"][0]["rotation"]
    assert rotation[0] == 0
    assert Rotation.from_euler("xyz", rotation).as_matrix() == pytest.approx(
        expected.as_matrix()
    )
