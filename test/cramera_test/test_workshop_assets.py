"""
Blender exports remain portable and retain original workshop attribution.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree

import numpy as np
from pathlib import Path

import pytest

from cramera.offline_assets import OfflineSceneAssets
from .test_laboratory_asset_rights import AssetDocument
from .test_workshop_geometry import CollisionBox


# %% packaged Blender resources
@pytest.fixture
def workshop_assets() -> Path:
    """
    Locate the actual distributed workshop bundle.
    """
    return Path(__file__).parents[2] / "cramera/src/cramera/resources/aicor_workshop"


@pytest.mark.parametrize("name", ["environment", "drawer", "assembly"])
def test_timber_finishes_have_embedded_colour_and_roughness(
    workshop_assets: Path, name: str
) -> None:
    """
    Wood grain survives the export without depending on Blender shader nodes.
    """
    document = AssetDocument.read(workshop_assets / "assets" / (name + ".glb")).document
    textured = [
        material["pbrMetallicRoughness"]
        for material in document["materials"]
        if "baseColorTexture" in material.get("pbrMetallicRoughness", {})
        and "metallicRoughnessTexture" in material["pbrMetallicRoughness"]
    ]
    assert textured
    for finish in textured:
        for channel in ("baseColorTexture", "metallicRoughnessTexture"):
            texture = document["textures"][finish[channel]["index"]]
            image = document["images"][texture["source"]]
            assert "uri" not in image
            assert "bufferView" in image
    assert all("uri" not in buffer for buffer in document["buffers"])


def test_every_export_carries_its_workshop_notices(workshop_assets: Path) -> None:
    """
    Standalone meshes resolve their author and permission metadata locally.
    """
    source = Path(__file__).parents[2] / "cramera/tools/workshop/asset_rights"
    rights = json.loads((source / "rights.json").read_text())
    scene = json.loads((workshop_assets / "scene.json").read_text())
    assert scene["assetRights"] == rights
    for path in (workshop_assets / "assets").glob("*.glb"):
        asset = AssetDocument.read(path).document["asset"]
        assert asset["copyright"] == rights["copyright"]
        embedded = asset["extras"]["assetRights"]
        assert embedded["license"] == rights["license"]
        assert embedded["principalAuthor"] == rights["principalAuthor"]
        for key in ("licenseFile", "noticeFile"):
            assert (path.parent / embedded[key]).resolve().read_bytes() == (
                source / rights[key]
            ).read_bytes()


def test_scene_references_are_local_and_complete(workshop_assets: Path) -> None:
    """
    Authorized copies retain model and loose-part descriptions and textures.
    """
    scene_path = workshop_assets / "scene.json"
    scene = json.loads(scene_path.read_text())
    resolver = OfflineSceneAssets(workshop_assets)
    references = [scene["trajectory"]] + [model["urdf"] for model in scene["models"]]
    for item in scene["objects"]:
        references.extend((item["urdf"], item["mesh"]))
    for reference in references:
        resolved = resolver.resolve_reference(scene_path, reference)
        assert resolved.is_file()
        assert resolved.is_relative_to(workshop_assets)


def test_room_export_batches_small_parts_by_material(workshop_assets: Path) -> None:
    """
    Pegboard holes do not each require their own browser draw call.
    """
    document = AssetDocument.read(workshop_assets / "assets/environment.glb").document
    primitives = sum(len(mesh["primitives"]) for mesh in document["meshes"])
    assert primitives <= 32


def test_branding_is_limited_to_wall_and_bench(workshop_assets: Path) -> None:
    """
    Keep one wall print and one bench print, leaving movable items unbranded.
    """
    branded_triangles = {}
    for path in (workshop_assets / "assets").glob("*.glb"):
        document = AssetDocument.read(path).document
        materials = {
            index
            for index, material in enumerate(document["materials"])
            if material.get("name") == "AICOR teal screen print"
        }
        triangles = sum(
            document["accessors"][primitive["indices"]]["count"] // 3
            for mesh in document["meshes"]
            for primitive in mesh["primitives"]
            if primitive.get("material") in materials
        )
        if triangles:
            branded_triangles[path.stem] = triangles
    assert branded_triangles == {"environment": 4}


def test_shelf_boxes_are_independent_resources(workshop_assets: Path) -> None:
    """
    Each visible shelf container can move with its own handle frame.
    """
    scene = json.loads((workshop_assets / "scene.json").read_text())
    boxes = [
        item for item in scene["workshop"]["bins"] if item.get("kind") == "storage_box"
    ]
    assert len(boxes) == 4
    objects = {item["key"]: item for item in scene["objects"]}
    room = ElementTree.parse(workshop_assets / "environment.urdf")
    supports = [
        CollisionBox.from_element(collision, np.zeros(3))
        for collision in room.find("link[@name='workshop_environment']").findall(
            "collision"
        )
    ]
    for box in boxes:
        item = objects[box["key"]]
        assert item["spawn"] == box["home"]
        assert (workshop_assets / item["urdf"]).is_file()
        assert (workshop_assets / item["mesh"]).is_file()
        model = ElementTree.parse(workshop_assets / item["urdf"])
        origin = np.array(box["home"][:3])
        parts = [
            CollisionBox.from_element(collision, origin)
            for collision in model.find(f"link[@name='{box['key']}']").findall(
                "collision"
            )
        ]
        handle = model.find(f"link[@name='{box['handle']}']")
        joint = model.find(f"joint[@name='{box['handle']}_joint']")
        offset = origin + np.fromstring(joint.find("origin").attrib["xyz"], sep=" ")
        parts.extend(
            CollisionBox.from_element(collision, offset)
            for collision in handle.findall("collision")
        )
        assert [
            (part.name, support.name)
            for part in parts
            for support in supports
            if part.intersects(support)
        ] == []
