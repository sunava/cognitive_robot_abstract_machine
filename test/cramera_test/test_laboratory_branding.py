"""
The authored lab uses the existing AICOR identity and clears the dispenser.
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pytest
from PIL import Image


# %% packaged branding assets
class BrandedAsset(StrEnum):
    """
    Rigid bodies that display the original AICOR artwork.
    """

    ENVIRONMENT = "environment"
    PIPETTING = "pipetting_station"


@dataclass(frozen=True)
class BrandedBody:
    """
    A packaged GLB document and its embedded binary payload.
    """

    document: dict
    """
    Mesh, material and image references exported by Blender.
    """

    binary: bytes
    """
    Embedded buffer containing geometry and the original logo image.
    """


@pytest.fixture(params=list(BrandedAsset))
def branded_body(request: pytest.FixtureRequest) -> BrandedBody:
    """
    Read each distributable instrument without requiring Blender in CI.
    """
    path = (
        Path(__file__).parents[2]
        / "cramera/src/cramera/resources/precision_lab/assets"
        / f"{request.param}.glb"
    )
    content = path.read_bytes()
    length = struct.unpack_from("<I", content, 12)[0]
    return BrandedBody(json.loads(content[20 : 20 + length]), content[28 + length :])


def test_dispenser_surroundings_have_no_obsolete_tray_or_pipettes(
    branded_body: BrandedBody,
) -> None:
    """
    The old decorative tray cannot intersect the functional dispenser.
    """
    unwanted = ("Work tray", "Pipette body", "Laboratory title", "Pipetting brand")
    assert [
        node["name"]
        for node in branded_body.document["nodes"]
        if node.get("name", "").startswith(unwanted)
    ] == []


def test_branding_mesh_embeds_the_existing_aicor_logo(
    branded_body: BrandedBody,
) -> None:
    """
    Every branded rigid body references the unchanged RGBA logo artwork.
    """
    document = branded_body.document
    logo_nodes = [
        node for node in document["nodes"] if "AICOR logo" in node.get("name", "")
    ]
    assert len(logo_nodes) == 1
    primitive = document["meshes"][logo_nodes[0]["mesh"]]["primitives"][0]
    material = document["materials"][primitive["material"]]
    texture = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    image = document["images"][document["textures"][texture]["source"]]
    buffer = document["bufferViews"][image["bufferView"]]
    start = buffer.get("byteOffset", 0)
    content = branded_body.binary[start : start + buffer["byteLength"]]
    expected = Image.open(
        Path(__file__).parents[2] / "cramera/src/cramera/web/img/aicor-logo.png"
    ).convert("RGBA")
    actual = Image.open(io.BytesIO(content)).convert("RGBA")
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()
