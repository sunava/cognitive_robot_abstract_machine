"""
Lab instruments display unobstructed AICOR artwork in the established teal ink.
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
    """Rigid bodies carrying the laboratory identity."""

    ENVIRONMENT = "environment"
    CENTRIFUGE = "centrifuge_base"
    PIPETTING = "pipetting_station"

    @property
    def logos(self) -> tuple[LogoSurface, ...]:
        """Identify every surface that must carry the original artwork."""
        return {
            self.ENVIRONMENT: (LogoSurface.WALL, LogoSurface.BALANCE, LogoSurface.RACK),
            self.CENTRIFUGE: (LogoSurface.CENTRIFUGE,),
            self.PIPETTING: (LogoSurface.DISPENSER,),
        }[self]


class LogoSurface(StrEnum):
    """Artwork surfaces named in the authored scene."""

    WALL = "Wall AICOR logo"
    BALANCE = "Balance AICOR logo"
    RACK = "Rack AICOR logo"
    CENTRIFUGE = "Centrifuge AICOR logo"
    DISPENSER = "Dispenser AICOR logo"

    @property
    def obstacles(self) -> tuple[str, ...]:
        """List controls and structures that must not cover the artwork."""
        return {
            self.WALL: (),
            self.BALANCE: ("Balance display", "Balance digits", "Balance weighing pan"),
            self.RACK: (
                "Rack perforated plate",
                "Rack standoff",
                "Rack standoff.001",
                "Rack standoff.002",
                "Rack standoff.003",
            ),
            self.CENTRIFUGE: (
                "Centrifuge control bezel",
                "Centrifuge display",
                "Centrifuge status lettering",
                "Centrifuge capacity lettering",
                "Centrifuge run key",
                "Centrifuge engraved identity",
            ),
            self.DISPENSER: (
                "Pipetting head",
                "Pipetting head cover",
                "Pipetting head nose",
                "Pipetting instrument label",
                "Pipetting screen bezel",
                "Pipetting display",
                "Pipetting channel label",
            ),
        }[self]


@dataclass(frozen=True)
class FrontBounds:
    """Axis-aligned artwork bounds in an instrument's authored front view."""

    left: float
    """Minimum horizontal coordinate."""
    right: float
    """Maximum horizontal coordinate."""
    bottom: float
    """Minimum height in the Z-up laboratory frame."""
    top: float
    """Maximum height in the Z-up laboratory frame."""

    def separation(self, other: FrontBounds) -> float:
        """Measure the greatest separating gap between two front rectangles."""
        return max(
            self.left - other.right,
            other.left - self.right,
            self.bottom - other.top,
            other.bottom - self.top,
        )


@dataclass(frozen=True)
class BrandedBody:
    """A packaged GLB document and its embedded binary payload."""

    asset: BrandedAsset
    """Rigid body represented by the document."""
    document: dict
    """Mesh, material and image references exported by Blender."""
    binary: bytes
    """Embedded buffer containing geometry and the original logo image."""

    def node(self, name: str) -> dict:
        """Resolve one authored mesh by its stable scene name."""
        return next(node for node in self.document["nodes"] if node["name"] == name)

    def material(self, surface: LogoSurface) -> dict:
        """Read the material assigned to a logo surface."""
        node = self.node(surface)
        primitive = self.document["meshes"][node["mesh"]]["primitives"][0]
        return self.document["materials"][primitive["material"]]

    def front_bounds(self, name: str) -> FrontBounds:
        """Read the X/Z bounds of geometry baked into the asset frame."""
        node = self.node(name)
        assert not {"matrix", "translation", "rotation", "scale"}.intersection(node)
        primitives = self.document["meshes"][node["mesh"]]["primitives"]
        accessors = [
            self.document["accessors"][primitive["attributes"]["POSITION"]]
            for primitive in primitives
        ]
        return FrontBounds(
            min(accessor["min"][0] for accessor in accessors),
            max(accessor["max"][0] for accessor in accessors),
            min(accessor["min"][2] for accessor in accessors),
            max(accessor["max"][2] for accessor in accessors),
        )


@pytest.fixture(params=list(BrandedAsset))
def branded_body(request: pytest.FixtureRequest) -> BrandedBody:
    """Read each distributable instrument without requiring Blender in CI."""
    path = (
        Path(__file__).parents[2]
        / "cramera/src/cramera/resources/precision_lab/assets"
        / f"{request.param}.glb"
    )
    content = path.read_bytes()
    length = struct.unpack_from("<I", content, 12)[0]
    return BrandedBody(
        request.param,
        json.loads(content[20 : 20 + length]),
        content[28 + length :],
    )


def test_dispenser_surroundings_have_no_obsolete_tray_or_pipettes(
    branded_body: BrandedBody,
) -> None:
    """The old decorative tray cannot intersect the functional dispenser."""
    unwanted = ("Work tray", "Pipette body", "Laboratory title", "Pipetting brand")
    assert [
        node["name"]
        for node in branded_body.document["nodes"]
        if node.get("name", "").startswith(unwanted)
    ] == []


def test_every_instrument_and_wall_carry_the_aicor_logo(
    branded_body: BrandedBody,
) -> None:
    """The balance, centrifuge, dispenser and wall all display the identity."""
    logo_names = {
        node["name"]
        for node in branded_body.document["nodes"]
        if "AICOR logo" in node.get("name", "")
    }
    assert logo_names == set(branded_body.asset.logos)


def test_branding_mesh_embeds_the_existing_aicor_logo(
    branded_body: BrandedBody,
) -> None:
    """Every branded surface references the unchanged RGBA logo artwork."""
    document = branded_body.document
    expected = Image.open(
        Path(__file__).parents[2] / "cramera/src/cramera/web/img/aicor-logo.png"
    ).convert("RGBA")
    for surface in branded_body.asset.logos:
        material = branded_body.material(surface)
        texture = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        image = document["images"][document["textures"][texture]["source"]]
        buffer = document["bufferViews"][image["bufferView"]]
        start = buffer.get("byteOffset", 0)
        content = branded_body.binary[start : start + buffer["byteLength"]]
        actual = Image.open(io.BytesIO(content)).convert("RGBA")
        assert actual.size == expected.size
        assert actual.tobytes() == expected.tobytes()


def test_logo_ink_matches_teal_trim_and_keeps_its_transparent_background(
    branded_body: BrandedBody,
) -> None:
    """Only the existing logo's opaque lettering receives the teal tint."""
    trim = next(
        material
        for material in branded_body.document["materials"]
        if material["name"].startswith("Petrol powder coat")
    )
    for surface in branded_body.asset.logos:
        material = branded_body.material(surface)
        assert material["alphaMode"] == "BLEND"
        assert material["pbrMetallicRoughness"]["baseColorFactor"] == pytest.approx(
            trim["pbrMetallicRoughness"]["baseColorFactor"]
        )


def test_branding_has_no_rectangular_background_plaques(
    branded_body: BrandedBody,
) -> None:
    """No solid plaque remains behind any transparent logo surface."""
    assert [
        node["name"]
        for node in branded_body.document["nodes"]
        if "AICOR plaque" in node.get("name", "")
    ] == []


def test_logo_bounds_clear_instrument_controls_and_dispenser_head(
    branded_body: BrandedBody,
) -> None:
    """Front-view artwork has a positive gap to structures and controls."""
    for surface in branded_body.asset.logos:
        bounds = branded_body.front_bounds(surface)
        for obstacle in surface.obstacles:
            assert bounds.separation(branded_body.front_bounds(obstacle)) > 0, (
                surface,
                obstacle,
            )
