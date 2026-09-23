"""
Collision and asset contracts for the stationary dispensing instrument.
"""

from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from .test_laboratory_bundle import laboratory_directory

# %% packaged visual instrument


def test_packaged_station_exposes_the_display_and_reservoir_meshes() -> None:
    """
    Runtime display and fill updates can resolve the actual authored mesh names.
    """
    directory = (
        Path(__file__).parents[2] / "cramera/src/cramera/resources/precision_lab"
    )
    scene = json.loads((directory / "scene.json").read_text())
    instrument = scene["laboratory"]["pipetting"]
    content = (directory / instrument["asset"]).read_bytes()
    chunk_length = struct.unpack_from("<I", content, 12)[0]
    document = json.loads(content[20 : 20 + chunk_length])
    meshes = {
        node["name"]: document["meshes"][node["mesh"]]
        for node in document["nodes"]
        if "mesh" in node
    }
    for name in (
        instrument["displayMesh"],
        *(channel["liquidMesh"] for channel in instrument["channels"]),
    ):
        assert name in meshes
        assert all("material" in primitive for primitive in meshes[name]["primitives"])


# %% fixed instrument and accessible vessel target


def test_tube_can_slide_horizontally_into_the_open_cradle(
    laboratory_directory: Path,
) -> None:
    """
    The full tube clears every obstruction on its forward receiving path.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    instrument = scene["laboratory"]["pipetting"]
    semantics = json.loads((laboratory_directory / "semantics.json").read_text())
    tube_radius = semantics["tube"]["outerRadius"]
    tube_bottom = instrument["targetPosition"][2]
    tube_top = tube_bottom + semantics["tube"]["height"]
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    for collision in description.findall(
        f"link[@name='{instrument['body']}']/collision"
    ):
        center = [
            float(value) + instrument["origin"][axis]
            for axis, value in enumerate(collision.find("origin").attrib["xyz"].split())
        ]
        size = [
            float(value)
            for value in collision.find("geometry/box").attrib["size"].split()
        ]
        if (
            center[2] + size[2] / 2 <= tube_bottom + 1e-9
            or center[2] - size[2] / 2 >= tube_top
        ):
            continue
        nearest = (
            max(abs(center[0] - instrument["targetPosition"][0]) - size[0] / 2, 0),
            max(center[1] - size[1] / 2 - instrument["targetPosition"][1], 0),
        )
        assert sum(distance**2 for distance in nearest) > tube_radius**2


def test_station_is_a_separate_fixed_material_asset(laboratory_directory: Path) -> None:
    """
    The instrument is independently authored and fixed to the laboratory.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    instrument = scene["laboratory"]["pipetting"]
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    joint = description.find(f"joint[@name='{instrument['joint']}']")
    assert joint.attrib["type"] == "fixed"
    assert joint.find("child").attrib["link"] == instrument["body"]
    assert [
        float(value) for value in joint.find("origin").attrib["xyz"].split()
    ] == instrument["origin"]
    link = description.find(f"link[@name='{instrument['body']}']")
    assert link.find("visual/geometry/mesh").attrib["filename"] == instrument["asset"]
    assert instrument["joint"] not in scene["models"][0]["movableJoints"]


def test_target_is_on_a_supported_platform(laboratory_directory: Path) -> None:
    """
    The declared tube root rests on the same elevation as the collision plate.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    instrument = scene["laboratory"]["pipetting"]
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    platform = description.find(
        f"link[@name='{instrument['body']}']/collision[@name='pipetting_platform']"
    )
    center = [float(value) for value in platform.find("origin").attrib["xyz"].split()]
    size = [
        float(value) for value in platform.find("geometry/box").attrib["size"].split()
    ]
    assert instrument["targetPosition"][2] == pytest.approx(
        instrument["origin"][2] + center[2] + size[2] / 2
    )
    for axis in range(2):
        assert (
            abs(
                instrument["targetPosition"][axis]
                - instrument["origin"][axis]
                - center[axis]
            )
            < size[axis] / 2
        )


def test_both_nozzles_clear_the_open_glass_rim(laboratory_directory: Path) -> None:
    """
    Two independent feeds aim inside the vessel without touching its rim.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    instrument = scene["laboratory"]["pipetting"]
    semantics = json.loads((laboratory_directory / "semantics.json").read_text())
    assert {channel["id"] for channel in instrument["channels"]} == {"amber", "teal"}
    rim = instrument["targetPosition"][2] + semantics["tube"]["height"]
    for channel in instrument["channels"]:
        nozzle = channel["nozzlePosition"]
        assert 0.012 <= nozzle[2] - rim <= 0.020
        assert (
            sum(
                (nozzle[axis] - instrument["targetPosition"][axis]) ** 2
                for axis in range(2)
            )
            < semantics["tube"]["innerRadius"] ** 2
        )


def test_front_grasp_has_clearance_below_the_nozzle(laboratory_directory: Path) -> None:
    """
    The complete approach corridor stays open around the front grasp height.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    instrument = scene["laboratory"]["pipetting"]
    semantics = json.loads((laboratory_directory / "semantics.json").read_text())
    grasp = instrument["targetPosition"][2] + semantics["tube"]["graspHeight"]
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    for collision in description.findall(
        f"link[@name='{instrument['body']}']/collision"
    ):
        center = [
            float(value) + instrument["origin"][axis]
            for axis, value in enumerate(collision.find("origin").attrib["xyz"].split())
        ]
        shape = collision.find("geometry/box")
        assert shape is not None
        size = [float(value) for value in shape.attrib["size"].split()]
        if abs(center[2] - grasp) >= size[2] / 2 + 0.025:
            continue
        assert center[1] - size[1] / 2 > instrument["targetPosition"][1] + 0.04
