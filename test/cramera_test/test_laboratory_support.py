"""
Resting poses, fitted stoppers and rounded collision bottoms share visual dimensions.
"""

from __future__ import annotations

import ast
from collections import Counter
import json
from math import hypot
from pathlib import Path
import struct
import xml.etree.ElementTree as ElementTree

import pytest

from .test_laboratory_bundle import BUILDER, laboratory_directory
from .test_laboratory_geometry import load_geometry


# %% shared visual dimensions
def stopper_profile() -> list[tuple[float, float]]:
    """
    Read the authored stopper cross section from the Blender construction source.
    """
    source = ast.parse(BUILDER.with_name("build_blender.py").read_text())
    method = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == "build_stopper"
    )
    profile = next(
        node.value
        for node in method.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "profile"
            for target in node.targets
        )
    )
    return [
        tuple(ast.literal_eval(value) for value in point.args) for point in profile.elts
    ]


def bottom_triangles(
    directory: Path, description: ElementTree.ElementTree
) -> list[tuple[tuple[float, ...], ...]]:
    """
    Read the actual binary collision mesh referenced by a tube description.
    """
    mesh = description.find(".//collision[@name='glass_bottom']/geometry/mesh")
    assert mesh is not None
    data = (directory / mesh.attrib["filename"]).read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + count * 50
    triangles = []
    for index in range(count):
        values = struct.unpack_from("<12fH", data, 84 + index * 50)
        triangles.append(tuple(tuple(values[start : start + 3]) for start in (3, 6, 9)))
    return triangles


# %% supporting contact planes
def test_rack_tubes_and_stopper_rest_on_their_supports(
    laboratory_directory: Path,
) -> None:
    """
    The resting geometry meets its support without a display-only air gap.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    semantics = json.loads((laboratory_directory / "semantics.json").read_text())
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    base = description.find(
        "link[@name='laboratory_rack']/collision[@name='rack_base']"
    )
    center = float(base.find("origin").attrib["xyz"].split()[2])
    thickness = float(base.find("geometry/box").attrib["size"].split()[2])
    rack_bottom = semantics["rack"]["origin"][2] + center - thickness / 2
    rack_top = rack_bottom + thickness
    assert rack_bottom == pytest.approx(semantics["workSurface"]["height"])
    for slot in scene["laboratory"]["rackSlots"]:
        assert slot["pose"][2] == pytest.approx(rack_top)
    assert (
        scene["laboratory"]["stopper"]["home"][2] == semantics["workSurface"]["height"]
    )


def test_stopper_shoulder_seats_on_the_tube_rim(laboratory_directory: Path) -> None:
    """
    The insertion distance and collision parts agree with the visible shoulder.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    stopper = scene["laboratory"]["stopper"]
    item = next(item for item in scene["objects"] if item["key"] == stopper["key"])
    dimensions = load_geometry().TubeDimensions()
    profile = stopper_profile()
    shoulder = next(
        height for radius, height in profile if radius > dimensions.inner_radius
    )
    height = max(height for _, height in profile)
    assert stopper["insertionDepth"] == shoulder
    assert item["height"] == height
    description = ElementTree.parse(laboratory_directory / item["urdf"])
    stem = description.find(".//collision[@name='stopper_stem']")
    head = description.find(".//collision[@name='stopper_head']")
    assert float(stem.find("geometry/cylinder").attrib["length"]) == shoulder
    assert float(stem.find("geometry/cylinder").attrib["radius"]) == max(
        radius for radius, _ in profile if radius <= dimensions.inner_radius
    )
    assert float(head.find("geometry/cylinder").attrib["radius"]) == max(
        radius for radius, _ in profile
    )
    center = float(head.find("origin").attrib["xyz"].split()[2])
    half_height = float(head.find("geometry/cylinder").attrib["length"]) / 2
    assert center - half_height == pytest.approx(shoulder)
    assert center + half_height == pytest.approx(height)


# %% rounded collision geometry
def test_collision_bottom_matches_the_glass_profile_and_reaches_the_walls(
    laboratory_directory: Path,
) -> None:
    """
    The rounded lower shell closes the collision gap without filling the opening.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    tube = scene["laboratory"]["tubes"][0]
    item = next(item for item in scene["objects"] if item["key"] == tube["key"])
    description = ElementTree.parse(laboratory_directory / item["urdf"])
    triangles = bottom_triangles(laboratory_directory, description)
    dimensions = load_geometry().TubeDimensions()
    profile = dimensions.profile()
    vertices = set(vertex for triangle in triangles for vertex in triangle)
    assert min(vertex[2] for vertex in vertices) == 0
    assert max(vertex[2] for vertex in vertices) == pytest.approx(
        dimensions.bottom_radius
    )
    for horizontal, depth, height in vertices:
        assert (
            min(
                hypot(hypot(horizontal, depth) - point.radius, height - point.height)
                for point in profile
            )
            < 1e-9
        )
    assert any(
        dimensions.wall_thickness < vertex[2] < dimensions.bottom_radius
        for vertex in vertices
    )
    for collision in description.findall(".//collision"):
        if not collision.attrib["name"].startswith("glass_wall"):
            continue
        center = float(collision.find("origin").attrib["xyz"].split()[2])
        size = float(collision.find("geometry/box").attrib["size"].split()[2])
        assert center - size / 2 == pytest.approx(dimensions.bottom_radius)
    for triangle in triangles:
        if all(
            vertex[2] == pytest.approx(dimensions.bottom_radius) for vertex in triangle
        ):
            assert (
                min(hypot(vertex[0], vertex[1]) for vertex in triangle)
                >= dimensions.inner_radius - 1e-9
            )


def test_rounded_collision_shell_is_one_closed_connected_surface(
    laboratory_directory: Path,
) -> None:
    """
    The exported bottom mesh has neither missing faces nor detached pieces.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    item = next(
        item
        for item in scene["objects"]
        if item["key"] == scene["laboratory"]["tubes"][0]["key"]
    )
    description = ElementTree.parse(laboratory_directory / item["urdf"])
    edges = Counter()
    neighbours = {}
    for triangle in bottom_triangles(laboratory_directory, description):
        for start, end in zip(triangle, (*triangle[1:], triangle[0])):
            edges[tuple(sorted((start, end)))] += 1
            neighbours.setdefault(start, set()).add(end)
            neighbours.setdefault(end, set()).add(start)
    assert set(edges.values()) == {2}
    pending = [next(iter(neighbours))]
    reached = set(pending)
    while pending:
        for following in neighbours[pending.pop()] - reached:
            reached.add(following)
            pending.append(following)
    assert reached == set(neighbours)


def test_concave_bottom_is_not_advertised_as_dynamic_contact_validation(
    laboratory_directory: Path,
) -> None:
    """
    The hollow collision mesh explicitly requires preparation for dynamic solvers.
    """
    semantics = json.loads((laboratory_directory / "semantics.json").read_text())
    assert semantics["physics"]["dynamicRequiresConvexDecomposition"] is True
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    assert scene["validation"]["contactPhysicsVerified"] is False
