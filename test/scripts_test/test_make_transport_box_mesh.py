"""
What the transport-box generator writes out.

The script lives in ``scripts/``, which is not an importable package, so it is loaded
from its path. It has to be registered in ``sys.modules`` before it is executed,
because :mod:`dataclasses` resolves a class's annotations through the module it
belongs to.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from typing_extensions import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "make_transport_box_mesh.py"


@pytest.fixture(scope="module")
def script() -> Any:
    """
    The generator module, loaded from its path in ``scripts/``.
    """
    specification = importlib.util.spec_from_file_location(
        "make_transport_box_mesh", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def box(script) -> Any:
    """
    A small labelled box, so a test needs to name only what it asserts on.
    """
    return script.TransportBox(
        name="screw_box",
        label="SCREWS",
        captions=["M6 x 60"],
        bounding_box=script.BoundingBox((-0.03, -0.03, -0.09), (0.03, 0.03, 0.11)),
        texture_size=64,
    )


# %% the bounding box the mesh is built around
def test_the_corners_start_at_the_minimum_and_end_at_the_maximum(script):
    """
    :attr:`TransportBox.FACES` indexes into this order, so it is part of the contract.
    """
    bounding_box = script.BoundingBox((-1.0, -2.0, -3.0), (1.0, 2.0, 3.0))

    corners = bounding_box.corners()

    assert corners.shape == (8, 3)
    assert np.allclose(corners[0], bounding_box.minimum)
    assert np.allclose(corners[6], bounding_box.maximum)


def test_every_face_names_four_corners_that_exist(script):
    for _, corners, _ in script.TransportBox.FACES:
        assert len(corners) == 4
        assert all(0 <= corner <= 7 for corner in corners)


def test_the_bounding_box_of_a_mesh_is_the_mesh_extent(script, tmp_path):
    mesh_path = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(0.2, 0.4, 0.6)).export(mesh_path)

    bounding_box = script.BoundingBox.of_mesh(mesh_path)

    assert np.allclose(bounding_box.minimum, (-0.1, -0.2, -0.3))
    assert np.allclose(bounding_box.maximum, (0.1, 0.2, 0.3))


# %% the OBJ and its material
def test_the_obj_declares_one_vertex_per_corner(box):
    vertices = [line for line in box.obj_text().splitlines() if line.startswith("v ")]

    assert len(vertices) == 8


def test_the_obj_gives_every_face_its_own_texture_corners(box, script):
    lines = box.obj_text().splitlines()
    faces = [line for line in lines if line.startswith("f ")]
    texture_coordinates = [line for line in lines if line.startswith("vt ")]
    normals = [line for line in lines if line.startswith("vn ")]

    assert len(faces) == len(script.TransportBox.FACES)
    assert len(texture_coordinates) == 4 * len(script.TransportBox.FACES)
    assert len(normals) == len(script.TransportBox.FACES)


def test_the_obj_points_at_the_material_it_writes(box):
    assert "mtllib screw_box.mtl" in box.obj_text()
    assert "usemtl screw_box" in box.obj_text()


def test_the_material_points_at_the_texture_it_writes(box):
    assert "map_Kd screw_box.png" in box.mtl_text()


# %% texture coordinates
def test_a_texture_region_is_flipped_from_texture_space_to_pixels(box):
    """
    Texture ``v`` runs up from the bottom while pixel ``y`` runs down from the top, so
    the bottom-left quarter of the texture is the bottom-left quarter in pixels.
    """
    left, top, right, bottom = box._pixels((0.0, 0.0, 0.5, 0.5))

    assert (left, right) == (0, 32)
    assert (top, bottom) == (32, 64)


# %% writing the files out
def test_writing_produces_the_obj_the_material_and_the_texture(box, tmp_path):
    written = box.write(tmp_path)

    assert sorted(path.name for path in written) == [
        "screw_box.mtl",
        "screw_box.obj",
        "screw_box.png",
    ]
    assert all(path.exists() for path in written)


def test_the_written_texture_is_square_at_the_requested_size(box, tmp_path):
    box.write(tmp_path)

    with Image.open(tmp_path / "screw_box.png") as texture:
        assert texture.size == (box.texture_size, box.texture_size)


def test_the_written_obj_names_the_material_file_beside_it(box, tmp_path):
    box.write(tmp_path)

    obj_text = (tmp_path / "screw_box.obj").read_text(encoding="utf-8")
    assert "mtllib screw_box.mtl" in obj_text
    assert (tmp_path / "screw_box.mtl").exists()
