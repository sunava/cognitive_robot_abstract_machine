"""
Triangulation of USD polygon meshes and tessellation of implicit shapes.
"""

from __future__ import annotations

import numpy
import pytest

from cramera.onboard.usd_scene.tessellation import (
    Axis,
    Interpolation,
    PolygonMesh,
    PrimvarData,
    ShapeTessellation,
    TriangleMesh,
)

# %% polygon meshes


def unit_square(**primvars) -> PolygonMesh:
    """
    A counter-clockwise unit square in the XY plane as one quad.

    :param primvars: Extra fields of the polygon mesh.
    """
    return PolygonMesh(
        points=numpy.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
        face_vertex_counts=numpy.array([4]),
        face_vertex_indices=numpy.array([0, 1, 2, 3]),
        **primvars,
    )


def facing(mesh: TriangleMesh) -> numpy.ndarray:
    """
    The unit geometric normal of every triangle.

    :param mesh: The triangles.
    """
    corners = mesh.points[mesh.triangles]
    normals = numpy.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return normals / numpy.linalg.norm(normals, axis=1, keepdims=True)


def signed_volume(mesh: TriangleMesh) -> float:
    """
    The enclosed volume, positive when the triangles face outwards.

    :param mesh: A closed triangle mesh.
    """
    corners = mesh.points[mesh.triangles]
    return float(
        numpy.einsum(
            "ij,ij->i", corners[:, 0], numpy.cross(corners[:, 1], corners[:, 2])
        ).sum()
        / 6.0
    )


class TestPolygonTriangulation:
    """
    Fan triangulation keeping the winding and spreading primvars.
    """

    def test_quad_becomes_two_triangles_facing_up(self):
        mesh = unit_square().triangulate()
        assert mesh.triangles.tolist() == [[0, 1, 2], [0, 2, 3]]
        assert numpy.allclose(facing(mesh), [[0, 0, 1], [0, 0, 1]])

    def test_left_handed_faces_are_reversed(self):
        mesh = unit_square(left_handed=True).triangulate()
        assert numpy.allclose(facing(mesh), [[0, 0, -1], [0, 0, -1]])

    def test_vertex_primvars_keep_points_shared(self):
        coordinates = numpy.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        mesh = unit_square(
            texture_coordinates=PrimvarData(coordinates, Interpolation.VERTEX)
        ).triangulate()
        assert len(mesh.points) == 4
        assert numpy.array_equal(mesh.texture_coordinates, coordinates)

    def test_face_varying_primvars_split_points_per_corner(self):
        coordinates = numpy.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        mesh = unit_square(
            texture_coordinates=PrimvarData(coordinates, Interpolation.FACE_VARYING)
        ).triangulate()
        corners = mesh.triangles.ravel()
        assert numpy.array_equal(
            mesh.texture_coordinates[corners], coordinates[[0, 1, 2, 0, 2, 3]]
        )
        assert numpy.array_equal(
            mesh.points[corners][:, :2], coordinates[[0, 1, 2, 0, 2, 3]]
        )

    def test_mismatched_primvar_is_dropped(self):
        mesh = unit_square(
            normals=PrimvarData(numpy.zeros((2, 3)), Interpolation.FACE_VARYING)
        ).triangulate()
        assert mesh.normals is None

    def test_faces_with_fewer_than_three_corners_are_dropped(self):
        mesh = PolygonMesh(
            points=numpy.eye(3),
            face_vertex_counts=numpy.array([2, 3]),
            face_vertex_indices=numpy.array([0, 1, 0, 1, 2]),
        ).triangulate()
        assert mesh.triangles.tolist() == [[0, 1, 2]]
        assert mesh.faces.tolist() == [1]


def test_mirroring_transform_keeps_triangles_facing_outwards():
    mirror = numpy.diag([1.0, 1.0, -1.0, 1.0])
    box = ShapeTessellation().box(1.0).transformed(mirror)
    assert signed_volume(box) == pytest.approx(1.0)


# %% implicit shapes


@pytest.mark.parametrize("axis", list(Axis))
def test_round_shapes_are_closed_and_face_outwards(axis):
    tessellation = ShapeTessellation(segments=64, rings=32)
    radius, height = 0.5, 2.0
    assert signed_volume(tessellation.cylinder(radius, height, axis)) == pytest.approx(
        numpy.pi * radius**2 * height, rel=0.01
    )
    assert signed_volume(tessellation.cone(radius, height, axis)) == pytest.approx(
        numpy.pi * radius**2 * height / 3.0, rel=0.01
    )
    assert signed_volume(tessellation.capsule(radius, height, axis)) == pytest.approx(
        numpy.pi * radius**2 * (height + 4.0 * radius / 3.0), rel=0.01
    )


@pytest.mark.parametrize("axis", list(Axis))
def test_round_shapes_extend_along_their_axis(axis):
    cylinder = ShapeTessellation().cylinder(0.5, 2.0, axis)
    extent = cylinder.points.max(axis=0) - cylinder.points.min(axis=0)
    assert extent[list(Axis).index(axis)] == pytest.approx(2.0)


@pytest.mark.parametrize("axis", list(Axis))
def test_cone_apex_lies_on_the_positive_side_of_its_axis(axis):
    radius, height = 0.5, 2.0
    cone = ShapeTessellation().cone(radius, height, axis)
    direction = numpy.eye(3)[list(Axis).index(axis)]
    along = cone.points @ direction
    tip = cone.points[numpy.isclose(along, along.max())]
    assert along.max() == pytest.approx(height / 2.0)
    assert numpy.allclose(tip - numpy.outer(tip @ direction, direction), 0.0)


def test_sphere_normals_point_away_from_the_centre():
    sphere = ShapeTessellation().sphere(2.0)
    assert numpy.allclose(sphere.normals * 2.0, sphere.points)
    assert signed_volume(sphere) > 0.0


def test_box_has_one_flat_normal_per_face():
    box = ShapeTessellation().box(2.0)
    assert numpy.allclose(abs(box.points).max(axis=0), [1.0, 1.0, 1.0])
    assert numpy.allclose(facing(box), box.normals[box.triangles[:, 0]])
