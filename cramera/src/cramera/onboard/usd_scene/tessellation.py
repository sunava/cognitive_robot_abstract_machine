"""
Triangle meshes from USD geometry data: polygon meshes with their primvar
interpolations, and the implicit shapes (cube, sphere, cylinder, cone, capsule).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy

# %% primvar data


class Interpolation(StrEnum):
    """
    How a primvar's values are spread over a mesh, named as USD names it.
    """

    CONSTANT = "constant"
    """
    One value for the whole mesh.
    """

    UNIFORM = "uniform"
    """
    One value per face.
    """

    VARYING = "varying"
    """
    One value per point, interpolated linearly.
    """

    VERTEX = "vertex"
    """
    One value per point.
    """

    FACE_VARYING = "faceVarying"
    """
    One value per face corner.
    """


class Axis(StrEnum):
    """
    The axis a round implicit shape is built around, named as USD names it.
    """

    X = "X"
    """
    The X axis.
    """

    Y = "Y"
    """
    The Y axis.
    """

    Z = "Z"
    """
    The Z axis.
    """

    def rotation_from_z(self) -> numpy.ndarray:
        """
        The proper rotation taking the Z axis onto this axis.
        """
        return {
            Axis.X: numpy.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
            Axis.Y: numpy.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
            Axis.Z: numpy.eye(3),
        }[self]


@dataclass(eq=False)
class PrimvarData:
    """
    The flattened values of one primvar and how they are spread over the mesh.
    """

    values: numpy.ndarray
    """
    ``(V, D)`` values, already de-indexed.
    """

    interpolation: Interpolation
    """
    How the values map onto points, faces or corners.
    """

    def per_corner(
        self,
        face_vertex_indices: numpy.ndarray,
        face_of_corner: numpy.ndarray,
    ) -> numpy.ndarray | None:
        """
        One value per face corner, or None when the value count does not match the
        interpolation.

        :param face_vertex_indices: The point index of every corner.
        :param face_of_corner: The face index of every corner.
        """
        expected = {
            Interpolation.CONSTANT: 1,
            Interpolation.UNIFORM: int(face_of_corner.max(initial=-1)) + 1,
            Interpolation.FACE_VARYING: len(face_vertex_indices),
        }.get(self.interpolation)
        if expected is not None and len(self.values) < expected:
            return None
        if self.interpolation is Interpolation.CONSTANT:
            return numpy.repeat(self.values[:1], len(face_vertex_indices), axis=0)
        if self.interpolation is Interpolation.UNIFORM:
            return self.values[face_of_corner]
        if self.interpolation is Interpolation.FACE_VARYING:
            return self.values[: len(face_vertex_indices)]
        if len(self.values) <= int(face_vertex_indices.max(initial=-1)):
            return None
        return self.values[face_vertex_indices]

    @property
    def per_point(self) -> bool:
        """
        Whether the values follow the points.
        """
        return self.interpolation in (Interpolation.VERTEX, Interpolation.VARYING)


# %% triangle meshes


@dataclass(eq=False)
class TriangleMesh:
    """
    Indexed triangles with optional normals and texture coordinates per vertex.
    """

    points: numpy.ndarray
    """
    ``(N, 3)`` positions.
    """

    triangles: numpy.ndarray
    """
    ``(M, 3)`` vertex indices, counter-clockwise seen from outside.
    """

    normals: numpy.ndarray | None = None
    """
    ``(N, 3)`` unit normals.
    """

    texture_coordinates: numpy.ndarray | None = None
    """
    ``(N, 2)`` texture coordinates.
    """

    faces: numpy.ndarray | None = None
    """
    ``(M,)`` index of the source polygon of every triangle.
    """

    def subset(self, keep: numpy.ndarray) -> TriangleMesh:
        """
        The triangles selected by a mask, with unused vertices dropped.

        :param keep: A boolean mask over the triangles.
        """
        triangles = self.triangles[keep]
        used, remapped = numpy.unique(triangles, return_inverse=True)
        return TriangleMesh(
            points=self.points[used],
            triangles=remapped.reshape(-1, 3).astype(numpy.uint32),
            normals=None if self.normals is None else self.normals[used],
            texture_coordinates=(
                None
                if self.texture_coordinates is None
                else self.texture_coordinates[used]
            ),
            faces=None if self.faces is None else self.faces[keep],
        )

    def transformed(self, matrix: numpy.ndarray) -> TriangleMesh:
        """
        The mesh under an affine transform; a mirroring transform reverses the winding
        so the triangles keep facing outwards.

        :param matrix: A homogeneous 4x4 matrix acting on column vectors.
        """
        linear = matrix[:3, :3]
        normals = None
        if self.normals is not None:
            normals = self.normals @ numpy.linalg.inv(linear)
            lengths = numpy.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / numpy.where(lengths > 0.0, lengths, 1.0)
        triangles = self.triangles
        if numpy.linalg.det(linear) < 0.0:
            triangles = triangles[:, ::-1]
        return TriangleMesh(
            points=self.points @ linear.T + matrix[:3, 3],
            triangles=triangles,
            normals=normals,
            texture_coordinates=self.texture_coordinates,
            faces=self.faces,
        )


@dataclass(eq=False)
class PolygonMesh:
    """
    A USD polygon mesh: points, faces as corner lists, and primvars.
    """

    points: numpy.ndarray
    """
    ``(N, 3)`` positions.
    """

    face_vertex_counts: numpy.ndarray
    """
    ``(F,)`` corners per face.
    """

    face_vertex_indices: numpy.ndarray
    """
    ``(K,)`` point index of every corner, face after face.
    """

    normals: PrimvarData | None = None
    """
    The normals, if authored.
    """

    texture_coordinates: PrimvarData | None = None
    """
    The texture coordinates, if authored.
    """

    left_handed: bool = False
    """
    Whether the faces wind clockwise seen from outside (``orientation`` leftHanded).
    """

    def triangulate(self) -> TriangleMesh:
        """
        Fan-triangulate the faces, keeping per-point data shared and splitting points
        only where a primvar varies per face or corner.
        """
        counts = numpy.asarray(self.face_vertex_counts, dtype=numpy.int64)
        indices = numpy.asarray(self.face_vertex_indices, dtype=numpy.int64)
        face_of_corner = numpy.repeat(numpy.arange(len(counts)), counts)
        corners = self.fan_corners(counts)
        if self.left_handed:
            corners = corners[:, ::-1]
        normals = self.corner_values(self.normals, indices, face_of_corner)
        texture_coordinates = self.corner_values(
            self.texture_coordinates, indices, face_of_corner
        )
        faces = face_of_corner[corners[:, 0]]
        shared = all(
            primvar is None or values is None or primvar.per_point
            for primvar, values in (
                (self.normals, normals),
                (self.texture_coordinates, texture_coordinates),
            )
        )
        if shared:
            return TriangleMesh(
                points=numpy.asarray(self.points, dtype=numpy.float64),
                triangles=indices[corners].astype(numpy.uint32),
                normals=(
                    None if normals is None else self.normals.values[: len(self.points)]
                ),
                texture_coordinates=(
                    None
                    if texture_coordinates is None
                    else self.texture_coordinates.values[: len(self.points)]
                ),
                faces=faces,
            )
        return self.welded(
            numpy.asarray(self.points, dtype=numpy.float64)[indices],
            normals,
            texture_coordinates,
            corners,
            faces,
        )

    @staticmethod
    def fan_corners(counts: numpy.ndarray) -> numpy.ndarray:
        """
        The corner indices of a fan triangulation, ``(first, k, k + 1)`` per face.

        :param counts: Corners per face; faces with fewer than three are dropped.
        """
        starts = numpy.concatenate([[0], numpy.cumsum(counts)[:-1]])
        per_face = numpy.where(counts >= 3, counts - 2, 0)
        face = numpy.repeat(numpy.arange(len(counts)), per_face)
        first_triangle = numpy.repeat(numpy.cumsum(per_face) - per_face, per_face)
        step = numpy.arange(int(per_face.sum())) - first_triangle + 1
        anchor = starts[face]
        return numpy.stack([anchor, anchor + step, anchor + step + 1], axis=1)

    @staticmethod
    def corner_values(
        primvar: PrimvarData | None,
        indices: numpy.ndarray,
        face_of_corner: numpy.ndarray,
    ) -> numpy.ndarray | None:
        """
        A primvar's value per corner, or None when absent or malformed.

        :param primvar: The primvar.
        :param indices: The point index of every corner.
        :param face_of_corner: The face index of every corner.
        """
        if primvar is None:
            return None
        return primvar.per_corner(indices, face_of_corner)

    @staticmethod
    def welded(
        positions: numpy.ndarray,
        normals: numpy.ndarray | None,
        texture_coordinates: numpy.ndarray | None,
        corners: numpy.ndarray,
        faces: numpy.ndarray,
    ) -> TriangleMesh:
        """
        Triangles over per-corner vertices, with identical corners merged.

        :param positions: The position of every corner.
        :param normals: The normal of every corner.
        :param texture_coordinates: The texture coordinates of every corner.
        :param corners: The corner indices of every triangle.
        :param faces: The source face of every triangle.
        """
        columns = [positions]
        columns += [] if normals is None else [normals]
        columns += [] if texture_coordinates is None else [texture_coordinates]
        rows = numpy.concatenate(columns, axis=1)
        unique, inverse = numpy.unique(rows, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        return TriangleMesh(
            points=unique[:, :3],
            triangles=inverse[corners].astype(numpy.uint32),
            normals=None if normals is None else unique[:, 3:6],
            texture_coordinates=(
                None if texture_coordinates is None else unique[:, -2:]
            ),
            faces=faces,
        )


# %% implicit shapes


@dataclass
class ShapeTessellation:
    """
    Triangle meshes for USD's implicit shapes, centred on the origin as USD defines
    them.
    """

    segments: int = 32
    """
    Subdivisions around the axis of a round shape.
    """

    rings: int = 16
    """
    Subdivisions from pole to pole of a sphere (half of that per capsule cap).
    """

    def box(self, size: float) -> TriangleMesh:
        """
        A cube with edge length ``size`` and one flat normal per face.

        :param size: The edge length.
        """
        points, normals, triangles = [], [], []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                normal = numpy.zeros(3)
                normal[axis] = sign
                first = numpy.roll(normal, 1) * sign
                second = numpy.cross(normal, first)
                base = len(points)
                for along, across in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    points.append((normal + along * first + across * second) * size / 2)
                    normals.append(normal)
                triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        return TriangleMesh(
            numpy.array(points),
            numpy.array(triangles, numpy.uint32),
            numpy.array(normals),
        )

    def sphere(self, radius: float) -> TriangleMesh:
        """
        A sphere.

        :param radius: The radius.
        """
        latitude = numpy.linspace(-numpy.pi / 2, numpy.pi / 2, self.rings + 1)
        directions = numpy.stack([numpy.cos(latitude), numpy.sin(latitude)], axis=1)
        return self.lathe(directions * radius, directions)

    def cylinder(self, radius: float, height: float, axis: Axis) -> TriangleMesh:
        """
        A capped cylinder.

        :param radius: The radius.
        :param height: The length along the axis.
        :param axis: The axis.
        """
        half = height / 2
        profile = numpy.array(
            [
                [0, -half],
                [radius, -half],
                [radius, -half],
                [radius, half],
                [radius, half],
                [0, half],
            ],
            dtype=float,
        )
        normals = numpy.array(
            [[0, -1], [0, -1], [1, 0], [1, 0], [0, 1], [0, 1]], dtype=float
        )
        return self.oriented(self.lathe(profile, normals), axis)

    def cone(self, radius: float, height: float, axis: Axis) -> TriangleMesh:
        """
        A capped cone with its apex on the positive side of the axis.

        :param radius: The base radius.
        :param height: The length along the axis.
        :param axis: The axis.
        """
        half = height / 2
        side = numpy.array([height, radius]) / numpy.hypot(height, radius)
        profile = numpy.array(
            [[0, -half], [radius, -half], [radius, -half], [0, half]], dtype=float
        )
        normals = numpy.array([[0, -1], [0, -1], side, side], dtype=float)
        return self.oriented(self.lathe(profile, normals), axis)

    def capsule(self, radius: float, height: float, axis: Axis) -> TriangleMesh:
        """
        A cylinder with hemispherical caps; the height excludes the caps.

        :param radius: The radius.
        :param height: The length of the cylindrical part.
        :param axis: The axis.
        """
        half_rings = max(self.rings // 2, 1)
        lower = numpy.linspace(-numpy.pi / 2, 0.0, half_rings + 1)
        upper = numpy.linspace(0.0, numpy.pi / 2, half_rings + 1)
        directions = numpy.concatenate(
            [
                numpy.stack([numpy.cos(lower), numpy.sin(lower)], axis=1),
                numpy.stack([numpy.cos(upper), numpy.sin(upper)], axis=1),
            ]
        )
        offsets = numpy.concatenate(
            [
                numpy.full(half_rings + 1, -height / 2),
                numpy.full(half_rings + 1, height / 2),
            ]
        )
        profile = directions * radius + numpy.stack(
            [numpy.zeros_like(offsets), offsets], axis=1
        )
        return self.oriented(self.lathe(profile, directions), axis)

    def lathe(self, profile: numpy.ndarray, normals: numpy.ndarray) -> TriangleMesh:
        """
        Revolve a profile about the Z axis; consecutive equal profile points form a
        crease and are not bridged.

        :param profile:``(P, 2)`` radius and height, from the bottom to the top.
        :param normals:``(P, 2)`` radial and vertical normal components.
        """
        angles = numpy.linspace(0.0, 2.0 * numpy.pi, self.segments + 1)
        cosine, sine = numpy.cos(angles), numpy.sin(angles)
        ring = self.segments + 1
        points = numpy.stack(
            [
                numpy.outer(profile[:, 0], cosine).ravel(),
                numpy.outer(profile[:, 0], sine).ravel(),
                numpy.repeat(profile[:, 1], ring),
            ],
            axis=1,
        )
        vertex_normals = numpy.stack(
            [
                numpy.outer(normals[:, 0], cosine).ravel(),
                numpy.outer(normals[:, 0], sine).ravel(),
                numpy.repeat(normals[:, 1], ring),
            ],
            axis=1,
        )
        triangles = []
        for level in range(len(profile) - 1):
            if numpy.allclose(profile[level], profile[level + 1]):
                continue
            for step in range(self.segments):
                lower_start = level * ring + step
                lower_end = lower_start + 1
                upper_end, upper_start = lower_end + ring, lower_start + ring
                triangles += [
                    (lower_start, lower_end, upper_end),
                    (lower_start, upper_end, upper_start),
                ]
        return TriangleMesh(
            points, numpy.array(triangles, numpy.uint32), vertex_normals
        )

    @staticmethod
    def oriented(mesh: TriangleMesh, axis: Axis) -> TriangleMesh:
        """
        A shape built around Z, turned onto another axis.

        :param mesh: The shape around Z.
        :param axis: The target axis.
        """
        rotation = numpy.eye(4)
        rotation[:3, :3] = axis.rotation_from_z()
        return mesh.transformed(rotation)
