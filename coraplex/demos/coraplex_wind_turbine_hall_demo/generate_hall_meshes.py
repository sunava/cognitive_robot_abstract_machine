"""
Generate the meshes, textures and transported items of the wind-turbine hall demo.

Produces into ``meshes/`` next to this script:

* ``large_turbine_blade.stl`` — a nine-meter rotor blade, lofted from a circular
  root flange into a cambered airfoil that tapers and untwists towards the tip,
  lying along +x with its root at the origin.
* ``nacelle_shell.obj``/``.mtl``/``nacelle.png`` — a rounded rectangular nacelle
  housing (superellipsoid), drivetrain axis along +y, with its baked paint job.
* ``tower_section.obj``/``.mtl``/``tower.png`` — a unit cylinder with painted-steel
  texture, instanced at different scales for the tower sections.
* ``spinner_cone.stl`` — the rounded spinner nose of a rotor hub, pointing +y.
* ``torque_wrench.stl``, ``bolt_crate.stl``, ``empty_crate.stl`` — the items the
  robot transports, shaped so they read as what they are.
* ``concrete.png``/``floor.obj``/``floor.mtl`` — the textured hall floor.
* ``wood.png``/``crate_wood.obj``/``crate_wood.mtl`` — a plank-textured unit cube
  the hall model instances for pallets, shelf boards and crates.

Also writes ``background.jpg`` next to this script: the photo backdrop the viewer's
*Background image* layer shows behind the scene.

Run once (the generated files are checked in)::

    python generate_hall_meshes.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import trimesh

# %% blade geometry


@dataclass
class BladeGeometry:
    """
    Parameters of one lofted rotor blade.
    """

    length: float
    """
    Root-to-tip length, in meters.
    """

    root_radius: float
    """
    Radius of the circular root flange, in meters.
    """

    maximum_chord: float
    """
    The widest chord, in meters, reached at :attr:`maximum_chord_station`.
    """

    tip_chord: float
    """
    Chord right before the tip closes, in meters.
    """

    maximum_chord_station: float = 0.28
    """
    Fraction of the length at which the chord is widest.
    """

    root_blend_end: float = 0.18
    """
    Fraction of the length over which the circular root blends into the airfoil.
    """

    root_twist: float = math.radians(16.0)
    """
    Twist of the airfoil at the root end of the blend; fades to zero at the tip.
    """

    thickness_ratio_root: float = 0.30
    """
    Airfoil thickness as a fraction of the chord where the root blend ends.
    """

    thickness_ratio_tip: float = 0.10
    """
    Airfoil thickness as a fraction of the chord at the tip.
    """

    camber_ratio: float = 0.035
    """
    Maximum camber of the airfoil as a fraction of the chord.
    """

    pivot_chord_fraction: float = 0.35
    """
    Chord fraction the profile is centered on, so the blade twists about its spar.
    """

    station_count: int = 48
    """
    Number of cross sections along the blade.
    """

    ring_point_count: int = 64
    """
    Number of points around each cross section.
    """

    @staticmethod
    def airfoil_half_thickness(chord_position: np.ndarray, ratio: float) -> np.ndarray:
        """
        Half thickness of a NACA four-digit airfoil along the chord.

        :param chord_position: Positions along the chord in ``[0, 1]``.
        :param ratio: Thickness as a fraction of the chord.
        :return: Half thickness at each position, as a fraction of the chord.
        """
        return (
            5.0
            * ratio
            * (
                0.2969 * np.sqrt(chord_position)
                - 0.1260 * chord_position
                - 0.3516 * chord_position**2
                + 0.2843 * chord_position**3
                - 0.1015 * chord_position**4
            )
        )

    def chord(self, span_fraction: float) -> float:
        """
        :param span_fraction: Position along the blade as a fraction of its length.
        :return: Chord length at that station, widening to the maximum and tapering
            to the tip.
        """
        if span_fraction <= self.maximum_chord_station:
            blend = span_fraction / self.maximum_chord_station
            return 2 * self.root_radius + (
                self.maximum_chord - 2 * self.root_radius
            ) * math.sin(blend * math.pi / 2)
        taper = (span_fraction - self.maximum_chord_station) / (
            1.0 - self.maximum_chord_station
        )
        return self.maximum_chord + (self.tip_chord - self.maximum_chord) * taper**0.9

    def ring(self, span_fraction: float) -> np.ndarray:
        """
        :param span_fraction: Position along the blade as a fraction of its length.
        :return: That cross section as ``(ring_point_count, 3)`` vertices.
        """
        angles = np.linspace(0.0, 2.0 * math.pi, self.ring_point_count, endpoint=False)
        chord_position = 0.5 * (1.0 + np.cos(angles))
        thickness_ratio = (
            self.thickness_ratio_root
            + (self.thickness_ratio_tip - self.thickness_ratio_root)
            * span_fraction**0.8
        )
        half_thickness = self.airfoil_half_thickness(chord_position, thickness_ratio)
        camber = self.camber_ratio * np.sin(math.pi * chord_position)
        sign = np.where(np.sin(angles) >= 0.0, 1.0, -1.0)
        chord = self.chord(span_fraction)
        airfoil_y = -(chord_position - self.pivot_chord_fraction) * chord
        airfoil_z = (camber + sign * half_thickness) * chord
        circle_y = self.root_radius * np.cos(angles)
        circle_z = self.root_radius * np.sin(angles)
        if span_fraction >= self.root_blend_end:
            weight = 0.0
        else:
            weight = 0.5 * (
                1.0 + math.cos(math.pi * span_fraction / self.root_blend_end)
            )
        y = weight * circle_y + (1.0 - weight) * airfoil_y
        z = weight * circle_z + (1.0 - weight) * airfoil_z
        twist = self.root_twist * (1.0 - span_fraction) ** 1.6
        twisted_y = y * math.cos(twist) - z * math.sin(twist)
        twisted_z = y * math.sin(twist) + z * math.cos(twist)
        x = np.full_like(y, span_fraction * self.length)
        return np.column_stack([x, twisted_y, twisted_z])

    def loft(self) -> trimesh.Trimesh:
        """
        :return: The closed blade surface lofted through all stations.
        """
        rings = [
            self.ring(span_fraction)
            for span_fraction in np.linspace(0.0, 1.0, self.station_count)
        ]
        vertices = np.concatenate(rings)
        faces = []
        for station_index in range(self.station_count - 1):
            base = station_index * self.ring_point_count
            for point_index in range(self.ring_point_count):
                next_point = (point_index + 1) % self.ring_point_count
                a = base + point_index
                b = base + next_point
                c = a + self.ring_point_count
                d = b + self.ring_point_count
                faces.append([a, b, c])
                faces.append([b, d, c])
        root_center = len(vertices)
        tip_center = root_center + 1
        vertices = np.vstack([vertices, rings[0].mean(axis=0), rings[-1].mean(axis=0)])
        tip_base = (self.station_count - 1) * self.ring_point_count
        for point_index in range(self.ring_point_count):
            next_point = (point_index + 1) % self.ring_point_count
            faces.append([root_center, next_point, point_index])
            faces.append([tip_center, tip_base + point_index, tip_base + next_point])
        blade = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
        blade.fix_normals()
        return blade


# %% rounded industrial shells


def superellipsoid(
    half_extents: tuple[float, float, float], boxiness: float
) -> trimesh.Trimesh:
    """
    A rounded box built by inflating a unit sphere towards a cube.

    :param half_extents: Half sizes along x, y and z, in meters.
    :param boxiness: Componentwise exponent in ``(0, 1]``; 1 keeps the ellipsoid,
        smaller values approach a box with rounded edges.
    :return: The watertight shell.
    """
    shell = trimesh.creation.icosphere(subdivisions=4)
    vertices = shell.vertices
    inflated = np.sign(vertices) * np.abs(vertices) ** boxiness
    shell.vertices = inflated * np.asarray(half_extents)
    shell.fix_normals()
    return shell


def spinner_cone(radius: float, length: float) -> trimesh.Trimesh:
    """
    The rounded spinner nose of a rotor hub, pointing along +y.

    :param radius: Radius at the hub flange, in meters.
    :param length: Length from flange to nose tip, in meters.
    :return: The watertight spinner shell.
    """
    shell = trimesh.creation.icosphere(subdivisions=4)
    vertices = shell.vertices.copy()
    # stretch the front hemisphere into the nose, keep the rear rounded
    forward = vertices[:, 1] > 0.0
    vertices[forward, 1] *= length / radius * 0.85
    shell.vertices = vertices * radius
    shell.fix_normals()
    return shell


# %% transported items


def torque_wrench() -> trimesh.Trimesh:
    """
    A click-type torque wrench lying flat, its handle along +y.

    Grip and adjustment knob at the -y end, ratchet ring head with the square drive stub
    at the +y end. Centered on its bounding box, so the body origin is the item's
    geometric center.
    """
    shaft = trimesh.creation.cylinder(radius=0.013, height=0.19, sections=24)
    shaft.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    shaft.apply_translation([0, 0.025, 0])
    grip = trimesh.creation.cylinder(radius=0.021, height=0.10, sections=24)
    grip.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    grip.apply_translation([0, -0.12, 0])
    knob = trimesh.creation.cylinder(radius=0.0235, height=0.02, sections=24)
    knob.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    knob.apply_translation([0, -0.18, 0])
    head = trimesh.creation.annulus(r_min=0.012, r_max=0.033, height=0.028)
    head.apply_translation([0, 0.145, 0])
    drive = trimesh.creation.cylinder(radius=0.0085, height=0.02, sections=6)
    drive.apply_translation([0, 0.145, 0.02])
    wrench = trimesh.util.concatenate([shaft, grip, knob, head, drive])
    wrench.apply_translation(-wrench.bounds.mean(axis=0))
    return wrench


def small_load_carrier(with_bolts: bool) -> trimesh.Trimesh:
    """
    A small industrial load carrier (KLT-style stacking bin) with an outward rim.

    :param with_bolts: Whether the bin is filled, with hex bolt heads showing above the
        fill plate; otherwise the bin is visibly empty.
    :return: The bin, centered on its bounding box.
    """
    length, width, height, wall = 0.13, 0.10, 0.09, 0.01
    parts = [
        trimesh.creation.box([length, width, wall]).apply_translation([0, 0, wall / 2])
    ]
    for side in (-1.0, 1.0):
        wall_y = trimesh.creation.box([length, wall, height - wall])
        wall_y.apply_translation([0, side * (width - wall) / 2, (height + wall) / 2])
        parts.append(wall_y)
        wall_x = trimesh.creation.box([wall, width - 2 * wall, height - wall])
        wall_x.apply_translation([side * (length - wall) / 2, 0, (height + wall) / 2])
        parts.append(wall_x)
        rim_y = trimesh.creation.box([length + 0.012, 0.014, 0.01])
        rim_y.apply_translation([0, side * width / 2, height - 0.005])
        parts.append(rim_y)
        rim_x = trimesh.creation.box([0.014, width + 0.012, 0.01])
        rim_x.apply_translation([side * length / 2, 0, height - 0.005])
        parts.append(rim_x)
    if with_bolts:
        fill = trimesh.creation.box([length - 2 * wall, width - 2 * wall, 0.008])
        fill.apply_translation([0, 0, 0.055])
        parts.append(fill)
        for bolt_x in (-0.04, 0.0, 0.04):
            for bolt_y in (-0.022, 0.022):
                bolt = trimesh.creation.cylinder(radius=0.009, height=0.016, sections=6)
                bolt.apply_translation([bolt_x, bolt_y, 0.065])
                parts.append(bolt)
    carrier = trimesh.util.concatenate(parts)
    carrier.apply_translation(-carrier.bounds.mean(axis=0))
    return carrier


# %% procedural textures and textured meshes

TEXTURE_SIZE = 512
"""
Side length of the generated texture images, in pixels.
"""


def concrete_texture() -> "np.ndarray":
    """
    :return: One tileable concrete floor tile with an expansion joint along two
        edges, as an ``(TEXTURE_SIZE, TEXTURE_SIZE, 3)`` uint8 array.
    """
    rng = np.random.default_rng(7)
    base = np.full((TEXTURE_SIZE, TEXTURE_SIZE), 200.0)
    from PIL import Image, ImageFilter

    blotches = rng.normal(0.0, 22.0, (TEXTURE_SIZE // 8, TEXTURE_SIZE // 8))
    blotches = np.asarray(
        Image.fromarray((blotches + 128).clip(0, 255).astype(np.uint8))
        .resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.BILINEAR)
        .filter(ImageFilter.GaussianBlur(9)),
        dtype=float,
    )
    base += (blotches - 128.0) * 0.35
    base += rng.normal(0.0, 3.5, base.shape)
    speckles = rng.random(base.shape) < 0.004
    base[speckles] -= 26.0
    base[:3, :] = 172.0
    base[:, :3] = 172.0
    grey = base.clip(0, 255).astype(np.uint8)
    return np.stack([grey, grey, (grey * 1.01).clip(0, 255).astype(np.uint8)], axis=2)


def wood_texture() -> "np.ndarray":
    """
    :return: A plank-wood texture as an ``(TEXTURE_SIZE, TEXTURE_SIZE, 3)`` uint8
        array, planks running vertically.
    """
    rng = np.random.default_rng(11)
    plank_count = 4
    plank_width = TEXTURE_SIZE // plank_count
    red = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE))
    green = np.zeros_like(red)
    blue = np.zeros_like(red)
    rows = np.arange(TEXTURE_SIZE)
    for plank in range(plank_count):
        tone = rng.normal(0.0, 9.0)
        grain = 8.0 * np.sin(
            rows / rng.uniform(9.0, 16.0) + rng.uniform(0, 6)
        ) + rng.normal(0.0, 5.0, TEXTURE_SIZE)
        column_slice = slice(plank * plank_width, (plank + 1) * plank_width)
        red[:, column_slice] = 172 + tone + grain[:, None]
        green[:, column_slice] = 128 + tone * 0.8 + grain[:, None] * 0.8
        blue[:, column_slice] = 82 + tone * 0.6 + grain[:, None] * 0.6
        red[:, plank * plank_width : plank * plank_width + 3] = 96
        green[:, plank * plank_width : plank * plank_width + 3] = 74
        blue[:, plank * plank_width : plank * plank_width + 3] = 52
    return np.stack(
        [channel.clip(0, 255).astype(np.uint8) for channel in (red, green, blue)],
        axis=2,
    )


FLOOR_EXTENTS = ((-13.1, 11.1), (-7.7, 7.7))
"""
The hall floor rectangle in world coordinates, as ``((min_x, max_x), (min_y, max_y))``.
"""

FLOOR_TILE_METERS = 2.7
"""
Edge length one concrete texture tile covers on the floor, in meters.
"""


def write_floor_mesh(output_directory: str) -> None:
    """
    Write ``floor.obj``/``floor.mtl``: one textured quad covering the hall floor.

    :param output_directory: The ``meshes/`` directory the files are written into.
    """
    (min_x, max_x), (min_y, max_y) = FLOOR_EXTENTS
    tiles_u = (max_x - min_x) / FLOOR_TILE_METERS
    tiles_v = (max_y - min_y) / FLOOR_TILE_METERS
    with open(os.path.join(output_directory, "floor.obj"), "w") as obj:
        obj.write("mtllib floor.mtl\nusemtl concrete\n")
        obj.write("v %f %f 0.001\n" % (min_x, min_y))
        obj.write("v %f %f 0.001\n" % (max_x, min_y))
        obj.write("v %f %f 0.001\n" % (max_x, max_y))
        obj.write("v %f %f 0.001\n" % (min_x, max_y))
        obj.write(
            "vt 0 0\nvt %f 0\nvt %f %f\nvt 0 %f\n"
            % (tiles_u, tiles_u, tiles_v, tiles_v)
        )
        obj.write("vn 0 0 1\n")
        obj.write("f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n")
    with open(os.path.join(output_directory, "floor.mtl"), "w") as mtl:
        mtl.write(
            "newmtl concrete\nKa 1 1 1\nKd 1 1 1\nKs 0.06 0.06 0.06\nNs 12\n"
            "map_Kd concrete.png\nmap_bump -bm 0.35 concrete_bump.png\n"
        )


def write_wood_cube_mesh(output_directory: str) -> None:
    """
    Write ``crate_wood.obj``/``crate_wood.mtl``: a plank-textured unit cube that the
    hall model instances at different scales for pallets, shelf boards and crates.

    :param output_directory: The ``meshes/`` directory the files are written into.
    """
    corners = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    faces = [
        ((1, 2, 6, 5), (1, 0, 0)),
        ((3, 0, 4, 7), (-1, 0, 0)),
        ((2, 3, 7, 6), (0, 1, 0)),
        ((0, 1, 5, 4), (0, -1, 0)),
        ((4, 5, 6, 7), (0, 0, 1)),
        ((3, 2, 1, 0), (0, 0, -1)),
    ]
    with open(os.path.join(output_directory, "crate_wood.obj"), "w") as obj:
        obj.write("mtllib crate_wood.mtl\nusemtl wood\n")
        for corner in corners:
            obj.write("v %f %f %f\n" % corner)
        obj.write("vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n")
        for _, normal in faces:
            obj.write("vn %d %d %d\n" % normal)
        for face_index, (vertex_indices, _) in enumerate(faces):
            obj.write(
                "f %d/1/%d %d/2/%d %d/3/%d %d/4/%d\n"
                % (
                    vertex_indices[0] + 1,
                    face_index + 1,
                    vertex_indices[1] + 1,
                    face_index + 1,
                    vertex_indices[2] + 1,
                    face_index + 1,
                    vertex_indices[3] + 1,
                    face_index + 1,
                )
            )
    with open(os.path.join(output_directory, "crate_wood.mtl"), "w") as mtl:
        mtl.write(
            "newmtl wood\nKa 1 1 1\nKd 1 1 1\nKs 0.04 0.04 0.04\nNs 8\n"
            "map_Kd wood.png\nmap_bump -bm 0.4 wood_bump.png\n"
        )


def write_textures(output_directory: str) -> None:
    """
    Write ``concrete.png`` and ``wood.png`` into the mesh directory.

    :param output_directory: The ``meshes/`` directory the files are written into.
    """
    from PIL import Image

    Image.fromarray(concrete_texture()).save(
        os.path.join(output_directory, "concrete.png")
    )
    Image.fromarray(wood_texture()).save(os.path.join(output_directory, "wood.png"))
    Image.fromarray(concrete_bump_texture()).save(
        os.path.join(output_directory, "concrete_bump.png")
    )
    Image.fromarray(wood_bump_texture()).save(
        os.path.join(output_directory, "wood_bump.png")
    )


# %% textured turbine shells


def nacelle_texture() -> "np.ndarray":
    """
    The nacelle housing's baked paint job, as an ``(1024, 1024, 3)`` uint8 array.

    White gelcoat with panel seams, teal accent bands along both sides, ambient shading
    towards the underside and the housing ends, grime rising from the bottom edge, and
    small warning decals. Image x is the circumferential UV coordinate (seam at the
    hidden underside), image y runs rear to nose.
    """
    size = 1024
    rng = np.random.default_rng(23)
    base = np.full((size, size), 236.0)
    base += rng.normal(0.0, 1.8, base.shape)
    from PIL import Image, ImageFilter

    mottling = rng.normal(0.0, 10.0, (size // 16, size // 16))
    mottling = np.asarray(
        Image.fromarray((mottling + 128).clip(0, 255).astype(np.uint8))
        .resize((size, size), Image.BILINEAR)
        .filter(ImageFilter.GaussianBlur(6)),
        dtype=float,
    )
    base += (mottling - 128.0) * 0.25

    red, green, blue = base.copy(), base.copy() + 2.0, base.copy() + 4.0

    # panel seams: rings along the body and lengthwise joints
    for ring in (1, 2, 3, 4, 5):
        row = int(size * ring / 6)
        for channel in (red, green, blue):
            channel[row - 1 : row + 1, :] -= 34.0
            channel[row + 1 : row + 3, :] += 8.0
    for joint in (0.125, 0.375, 0.625, 0.875):
        column = int(size * joint)
        for channel in (red, green, blue):
            channel[:, column - 1 : column + 1] -= 30.0

    # teal accent bands along both sides of the housing
    for band_center in (0.26, 0.74):
        start, end = int(size * (band_center - 0.035)), int(
            size * (band_center + 0.035)
        )
        red[:, start:end] = 12.0
        green[:, start:end] = 132.0
        blue[:, start:end] = 140.0
        for channel, tone in ((red, 236.0), (green, 238.0), (blue, 240.0)):
            channel[:, start - 4 : start - 1] = tone - 16.0
            channel[:, end + 1 : end + 4] = tone - 16.0

    # ambient shading: darker towards the underside (u near 0 and 1) and the ends
    u_coordinates = np.linspace(0.0, 1.0, size)
    underside = 1.0 - 0.30 * np.exp(
        -(((np.minimum(u_coordinates, 1.0 - u_coordinates)) / 0.12) ** 2)
    )
    v_coordinates = np.linspace(0.0, 1.0, size)
    ends = 1.0 - 0.10 * np.exp(
        -(((np.minimum(v_coordinates, 1.0 - v_coordinates)) / 0.05) ** 2)
    )
    shading = ends[:, None] * underside[None, :]
    red *= shading
    green *= shading
    blue *= shading

    # grime streaks creeping up from the underside
    for _ in range(60):
        column = int(rng.uniform(0, size))
        u = column / size
        if 0.18 < u < 0.82:
            continue
        length = int(rng.uniform(20, 90))
        row = int(rng.uniform(0, size - length))
        strength = rng.uniform(6.0, 18.0)
        for channel in (red, green, blue):
            channel[row : row + length, column : column + 2] -= strength

    # bolt circle at the nose end
    for column in range(0, size, 20):
        for channel in (red, green, blue):
            channel[size - 22 : size - 17, column : column + 5] -= 60.0

    # warning decals
    for decal_u, decal_v in ((0.35, 0.22), (0.63, 0.72)):
        x0, y0 = int(size * decal_u), int(size * decal_v)
        red[y0 : y0 + 26, x0 : x0 + 38] = 240.0
        green[y0 : y0 + 26, x0 : x0 + 38] = 196.0
        blue[y0 : y0 + 26, x0 : x0 + 38] = 30.0
        for channel in (red, green, blue):
            channel[y0 : y0 + 2, x0 : x0 + 38] = 40.0
            channel[y0 + 24 : y0 + 26, x0 : x0 + 38] = 40.0
            channel[y0 : y0 + 26, x0 : x0 + 2] = 40.0
            channel[y0 : y0 + 26, x0 + 36 : x0 + 38] = 40.0

    stacked = np.stack(
        [channel.clip(0, 255).astype(np.uint8) for channel in (red, green, blue)],
        axis=2,
    )
    return np.flipud(stacked)


def tower_texture() -> "np.ndarray":
    """
    Painted tower-section steel, as a ``(512, 512, 3)`` uint8 array: brushed vertical
    streaks, weld seams and bolt rows near both flanges.

    Image y runs along the section's axis.
    """
    size = 512
    rng = np.random.default_rng(31)
    base = np.full((size, size), 238.0)
    base += rng.normal(0.0, 1.5, (1, size))  # vertical brush streaks
    base += rng.normal(0.0, 1.2, base.shape)
    for ring in (0.25, 0.5, 0.75):
        row = int(size * ring)
        base[row - 2 : row + 1, :] -= 36.0
        base[row + 1 : row + 3, :] += 10.0
    for row in (int(size * 0.035), int(size * 0.965)):
        for column in range(0, size, 24):
            base[row - 3 : row + 3, column : column + 6] -= 70.0
    grey = base.clip(0, 255).astype(np.uint8)
    return np.flipud(
        np.stack([grey, grey + 2, grey + 4], axis=2).clip(0, 255).astype(np.uint8)
    )


def nacelle_bump_texture() -> "np.ndarray":
    """
    Height map matching :func:`nacelle_texture`: recessed panel seams, raised accent-
    band edges and bolt heads, as an ``(1024, 1024)`` uint8 array.
    """
    size = 1024
    rng = np.random.default_rng(29)
    height_map = np.full((size, size), 128.0)
    height_map += rng.normal(0.0, 2.5, height_map.shape)
    for ring in (1, 2, 3, 4, 5):
        row = int(size * ring / 6)
        height_map[row - 1 : row + 2, :] = 78.0
    for joint in (0.125, 0.375, 0.625, 0.875):
        column = int(size * joint)
        height_map[:, column - 1 : column + 2] = 82.0
    for band_center in (0.26, 0.74):
        for edge in (band_center - 0.035, band_center + 0.035):
            column = int(size * edge)
            height_map[:, column - 1 : column + 1] = 176.0
    for column in range(0, size, 20):
        height_map[size - 22 : size - 17, column : column + 5] = 200.0
    return np.flipud(height_map.clip(0, 255).astype(np.uint8))


def tower_bump_texture() -> "np.ndarray":
    """
    Height map matching :func:`tower_texture`: raised weld beads and bolt heads over
    brushed steel, as a ``(512, 512)`` uint8 array.
    """
    size = 512
    rng = np.random.default_rng(37)
    height_map = np.full((size, size), 128.0)
    height_map += rng.normal(0.0, 2.0, (1, size))
    for ring in (0.25, 0.5, 0.75):
        row = int(size * ring)
        height_map[row - 2 : row + 2, :] = 196.0
    for row in (int(size * 0.035), int(size * 0.965)):
        for column in range(0, size, 24):
            height_map[row - 3 : row + 3, column : column + 6] = 215.0
    return np.flipud(height_map.clip(0, 255).astype(np.uint8))


def concrete_bump_texture() -> "np.ndarray":
    """
    Height map for the concrete floor: pitted surface with recessed expansion joints, as
    a ``(TEXTURE_SIZE, TEXTURE_SIZE)`` uint8 array.
    """
    rng = np.random.default_rng(41)
    height_map = np.full((TEXTURE_SIZE, TEXTURE_SIZE), 128.0)
    height_map += rng.normal(0.0, 4.0, height_map.shape)
    pits = rng.random(height_map.shape) < 0.003
    height_map[pits] -= 45.0
    height_map[:4, :] = 70.0
    height_map[:, :4] = 70.0
    return height_map.clip(0, 255).astype(np.uint8)


def wood_bump_texture() -> "np.ndarray":
    """
    Height map for the plank wood: recessed gaps between planks and grain ridges, as a
    ``(TEXTURE_SIZE, TEXTURE_SIZE)`` uint8 array.
    """
    rng = np.random.default_rng(43)
    plank_width = TEXTURE_SIZE // 4
    height_map = np.full((TEXTURE_SIZE, TEXTURE_SIZE), 128.0)
    rows = np.arange(TEXTURE_SIZE)
    for plank in range(4):
        grain = 5.0 * np.sin(rows / rng.uniform(9.0, 16.0)) + rng.normal(
            0.0, 2.5, TEXTURE_SIZE
        )
        height_map[:, plank * plank_width : (plank + 1) * plank_width] += grain[:, None]
        height_map[:, plank * plank_width : plank * plank_width + 3] = 62.0
    return height_map.clip(0, 255).astype(np.uint8)


def write_nacelle_mesh(output_directory: str) -> None:
    """
    Write the UV-mapped nacelle housing: ``nacelle_shell.obj``/``.mtl`` and its baked
    ``nacelle.png`` paint job.

    The UV seam lies on the hidden underside, and the texture's v axis runs from the
    rear of the housing to the nose.

    :param output_directory: The ``meshes/`` directory the files are written into.
    """
    from PIL import Image

    shell = superellipsoid(NACELLE_HALF_EXTENTS, NACELLE_BOXINESS)
    vertices = shell.vertices
    u = 0.5 + np.arctan2(vertices[:, 0], vertices[:, 2]) / (2.0 * np.pi)
    v = (vertices[:, 1] / NACELLE_HALF_EXTENTS[1] + 1.0) / 2.0
    with open(os.path.join(output_directory, "nacelle_shell.obj"), "w") as obj:
        obj.write("mtllib nacelle_shell.mtl\nusemtl nacelle\n")
        for vertex in vertices:
            obj.write("v %f %f %f\n" % tuple(vertex))
        for vertex_u, vertex_v in zip(u, v):
            obj.write("vt %f %f\n" % (vertex_u, vertex_v))
        for face in shell.faces:
            obj.write(
                "f %d/%d %d/%d %d/%d\n"
                % (
                    face[0] + 1,
                    face[0] + 1,
                    face[1] + 1,
                    face[1] + 1,
                    face[2] + 1,
                    face[2] + 1,
                )
            )
    with open(os.path.join(output_directory, "nacelle_shell.mtl"), "w") as mtl:
        mtl.write(
            "newmtl nacelle\nKa 1 1 1\nKd 1 1 1\nKs 0.22 0.22 0.22\nNs 40\n"
            "map_Kd nacelle.png\nmap_bump -bm 0.5 nacelle_bump.png\n"
        )
    Image.fromarray(nacelle_texture()).save(
        os.path.join(output_directory, "nacelle.png")
    )
    Image.fromarray(nacelle_bump_texture()).save(
        os.path.join(output_directory, "nacelle_bump.png")
    )


def write_tower_mesh(output_directory: str) -> None:
    """
    Write the UV-mapped tower section: a unit cylinder (radius 1, height 1, axis z) as
    ``tower_section.obj``/``.mtl`` with its ``tower.png`` steel paint, meant to be
    instanced at different scales.

    :param output_directory: The ``meshes/`` directory the files are written into.
    """
    from PIL import Image

    segment_count = 48
    angles = np.linspace(0.0, 2.0 * np.pi, segment_count + 1)
    with open(os.path.join(output_directory, "tower_section.obj"), "w") as obj:
        obj.write("mtllib tower_section.mtl\nusemtl tower\n")
        for z in (-0.5, 0.5):
            for angle in angles:
                obj.write("v %f %f %f\n" % (np.cos(angle), np.sin(angle), z))
        obj.write("v 0 0 -0.5\nv 0 0 0.5\n")
        for v_row in (0.0, 1.0):
            for segment in range(segment_count + 1):
                obj.write("vt %f %f\n" % (segment / segment_count, v_row))
        obj.write("vt 0.02 0.02\n")
        ring = segment_count + 1
        cap_bottom, cap_top = 2 * ring + 1, 2 * ring + 2
        cap_uv = 2 * ring + 1
        for segment in range(segment_count):
            a, b = segment + 1, segment + 2
            c, d = ring + segment + 1, ring + segment + 2
            obj.write("f %d/%d %d/%d %d/%d\n" % (a, a, b, b, d, ring + b))
            obj.write("f %d/%d %d/%d %d/%d\n" % (a, a, d, ring + b, c, ring + a))
            obj.write(
                "f %d/%d %d/%d %d/%d\n" % (cap_bottom, cap_uv, b, cap_uv, a, cap_uv)
            )
            obj.write("f %d/%d %d/%d %d/%d\n" % (cap_top, cap_uv, c, cap_uv, d, cap_uv))
    with open(os.path.join(output_directory, "tower_section.mtl"), "w") as mtl:
        mtl.write(
            "newmtl tower\nKa 1 1 1\nKd 1 1 1\nKs 0.2 0.2 0.2\nNs 35\n"
            "map_Kd tower.png\nmap_bump -bm 0.4 tower_bump.png\n"
        )
    Image.fromarray(tower_texture()).save(os.path.join(output_directory, "tower.png"))
    Image.fromarray(tower_bump_texture()).save(
        os.path.join(output_directory, "tower_bump.png")
    )


# %% the photo backdrop for the viewer's background layer


def write_backdrop(demo_directory: str) -> None:
    """
    Write ``background.jpg``: an out-of-focus industrial-hall backdrop for the viewer's
    *Background image* layer, warm skylights over dark steel trusses.

    The viewer blurs and darkens the layer further, so the image only has to read as a
    factory hall at bokeh level. Drop a real photo of the presented plant over this file
    for the genuine look.

    :param demo_directory: The demo directory the image is written into.
    """
    from PIL import Image, ImageDraw, ImageFilter

    width, height = 1600, 1000
    rng = np.random.default_rng(3)
    image = Image.new("RGB", (width, height))
    drawer = ImageDraw.Draw(image)
    for row in range(height):
        t = row / height
        if t < 0.62:
            tone = (
                int(212 - 90 * t),
                int(204 - 92 * t),
                int(188 - 90 * t),
            )
        else:
            floor_t = (t - 0.62) / 0.38
            tone = (
                int(118 + 26 * floor_t),
                int(116 + 24 * floor_t),
                int(112 + 22 * floor_t),
            )
        drawer.line([(0, row), (width, row)], fill=tone)
    # skylight glow
    for blob_x, blob_width in ((260, 300), (760, 340), (1280, 300)):
        drawer.ellipse(
            [blob_x - blob_width, -140, blob_x + blob_width, 140],
            fill=(248, 240, 218),
        )
    # roof trusses
    for truss_row in (70, 160, 250):
        drawer.rectangle([0, truss_row, width, truss_row + 16], fill=(66, 63, 58))
    # columns
    for column_x in range(90, width, 230):
        drawer.rectangle([column_x, 120, column_x + 26, 640], fill=(76, 72, 66))
    # amber hall lights with floor reflections
    for light_x in range(200, width, 260):
        drawer.ellipse([light_x - 9, 292, light_x + 9, 310], fill=(255, 206, 128))
        drawer.ellipse([light_x - 26, 700, light_x + 26, 860], fill=(160, 152, 138))
    # vague equipment silhouettes on the floor line
    for block_x, block_width, block_height in (
        (120, 210, 130),
        (540, 260, 170),
        (1050, 230, 150),
        (1400, 190, 120),
    ):
        top = 640 - block_height
        drawer.rectangle(
            [block_x, top, block_x + block_width, 640],
            fill=(
                int(rng.uniform(90, 120)),
                int(rng.uniform(88, 116)),
                int(rng.uniform(86, 112)),
            ),
        )
    image = image.filter(ImageFilter.GaussianBlur(7))
    image.save(os.path.join(demo_directory, "background.jpg"), quality=88)


# %% writing the meshes

LARGE_BLADE = BladeGeometry(
    length=9.0, root_radius=0.34, maximum_chord=1.15, tip_chord=0.16
)
"""
The rotor blade stored on the racks along the hall's south wall.
"""

NACELLE_HALF_EXTENTS = (1.25, 2.1, 1.15)
"""
Half sizes of the nacelle housing along x, y and z, in meters.
"""

NACELLE_BOXINESS = 0.38
"""
How box-like the nacelle shell is rounded; real nacelles are close to rounded boxes.
"""

SPINNER_RADIUS = 0.85
"""
Radius of the spinner at its hub flange, in meters.
"""

SPINNER_LENGTH = 1.1
"""
Length of the spinner from flange to nose tip, in meters.
"""


def main() -> None:
    """
    Write all hall meshes into ``meshes/`` next to this script.
    """
    output_directory = os.path.join(os.path.dirname(__file__), "meshes")
    meshes = {
        "large_turbine_blade.stl": LARGE_BLADE.loft(),
        "spinner_cone.stl": spinner_cone(SPINNER_RADIUS, SPINNER_LENGTH),
        "torque_wrench.stl": torque_wrench(),
        "bolt_crate.stl": small_load_carrier(with_bolts=True),
        "empty_crate.stl": small_load_carrier(with_bolts=False),
    }
    for name, mesh in meshes.items():
        path = os.path.join(output_directory, name)
        mesh.export(path)
        extents = np.round(mesh.extents, 4)
        print("wrote %s: %d faces, extents=%s" % (path, len(mesh.faces), extents))
    write_textures(output_directory)
    write_floor_mesh(output_directory)
    write_wood_cube_mesh(output_directory)
    write_nacelle_mesh(output_directory)
    write_tower_mesh(output_directory)
    write_backdrop(os.path.dirname(__file__))
    print("wrote textures, textured meshes and the backdrop")


if __name__ == "__main__":
    main()
