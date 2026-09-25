"""Author textured timber workshop meshes and editable Blender reference views.

Run with Blender 4.0.2::

    blender --background --python cramera/tools/workshop/build_blender.py -- \
        --output cramera/src/cramera/resources/aicor_workshop \
        --source ~/.cramera/scenes/aicor_workshop
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from math import cos, pi, sin
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laboratory"))
from build_blender import LaboratoryBuilder, MaterialLibrary, MaterialName
from rights import LaboratoryAssetRights, PackagePath


# %% Authoring contract
class WorkshopAsset(StrEnum):
    """Independent bodies exported in metres with the Z axis pointing upward."""

    ENVIRONMENT = "environment"
    DRAWER = "drawer"
    BIN = "bin"
    SCREW = "screw"
    NAIL = "nail"
    ASSEMBLY = "assembly"


class Finish(StrEnum):
    """Wood and printed finishes supplementing the shared laboratory palette."""

    TIMBER = "Oiled European ash"
    PLYWOOD = "Birch plywood"
    ORANGE = "Safety orange elastomer"
    BRASS = "Brass screw finish"
    LOGO = "AICOR teal screen print"


@dataclass(frozen=True)
class WorkshopOptions:
    """Destination directories and render quality for workshop authoring."""

    output: Path
    """Resource directory receiving the exported rigid meshes."""
    source: Path
    """Directory receiving the editable source, material textures and previews."""
    samples: int = 192
    """Maximum Cycles samples per reference image."""
    width: int = 1600
    """Horizontal preview resolution in pixels."""
    render: bool = True
    """Whether to create both reference images."""


# %% Workshop materials and construction
@dataclass
class WorkshopBuilder(LaboratoryBuilder):
    """Build a woodworking bench with separate drawers and loose fasteners."""

    options: WorkshopOptions
    """Resource, source and rendering settings."""
    library: MaterialLibrary = field(default_factory=MaterialLibrary)
    """Shared physically based material palette."""
    current: WorkshopAsset = WorkshopAsset.ENVIRONMENT
    """Rigid body receiving new mesh components."""
    logo_image: bpy.types.Image | None = None
    """Packed original artwork recoloured with the workshop teal."""

    def timber_material(self, finish: Finish, seed: int, pale: bool = False) -> None:
        """Create packed albedo and roughness textures with continuous timber grain."""
        resolution = 1024
        random = np.random.default_rng(seed)
        horizontal, vertical = np.meshgrid(
            np.linspace(0, 1, resolution), np.linspace(0, 1, resolution)
        )
        warp = vertical + 0.012 * np.sin(horizontal * 11 + vertical * 5)
        warp += 0.004 * np.sin(horizontal * 27 + vertical * 13)
        grain = np.sin(
            warp * 480 + 7.0 * np.sin(warp * 35) + 1.8 * np.sin(horizontal * 5)
        )
        pores = np.maximum(0, np.sin(warp * 1070 + horizontal * 9)) ** 20
        figure = np.sin(warp * 34 + horizontal * 1.3)
        noise = random.normal(0, 0.007, horizontal.shape)
        knot = np.sqrt(
            ((horizontal - 0.68) / 0.32) ** 2 + ((vertical - 0.73) / 0.043) ** 2
        )
        knot_rings = np.sin(knot * 16) * np.exp(-knot * 1.1)
        tone = (
            0.018 * grain - 0.022 * pores + 0.028 * figure + 0.035 * knot_rings + noise
        )
        base = np.array((0.71, 0.55, 0.34) if not pale else (0.79, 0.68, 0.49))
        pixels = np.ones((resolution, resolution, 4), dtype=np.float32)
        pixels[:, :, :3] = np.clip(base + tone[:, :, None], 0, 1)
        colour = bpy.data.images.new(f"{finish.value} albedo", resolution, resolution)
        colour.pixels.foreach_set(pixels.ravel())
        colour.filepath_raw = str(
            self.options.source / f"{finish.name.lower()}-albedo.png"
        )
        colour.file_format = "PNG"
        colour.save()
        colour.pack()
        pixels[:, :, :3] = np.clip(0.43 + 0.10 * pores[:, :, None], 0, 1)
        roughness = bpy.data.images.new(
            f"{finish.value} roughness", resolution, resolution
        )
        roughness.colorspace_settings.name = "Non-Color"
        roughness.pixels.foreach_set(pixels.ravel())
        roughness.filepath_raw = str(
            self.options.source / f"{finish.name.lower()}-roughness.png"
        )
        roughness.file_format = "PNG"
        roughness.save()
        roughness.pack()
        material = self.library.create(finish, tuple(base), 0.43)
        shader = material.node_tree.nodes.get("Principled BSDF")
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = colour
        texture.extension = "REPEAT"
        material.node_tree.links.new(
            texture.outputs["Color"], shader.inputs["Base Color"]
        )
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = roughness
        texture.extension = "REPEAT"
        material.node_tree.links.new(
            texture.outputs["Color"], shader.inputs["Roughness"]
        )

    def prepare_materials(self) -> None:
        """Create wood textures and transparent, unboxed AICOR print artwork."""
        self.library.build()
        self.library.create(Finish.ORANGE, (0.73, 0.24, 0.035), 0.42)
        self.library.create(Finish.BRASS, (0.46, 0.32, 0.11), 0.3, 0.83)
        self.timber_material(Finish.TIMBER, 83)
        self.timber_material(Finish.PLYWOOD, 148, pale=True)
        artwork = (
            Path(__file__).resolve().parents[2] / "src/cramera/web/img/aicor-logo.png"
        )
        image = bpy.data.images.load(str(artwork), check_existing=False)
        pixels = np.array(image.pixels[:], dtype=np.float32).reshape((-1, 4))
        pixels[:, :3] = (0.025, 0.22, 0.23)
        image.pixels.foreach_set(pixels.ravel())
        image.filepath_raw = str(self.options.source / "aicor-teal.png")
        image.file_format = "PNG"
        image.save()
        image.pack()
        self.logo_image = image
        material = self.library.create(Finish.LOGO, (0.025, 0.22, 0.23), 0.55)
        material.blend_method = "CLIP"
        material.alpha_threshold = 0.1
        shader = material.node_tree.nodes.get("Principled BSDF")
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        material.node_tree.links.new(
            texture.outputs["Color"], shader.inputs["Base Color"]
        )
        material.node_tree.links.new(texture.outputs["Alpha"], shader.inputs["Alpha"])

    def wood(
        self,
        name: str,
        centre: tuple[float, float, float],
        size: tuple[float, float, float],
        pale: bool = False,
        bevel: float = 0.002,
    ) -> bpy.types.Object:
        """Give a rounded timber component measured, continuous planar grain UVs."""
        body = self.box(
            name, centre, size, Finish.PLYWOOD if pale else Finish.TIMBER, bevel
        )
        coordinates = body.data.uv_layers.active
        for polygon in body.data.polygons:
            axis = max(range(3), key=lambda index: abs(polygon.normal[index]))
            for index in polygon.loop_indices:
                vertex = body.data.vertices[body.data.loops[index].vertex_index].co
                if axis == 2:
                    coordinates.data[index].uv = (
                        vertex.x / 1.10 + 0.5,
                        vertex.y / 0.40 + 0.5,
                    )
                elif axis == 1:
                    coordinates.data[index].uv = (
                        vertex.x / 1.10 + 0.5,
                        vertex.z / 0.40 + 0.5,
                    )
                else:
                    coordinates.data[index].uv = (
                        vertex.y / 1.10 + 0.5,
                        vertex.z / 0.40 + 0.5,
                    )
        return body

    def print_logo(
        self, name: str, centre: tuple[float, float, float], width: float
    ) -> None:
        """Mount the transparent original artwork flush on a front-facing surface."""
        height = width * self.logo_image.size[1] / self.logo_image.size[0]
        x, y, z = centre
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(
            [
                (x - width / 2, y, z - height / 2),
                (x + width / 2, y, z - height / 2),
                (x + width / 2, y, z + height / 2),
                (x - width / 2, y, z + height / 2),
            ],
            [],
            [(0, 1, 2, 3)],
        )
        coordinates = mesh.uv_layers.new()
        for index, coordinate in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
            coordinates.data[index].uv = coordinate
        body = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(body)
        self.register(body, name, Finish.LOGO)

    def rod(
        self,
        name: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        radius: float,
        material: MaterialName | Finish,
    ) -> bpy.types.Object:
        """Place a cylindrical bar between two measured endpoints."""
        direction = Vector(end) - Vector(start)
        body = self.cylinder(
            name,
            tuple((Vector(start) + Vector(end)) / 2),
            radius,
            direction.length,
            material,
        )
        body.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
        return body

    def tube_curve(
        self,
        name: str,
        points: list[tuple[float, float, float]],
        radius: float,
        material: MaterialName | Finish,
    ) -> bpy.types.Object:
        """Create a continuous round section along the supplied centreline."""
        curve = bpy.data.curves.new(name, "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = radius
        curve.bevel_resolution = 2
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for point, coordinate in zip(spline.points, points):
            point.co = (*coordinate, 1)
        body = bpy.data.objects.new(name, curve)
        bpy.context.collection.objects.link(body)
        bpy.context.view_layer.objects.active = body
        body.select_set(True)
        bpy.ops.object.convert(target="MESH")
        return self.register(bpy.context.object, name, material)

    def open_box(
        self,
        name: str,
        centre: tuple[float, float, float],
        size: tuple[float, float, float],
        material: MaterialName | Finish,
        wall: float = 0.005,
    ) -> None:
        """Construct an open container with distinct floor and four thin walls."""
        x, y, z = centre
        width, depth, height = size
        self.box(
            name + " floor", (x, y, z + wall / 2), (width, depth, wall), material, 0.001
        )
        for side in (-1, 1):
            self.box(
                name + " side",
                (x + side * (width - wall) / 2, y, z + height / 2),
                (wall, depth, height),
                material,
                0.001,
            )
            self.box(
                name + " end",
                (x, y + side * (depth - wall) / 2, z + height / 2),
                (width - 2 * wall, wall, height),
                material,
                0.001,
            )

    def hex_head(
        self,
        name: str,
        centre: tuple[float, float, float],
        radius: float,
        height: float,
        material: MaterialName | Finish,
    ) -> None:
        """Create a beveled hexagonal fastener head with an inset drive slot."""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=6, radius=radius, depth=height, location=centre
        )
        body = self.register(bpy.context.object, name, material)
        modifier = body.modifiers.new("Rounded head edges", "BEVEL")
        modifier.width = 0.0006
        modifier.segments = 2
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        self.box(
            name + " driver slot",
            (centre[0], centre[1], centre[2] + height / 2 + 0.00005),
            (radius * 1.2, 0.002, 0.00015),
            MaterialName.DARK,
            0,
        )

    def drill(self, position: tuple[float, float, float]) -> None:
        """Construct a cordless driver with motor casing, trigger and removable-style battery."""
        x, y, z = position
        self.box(
            "Driver battery shoe",
            (x, y, z + 0.023),
            (0.15, 0.09, 0.046),
            MaterialName.DARK,
            0.008,
        )
        self.box(
            "Driver battery teal inset",
            (x, y - 0.046, z + 0.026),
            (0.106, 0.003, 0.026),
            MaterialName.TEAL,
            0.003,
        )
        self.text(
            "Battery capacity",
            "18 V",
            (x - 0.036, y - 0.048, z + 0.018),
            0.017,
            MaterialName.LABEL,
        )
        grip = self.box(
            "Driver soft grip",
            (x - 0.012, y, z + 0.107),
            (0.052, 0.059, 0.135),
            MaterialName.DARK,
            0.012,
        )
        grip.rotation_euler.y = -0.16
        self.box(
            "Driver grip accent",
            (x - 0.032, y - 0.031, z + 0.109),
            (0.027, 0.006, 0.092),
            MaterialName.TEAL,
            0.006,
        )
        self.box(
            "Driver trigger",
            (x + 0.024, y, z + 0.155),
            (0.018, 0.038, 0.029),
            Finish.ORANGE,
            0.005,
        )
        self.rod(
            "Driver motor",
            (x - 0.10, y, z + 0.216),
            (x + 0.05, y, z + 0.216),
            0.053,
            MaterialName.TEAL,
        )
        self.rod(
            "Driver rear bumper",
            (x - 0.105, y, z + 0.216),
            (x - 0.079, y, z + 0.216),
            0.054,
            MaterialName.DARK,
        )
        self.rod(
            "Driver torque collar",
            (x + 0.046, y, z + 0.216),
            (x + 0.085, y, z + 0.216),
            0.043,
            MaterialName.DARK,
        )
        self.rod(
            "Driver steel chuck",
            (x + 0.08, y, z + 0.216),
            (x + 0.12, y, z + 0.216),
            0.032,
            MaterialName.STEEL,
        )
        self.rod(
            "Driver mounted bit",
            (x + 0.12, y, z + 0.216),
            (x + 0.184, y, z + 0.216),
            0.005,
            MaterialName.STEEL,
        )
        for offset in range(5):
            self.box(
                "Motor cooling vent",
                (x - 0.058 + offset * 0.013, y - 0.0505, z + 0.226),
                (0.004, 0.003, 0.025),
                MaterialName.DARK,
                0.001,
            )
        self.print_logo("Driver AICOR", (x - 0.004, y - 0.0532, z + 0.197), 0.055)

    def tools(self) -> None:
        """Arrange a timber tool board, bit caddy and familiar hand tools."""
        self.wood(
            "Birch tool wall",
            (-0.06, 0.386, 1.44),
            (1.90, 0.026, 0.89),
            pale=True,
            bevel=0.005,
        )
        for x in np.arange(-0.94, 0.89, 0.075):
            for z in np.arange(1.08, 1.78, 0.075):
                body = self.cylinder(
                    "Pegboard socket",
                    (float(x), 0.3718, float(z)),
                    0.0032,
                    0.0008,
                    MaterialName.DARK,
                )
                body.rotation_euler.x = pi / 2
        self.print_logo("Tool wall AICOR", (-0.66, 0.3712, 1.78), 0.37)
        self.text(
            "Tool wall subtitle",
            "WOODWORK / ASSEMBLY",
            (-0.43, 0.3707, 1.764),
            0.027,
            MaterialName.TEAL,
        )
        self.wood(
            "Tool caddy rail", (-0.57, 0.292, 1.19), (0.64, 0.12, 0.071), pale=True
        )
        self.print_logo("Holder AICOR", (-0.78, 0.2308, 1.185), 0.13)
        for index in range(5):
            x = -0.76 + index * 0.105
            self.rod(
                "Hanging screwdriver steel",
                (x, 0.288, 1.18),
                (x, 0.288, 1.48),
                0.004,
                MaterialName.STEEL,
            )
            self.cylinder(
                "Screwdriver soft handle",
                (x, 0.288, 1.47),
                0.016,
                0.098,
                MaterialName.TEAL,
            )
            self.cylinder(
                "Screwdriver cap", (x, 0.288, 1.518), 0.017, 0.009, MaterialName.DARK
            )
            for side in (-1, 1):
                self.box(
                    "Screwdriver rubber grip",
                    (x + side * 0.012, 0.281, 1.47),
                    (0.006, 0.021, 0.063),
                    MaterialName.DARK,
                    0.003,
                )
        self.wood(
            "Drill bit caddy", (0.10, 0.205, 0.945), (0.40, 0.13, 0.05), pale=True
        )
        self.print_logo("Bit holder AICOR", (0.10, 0.1398, 0.946), 0.11)
        for index in range(8):
            x = -0.063 + index * 0.044
            radius = 0.002 + index * 0.0008
            length = 0.08 + index * 0.012
            self.cylinder(
                "Drill bit",
                (x, 0.215, 0.960 + length / 2),
                radius,
                length,
                MaterialName.STEEL,
            )
            points = [
                (
                    x + radius * cos(angle),
                    0.215 + radius * sin(angle),
                    0.966 + (length - 0.013) * step / 90,
                )
                for step in range(91)
                for angle in (step * pi / 8,)
            ]
            self.tube_curve("Bit flute", points, 0.00032, MaterialName.DARK)
            bpy.ops.mesh.primitive_cone_add(
                vertices=32,
                radius1=radius,
                radius2=0,
                depth=radius * 1.25,
                location=(x, 0.215, 0.960 + length + radius * 0.625),
            )
            self.register(bpy.context.object, "Drill cutting tip", MaterialName.STEEL)
        self.drill((0.64, 0.20, 0.925))
        self.wood(
            "Driver docking base", (0.64, 0.19, 0.928), (0.24, 0.16, 0.016), pale=True
        )
        self.print_logo("Dock AICOR", (0.65, 0.1098, 0.929), 0.10)
        # The saw plate and hammer remain fixed exhibits on the tool wall.
        self.box(
            "Saw steel blade",
            (0.39, 0.357, 1.42),
            (0.40, 0.003, 0.094),
            MaterialName.STEEL,
            0.001,
        )
        for index in range(26):
            x = 0.207 + index * 0.014
            self.rod(
                "Saw tooth",
                (x, 0.355, 1.374),
                (x + 0.007, 0.355, 1.361),
                0.002,
                MaterialName.STEEL,
            )
        self.wood(
            "Saw wooden grip",
            (0.64, 0.346, 1.423),
            (0.11, 0.025, 0.075),
            pale=False,
            bevel=0.013,
        )
        self.print_logo("Saw AICOR", (0.637, 0.331, 1.426), 0.075)
        self.wood(
            "Hammer ash handle", (0.79, 0.339, 1.20), (0.028, 0.029, 0.30), bevel=0.010
        )
        self.box(
            "Hammer head",
            (0.79, 0.339, 1.36),
            (0.14, 0.053, 0.054),
            MaterialName.STEEL,
            0.006,
        )
        self.box(
            "Hammer striking face",
            (0.866, 0.339, 1.36),
            (0.012, 0.059, 0.062),
            MaterialName.DARK,
            0.003,
        )
        for z in (1.20, 1.61):
            self.rod(
                "Tool hanger pin",
                (0.79, 0.370, z),
                (0.79, 0.323, z),
                0.004,
                MaterialName.STEEL,
            )
        self.box(
            "Bench ruler",
            (0.63, -0.296, 0.924),
            (0.48, 0.026, 0.003),
            MaterialName.STEEL,
            0.001,
        )
        for index in range(45):
            x = 0.408 + index * 0.010
            self.box(
                "Ruler engraving",
                (x, -0.291, 0.9258),
                (0.0007, 0.012 if index % 5 else 0.020, 0.00012),
                MaterialName.DARK,
                0,
            )

    def environment(self) -> None:
        """Build a light woodworking room, workbench and three-drawer cabinet shell."""
        self.current = WorkshopAsset.ENVIRONMENT
        self.box(
            "Workshop floor",
            (0, -0.05, -0.065),
            (4.7, 4.4, 0.13),
            MaterialName.FLOOR,
            0.015,
        )
        self.box(
            "Back plaster wall",
            (0, 1.00, 1.40),
            (4.7, 0.12, 2.8),
            MaterialName.WALL,
            0.002,
        )
        self.box(
            "Left plaster wall",
            (-2.31, 0.14, 1.40),
            (0.12, 1.75, 2.8),
            MaterialName.WALL,
            0.002,
        )
        self.box(
            "Back skirting",
            (0, 0.929, 0.07),
            (4.62, 0.026, 0.14),
            MaterialName.WHITE,
            0.002,
        )
        self.box(
            "Wall batten left",
            (-1.90, 0.922, 1.36),
            (0.018, 0.025, 2.48),
            MaterialName.WHITE,
            0.002,
        )
        self.box(
            "Wall batten right",
            (1.98, 0.922, 1.36),
            (0.018, 0.025, 2.48),
            MaterialName.WHITE,
            0.002,
        )
        self.print_logo("Room AICOR", (-1.26, 0.9288, 2.17), 0.66)
        self.text(
            "Workshop room title",
            "WOODWORK STUDIO",
            (-0.75, 0.9278, 2.123),
            0.062,
            MaterialName.TEAL,
        )
        self.text(
            "Workshop room descriptor",
            "A S S E M B L Y   /   P R O T O T Y P I N G",
            (-0.76, 0.9278, 2.025),
            0.022,
            MaterialName.TEAL,
        )
        self.wood("Solid ash bench top", (0, 0, 0.8875), (2.2, 0.8, 0.065), bevel=0.008)
        for x in (-0.95, 0.95):
            for y in (-0.30, 0.30):
                self.box(
                    "Bench steel leg",
                    (x, y, 0.433),
                    (0.065, 0.065, 0.855),
                    MaterialName.TEAL,
                    0.006,
                )
                self.box(
                    "Bench leveling pad",
                    (x, y, 0.013),
                    (0.083, 0.083, 0.026),
                    MaterialName.DARK,
                    0.007,
                )
            self.box(
                "Bench lower side brace",
                (x, 0, 0.15),
                (0.046, 0.60, 0.046),
                MaterialName.TEAL,
                0.003,
            )
        for y in (-0.305, 0.305):
            self.box(
                "Bench apron",
                (0, y, 0.81),
                (1.94, 0.046, 0.075),
                MaterialName.TEAL,
                0.004,
            )
        self.wood(
            "Lower storage shelf", (0.36, 0, 0.183), (1.06, 0.57, 0.028), pale=True
        )
        for index in range(5):
            self.wood(
                "Stored timber offcut",
                (0.41, 0.04, 0.222 + index * 0.042),
                (0.81 - index * 0.078, 0.26, 0.037),
                pale=index % 2 == 0,
            )
        self.print_logo("Bench apron AICOR", (0.58, -0.3288, 0.810), 0.22)
        self.print_logo("Bench ash AICOR", (-0.82, -0.4002, 0.886), 0.27)
        self.text(
            "Bench serial",
            "01 / WOODWORK",
            (0.58, -0.4003, 0.878),
            0.019,
            MaterialName.TEAL,
        )
        for x in (-1.044, -0.516):
            self.box(
                "Drawer cabinet side",
                (x, 0.009, 0.497),
                (0.03, 0.64, 0.69),
                MaterialName.WHITE,
                0.004,
            )
        self.box(
            "Drawer cabinet back",
            (-0.78, 0.315, 0.497),
            (0.54, 0.025, 0.69),
            MaterialName.WHITE,
            0.003,
        )
        self.box(
            "Drawer cabinet plinth",
            (-0.78, 0, 0.17),
            (0.55, 0.64, 0.034),
            MaterialName.TEAL,
            0.004,
        )
        for z in (0.29, 0.48, 0.67, 0.83):
            self.box(
                "Drawer cabinet cross rail",
                (-0.78, -0.304, z),
                (0.50, 0.023, 0.018),
                MaterialName.TEAL,
                0.002,
            )
        for z in (0.31, 0.50, 0.69):
            for x in (-1.014, -0.546):
                self.box(
                    "Drawer slide rail",
                    (x, 0.015, z),
                    (0.015, 0.50, 0.018),
                    MaterialName.STEEL,
                    0.001,
                )
        self.tools()
        self.shelves()
        self.vise()

    def shelves(self) -> None:
        """Build a separate shelving unit with timber shelves and open storage boxes."""
        x, y = 1.48, 0.15
        for dx in (-0.262, 0.262):
            for dy in (-0.28, 0.28):
                self.box(
                    "Shelving upright",
                    (x + dx, y + dy, 0.825),
                    (0.036, 0.036, 1.65),
                    MaterialName.TEAL,
                    0.003,
                )
                self.box(
                    "Shelving foot",
                    (x + dx, y + dy, 0.015),
                    (0.054, 0.054, 0.030),
                    MaterialName.DARK,
                    0.004,
                )
        for z in (0.13, 0.53, 0.96, 1.55):
            self.wood("Shelving birch shelf", (x, y, z), (0.55, 0.60, 0.025), pale=True)
            self.box(
                "Shelving front rail",
                (x, y - 0.292, z - 0.026),
                (0.54, 0.017, 0.042),
                MaterialName.WHITE,
                0.002,
            )
        self.print_logo("Shelf crown AICOR", (x, 0.15 - 0.301, 1.541), 0.25)
        for level, z in enumerate((0.144, 0.544)):
            for side in (-1, 1):
                centre = (x + side * 0.137, y - 0.026, z)
                self.open_box(
                    "Shelf parts crate",
                    centre,
                    (0.24, 0.44, 0.23),
                    MaterialName.PALE_TEAL,
                    0.010,
                )
                self.print_logo(
                    "Storage box AICOR",
                    (centre[0], centre[1] - 0.2202, z + 0.139),
                    0.135,
                )
                self.text(
                    "Storage box description",
                    "TIMBER" if level else "FIXINGS",
                    (centre[0] - 0.042, centre[1] - 0.2204, z + 0.085),
                    0.016,
                    MaterialName.TEAL,
                )
                for offset in (-0.062, 0.062):
                    self.box(
                        "Crate handle opening",
                        (centre[0] + offset, centre[1] - 0.226, z + 0.198),
                        (0.055, 0.002, 0.013),
                        MaterialName.DARK,
                        0.004,
                    )
        for index in range(4):
            self.wood(
                "Shelf board stack",
                (x - 0.03, y, 0.998 + index * 0.055),
                (0.43, 0.42, 0.042),
                pale=index % 2 == 0,
            )
        self.wood(
            "Shelf upright offcut",
            (x + 0.19, y + 0.07, 1.255),
            (0.045, 0.28, 0.48),
            pale=False,
        )
        self.box(
            "Workshop wall task light",
            (0, 0.29, 1.947),
            (1.65, 0.11, 0.033),
            MaterialName.TEAL,
            0.004,
        )
        self.box(
            "Task light diffuser",
            (0, 0.288, 1.928),
            (1.57, 0.084, 0.006),
            MaterialName.LIGHT,
            0.002,
        )

    def vise(self) -> None:
        """Add a compact woodworking vise at the right front bench corner."""
        self.box(
            "Vise fixed casting",
            (0.90, -0.19, 0.965),
            (0.14, 0.19, 0.071),
            MaterialName.TEAL,
            0.008,
        )
        self.wood(
            "Vise fixed wooden jaw",
            (0.90, -0.306, 0.977),
            (0.185, 0.024, 0.081),
            pale=True,
        )
        self.wood(
            "Vise moving wooden jaw",
            (0.90, -0.396, 0.977),
            (0.185, 0.030, 0.081),
            pale=True,
        )
        self.rod(
            "Vise leadscrew",
            (0.90, -0.28, 0.956),
            (0.90, -0.452, 0.956),
            0.009,
            MaterialName.STEEL,
        )
        self.rod(
            "Vise cross handle",
            (0.829, -0.466, 0.956),
            (0.971, -0.466, 0.956),
            0.006,
            MaterialName.STEEL,
        )
        self.print_logo("Vise AICOR", (0.90, -0.4113, 0.986), 0.105)

    def drawer(self) -> None:
        """Build the drawer cavity and a raised pull with a consistent grasp frame."""
        self.current = WorkshopAsset.DRAWER
        self.open_box(
            "Drawer birch", (0, 0, 0), (0.48, 0.56, 0.14), Finish.PLYWOOD, 0.009
        )
        self.box(
            "Drawer enamel fascia",
            (0, -0.285, 0.076),
            (0.48, 0.020, 0.148),
            MaterialName.WHITE,
            0.004,
        )
        self.wood(
            "Drawer timber accent",
            (0, -0.297, 0.027),
            (0.464, 0.003, 0.031),
            pale=True,
            bevel=0.001,
        )
        self.print_logo("Drawer AICOR", (-0.157, -0.2953, 0.103), 0.105)
        for x in (-0.065, 0.065):
            self.box(
                "Pull mounting foot",
                (x, -0.307, 0.085),
                (0.016, 0.028, 0.018),
                MaterialName.TEAL,
                0.004,
            )
        self.box(
            "Drawer pull",
            (0, -0.318, 0.085),
            (0.158, 0.016, 0.021),
            MaterialName.TEAL,
            0.006,
        )
        self.box(
            "Drawer felt lining",
            (0, 0, 0.010),
            (0.455, 0.535, 0.002),
            MaterialName.DARK,
            0.002,
        )

    def bin(self) -> None:
        """Make a shallow fastener container with a genuine open cavity."""
        self.current = WorkshopAsset.BIN
        self.open_box(
            "Fastener bin",
            (0, 0, 0),
            (0.28, 0.32, 0.075),
            MaterialName.PALE_TEAL,
            0.006,
        )
        self.print_logo("Fastener bin AICOR", (0, -0.1603, 0.039), 0.105)
        for x in (-0.115, 0.115):
            self.box(
                "Bin side grip",
                (x, -0.164, 0.060),
                (0.025, 0.006, 0.011),
                MaterialName.TEAL,
                0.002,
            )

    def screw(self) -> None:
        """Create a large visible screw thread around a solid shank and hex head."""
        self.current = WorkshopAsset.SCREW
        self.cylinder("Screw shank", (0, 0, 0.061), 0.0097, 0.098, Finish.BRASS)
        bpy.ops.mesh.primitive_cone_add(
            vertices=64,
            radius1=0.001,
            radius2=0.0097,
            depth=0.018,
            location=(0, 0, 0.009),
        )
        self.register(bpy.context.object, "Screw pilot tip", Finish.BRASS)
        turns = 10
        points = []
        for index in range(721):
            angle = index / 720 * turns * 2 * pi
            height = 0.013 + index / 720 * 0.086
            points.append((0.01025 * cos(angle), 0.01025 * sin(angle), height))
        self.tube_curve("Continuous screw thread", points, 0.00175, Finish.BRASS)
        self.cylinder("Screw head washer", (0, 0, 0.109), 0.022, 0.004, Finish.BRASS)
        self.hex_head("Screw hex head", (0, 0, 0.119), 0.026, 0.018, Finish.BRASS)

    def nail(self) -> None:
        """Create a separate steel nail with a sharp tip and flattened head."""
        self.current = WorkshopAsset.NAIL
        self.cylinder("Nail shank", (0, 0, 0.042), 0.0035, 0.066, MaterialName.STEEL)
        bpy.ops.mesh.primitive_cone_add(
            vertices=32, radius1=0, radius2=0.0035, depth=0.009, location=(0, 0, 0.0045)
        )
        self.register(bpy.context.object, "Nail point", MaterialName.STEEL)
        self.cylinder("Nail head", (0, 0, 0.078), 0.009, 0.006, MaterialName.STEEL)

    def assembly(self) -> None:
        """Model a timber assembly block with a real circular pilot opening."""
        self.current = WorkshopAsset.ASSEMBLY
        body = self.wood(
            "Pilot-hole timber block", (0, 0, 0.0375), (0.25, 0.20, 0.075), bevel=0.003
        )
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=96, radius=0.014, depth=0.12, location=(0, 0, 0.0375)
        )
        cutter = bpy.context.object
        bpy.context.view_layer.objects.active = body
        modifier = body.modifiers.new("Through pilot hole", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
        self.print_logo("Assembly AICOR", (0, -0.1002, 0.042), 0.13)
        self.text(
            "Assembly pilot legend",
            "PILOT 28",
            (-0.096, -0.1003, 0.012),
            0.011,
            MaterialName.TEAL,
        )
        for x in (-0.109, 0.109):
            self.box(
                "Block clamping strip",
                (x, 0, 0.002),
                (0.020, 0.235, 0.004),
                MaterialName.TEAL,
                0.002,
            )
            for y in (-0.108, 0.108):
                self.cylinder(
                    "Strip mounting hole",
                    (x, y, 0.0041),
                    0.0027,
                    0.0003,
                    MaterialName.DARK,
                )

    def export_asset(
        self, asset: WorkshopAsset, placement: tuple[float, float, float]
    ) -> None:
        """Batch stationary meshes by material while retaining editable source parts."""
        if asset != WorkshopAsset.ENVIRONMENT:
            super().export_asset(asset, placement)
            return
        originals = self.objects[asset]
        bpy.ops.object.select_all(action="DESELECT")
        copies = []
        for original in originals:
            duplicate = original.copy()
            duplicate.data = original.data.copy()
            bpy.context.collection.objects.link(duplicate)
            duplicate.select_set(True)
            copies.append(duplicate)
        bpy.context.view_layer.objects.active = copies[0]
        bpy.ops.object.join()
        merged = bpy.context.object
        merged.name = "Workshop stationary surfaces"
        self.objects[asset] = [merged]
        super().export_asset(asset, placement)
        parent = merged.parent
        mesh = merged.data
        bpy.data.objects.remove(merged, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        self.objects[asset] = originals
        for original in originals:
            original.parent = parent

    def pose_assets(self) -> None:
        """Arrange exported objects for the editable source and photographic previews."""
        self.export_asset(WorkshopAsset.ENVIRONMENT, (0, 0, 0))
        self.export_asset(WorkshopAsset.DRAWER, (-0.78, -0.14, 0.68))
        bpy.data.objects[WorkshopAsset.DRAWER].constraints[
            "Prismatic drawer travel"
        ].min_y = -0.40
        for index, z in enumerate((0.30, 0.49)):
            body = bpy.data.objects.new(f"Drawer {index+1}", None)
            bpy.context.collection.objects.link(body)
            body.location = (-0.78, 0, z)
            for original in self.objects[WorkshopAsset.DRAWER]:
                instance = original.copy()
                bpy.context.collection.objects.link(instance)
                instance.parent = body
        self.export_asset(WorkshopAsset.BIN, (-0.25, -0.10, 0.92))
        body = bpy.data.objects.new("Nail bin", None)
        bpy.context.collection.objects.link(body)
        body.location = (-0.72, -0.10, 0.92)
        for original in self.objects[WorkshopAsset.BIN]:
            instance = original.copy()
            bpy.context.collection.objects.link(instance)
            instance.parent = body
        self.export_asset(WorkshopAsset.ASSEMBLY, (0.28, -0.10, 0.92))
        self.export_asset(WorkshopAsset.SCREW, (-0.28, -0.12, 0.935))
        bpy.data.objects[WorkshopAsset.SCREW].rotation_euler = (0, pi / 2, 0.28)
        self.export_asset(WorkshopAsset.NAIL, (-0.77, -0.14, 0.929))
        bpy.data.objects[WorkshopAsset.NAIL].rotation_euler = (0, pi / 2, 0.48)
        random = np.random.default_rng(84)
        for asset, count, centre in (
            (WorkshopAsset.SCREW, 5, (-0.28, -0.10, 0.943)),
            (WorkshopAsset.NAIL, 16, (-0.76, -0.10, 0.934)),
        ):
            for index in range(count):
                parent = bpy.data.objects.new(f"{asset.value} preview {index}", None)
                bpy.context.collection.objects.link(parent)
                parent.location = (
                    centre[0] + random.uniform(-0.055, 0.045),
                    centre[1] + random.uniform(-0.063, 0.063),
                    centre[2] + index * 0.0015,
                )
                parent.rotation_euler = (0, pi / 2, random.uniform(-0.65, 0.65))
                for original in self.objects[asset]:
                    instance = original.copy()
                    bpy.context.collection.objects.link(instance)
                    instance.parent = parent

    def render_views(self) -> None:
        """Render a room overview and a detailed assembly bench photograph."""
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "CUDA"
        preferences.get_devices()
        if any(device.type == "CUDA" for device in preferences.devices):
            for device in preferences.devices:
                device.use = device.type == "CUDA"
            scene.cycles.device = "GPU"
        scene.cycles.samples = self.options.samples
        scene.cycles.use_denoising = False
        scene.cycles.adaptive_threshold = 0.02
        scene.render.resolution_x = self.options.width
        scene.render.resolution_y = round(self.options.width * 0.70)
        scene.render.resolution_percentage = 100
        scene.world.use_nodes = True
        background = scene.world.node_tree.nodes["Background"]
        background.inputs["Color"].default_value = (0.78, 0.86, 1, 1)
        background.inputs["Strength"].default_value = 0.28
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -1.15
        self.area_light(
            "Large studio window",
            (-1.7, -1.0, 3.1),
            (0, 0.05, 1),
            480,
            2.4,
            (1, 0.95, 0.87),
        )
        self.area_light(
            "Soft right fill",
            (2.1, -0.5, 2.5),
            (0.4, 0, 0.9),
            320,
            2.0,
            (0.84, 0.93, 1),
        )
        self.area_light(
            "Bench task illumination",
            (0, 0.25, 1.89),
            (0, -0.1, 0.92),
            38,
            1.3,
            (1, 0.88, 0.69),
        )
        self.area_light("Front softbox", (0.4, -2.4, 1.7), (0, 0.1, 1), 180, 2.0)
        overview = self.camera(
            "Workshop overview", (3.15, -4.15, 2.75), (0.20, 0.05, 1.04), 47, 11
        )
        detail = self.camera(
            "Workshop joinery detail",
            (0.97, -1.14, 1.66),
            (-0.02, -0.055, 1.025),
            54,
            11,
        )
        scene.camera = overview
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(
            filepath=str(self.options.source / "workshop.blend")
        )
        if self.options.render:
            for camera, filename in (
                (overview, "overview.png"),
                (detail, "closeup.png"),
            ):
                scene.camera = camera
                scene.render.filepath = str(self.options.source / filename)
                bpy.ops.render.render(write_still=True)

    def build(self) -> None:
        """Create six independent GLB resources and the editable Blender workshop."""
        self.options.output.joinpath("assets").mkdir(parents=True, exist_ok=True)
        self.options.source.mkdir(parents=True, exist_ok=True)
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        self.prepare_materials()
        self.environment()
        self.drawer()
        self.bin()
        self.screw()
        self.nail()
        self.assembly()
        self.pose_assets()
        LaboratoryAssetRights(
            source=Path(__file__).with_name(PackagePath.CANONICAL)
        ).install(self.options.output)
        (self.options.source / "asset-measurements.json").write_text(
            json.dumps(self.measurements, indent=2) + "\n"
        )
        self.render_views()


# %% Command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--no-render", action="store_true")
    arguments = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    WorkshopBuilder(
        WorkshopOptions(
            arguments.output.resolve(),
            arguments.source.expanduser().resolve(),
            arguments.samples,
            arguments.width,
            not arguments.no_render,
        )
    ).build()
