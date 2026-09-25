"""Build the AICOR medical laboratory and its independently movable GLB assets.

Run with Blender 4.0.2::

    blender --background --python cramera/tools/laboratory/build_blender.py -- \
        --output /home/hassouna/.cramera/scenes/aicor_medical_lab --samples 160
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from math import cos, pi, sin
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import LatheMesh, ProfilePoint, RackDimensions, TubeDimensions
from centrifuge import CentrifugeDimensions, CentrifugePart
from pipetting import PipettingChannel, PipettingDimensions, PipettingPart
from refrigerator import RefrigeratorDimensions, RefrigeratorPart
from rights import LaboratoryAssetRights


# %% Asset contract
class AssetName(StrEnum):
    """Files loaded as separate rigid bodies by the laboratory scene."""

    ENVIRONMENT = "environment"
    DRAWER = "drawer"
    TUBE_RACK = "tube_rack"
    """Independent carrier with six sockets and a front grasp bar."""
    TUBE_CLEAR = "tube_clear"
    TUBE_AMBER = "tube_amber"
    TUBE_TEAL = "tube_teal"
    TUBE_RESERVE = "tube_reserve"
    """Independent spare body sharing the clear glass visual geometry."""
    STOPPER = "stopper"
    CENTRIFUGE_BASE = CentrifugePart.BASE_ASSET.value
    CENTRIFUGE_ROTOR = CentrifugePart.ROTOR_ASSET.value
    CENTRIFUGE_LID = CentrifugePart.LID_ASSET.value
    PIPETTING = PipettingPart.ASSET.value
    """Stationary dual-feed dispensing instrument."""
    REFRIGERATOR_BASE = RefrigeratorPart.BASE_ASSET.value
    """Insulated floor-standing cabinet with sample shelves."""
    REFRIGERATOR_DOOR = RefrigeratorPart.DOOR_ASSET.value
    """Independently hinged glazed door."""


class RenderView(StrEnum):
    """Reference cameras available for selective rerendering."""

    ALL = "all"
    OVERVIEW = "overview"
    CLOSEUP = "closeup"


class BrandingName(StrEnum):
    """Existing artwork and retired decorative mesh names."""

    IMAGE = "aicor-logo.png"
    MATERIAL = "AICOR logo artwork"
    LOGO = "AICOR logo"
    PLAQUE = "AICOR plaque"
    ROOM_TITLE = "Laboratory title"
    INSTRUMENT_TITLE = "Pipetting brand"
    TRAY = "Work tray"
    PIPETTE = "Pipette body"
    WALL = "Wall AICOR logo"
    BALANCE = "Balance AICOR logo"
    RACK = "Rack AICOR logo"
    CENTRIFUGE = "Centrifuge AICOR logo"
    DISPENSER = "Dispenser AICOR logo"
    CHAMBER = "Centrifuge chamber shell"
    REFRIGERATOR = "Refrigerator AICOR logo"
    DRAWER_MAT = "Drawer mat"
    """Decorative mat removed to expose the sample carrier support floor."""
    DRAWER_DIVIDER = "Tray divider"
    """Old tray separators occupying the stored rack footprint."""


@dataclass(frozen=True)
class LogoPlacement:
    """Artwork dimensions and mounting surface in an instrument's frame."""

    name: BrandingName
    """Stable scene name of the artwork mesh."""
    center: tuple[float, float, float]
    """Center of the front-facing artwork before surface projection."""
    width: float
    """Printed artwork width in metres."""
    surface: BrandingName | None = None
    """Curved housing onto which the artwork is projected."""
    segments: int = 48
    """Horizontal subdivisions preserving the housing curvature."""
    clearance: float = 0.0002
    """Outward offset preventing coincident surface rendering."""

    @classmethod
    def for_asset(cls, asset: AssetName) -> tuple[LogoPlacement, ...]:
        """Place each logo in an unobstructed area of its instrument."""
        return {
            AssetName.ENVIRONMENT: (
                cls(BrandingName.WALL, (-0.40, 1.0995, 2.04), 0.90),
                cls(BrandingName.BALANCE, (0.750, 0.0797, 0.940), 0.058),
            ),
            AssetName.TUBE_RACK: (cls(BrandingName.RACK, (0, -0.0552, 0.003), 0.010),),
            AssetName.DRAWER: (),
            AssetName.CENTRIFUGE_BASE: (
                cls(
                    BrandingName.CENTRIFUGE,
                    (0, -0.17, 0.173),
                    0.086,
                    BrandingName.CHAMBER,
                ),
            ),
            AssetName.PIPETTING: (
                cls(BrandingName.DISPENSER, (-0.063, 0.0391, 0.227), 0.065),
            ),
            AssetName.REFRIGERATOR_BASE: (
                cls(BrandingName.REFRIGERATOR, (-0.118, -0.2902, 0.791), 0.14),
            ),
        }[asset]


class LogoProjectionError(RuntimeError):
    """The proposed artwork extends beyond its supporting housing."""


class MaterialName(StrEnum):
    """Shared finishes in the authored laboratory."""

    WHITE = "Warm white enamel"
    TEAL = "Petrol powder coat"
    PALE_TEAL = "Pale teal powder coat"
    STEEL = "Satin stainless steel"
    DARK = "Graphite elastomer"
    TOP = "Porcelain work surface"
    FLOOR = "Warm grey flooring"
    WALL = "Ivory wall"
    GLASS = "Borosilicate glass"
    AMBER = "Amber solution"
    LIQUID_TEAL = "Teal solution"
    LABEL = "Ceramic graduation ink"
    PAPER = "Matte white adhesive label"
    """Opaque uncoated paper for reservoir identification."""
    TEAL_PAPER = "Matte teal adhesive label"
    """Tinted uncoated paper for stored reagents."""
    LIGHT = "Diffuser"


@dataclass(frozen=True)
class BuildOptions:
    """Output and render quality controls for reproducible generation."""

    output: Path
    """Directory containing the scene and its assets."""
    samples: int = 160
    """Maximum Cycles samples for each reference image."""
    width: int = 1600
    """Horizontal output resolution."""
    render: bool = True
    """Whether to render the two authored cameras."""
    view: RenderView = RenderView.ALL
    """Reference view to render after asset generation."""
    centrifuge_only: bool = False
    """Export only the centrifuge while preserving existing laboratory assets."""
    pipetting_only: bool = False
    """Export only the dispensing station and preserve existing laboratory assets."""
    branding_only: bool = False
    """Refresh existing instrument artwork without rebuilding bodies."""
    refrigerator_only: bool = False
    """Add or replace the cold-storage cabinet while preserving the room."""


@dataclass
class MaterialLibrary:
    """Exportable Principled BSDF finishes with shared instances."""

    materials: dict[MaterialName, bpy.types.Material] = field(default_factory=dict)
    """Materials indexed by their visible finish name."""

    def create(
        self,
        name: MaterialName,
        colour: tuple[float, float, float],
        roughness: float,
        metalness: float = 0,
        transmission: float = 0,
        index_of_refraction: float = 1.5,
    ) -> bpy.types.Material:
        """Create a physically based finish suitable for GLB and Cycles."""
        material = bpy.data.materials.new(name.value)
        material.use_nodes = True
        material.diffuse_color = (*colour, 1)
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*colour, 1)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metalness
        shader.inputs["Transmission Weight"].default_value = transmission
        shader.inputs["IOR"].default_value = index_of_refraction
        self.materials[name] = material
        return material

    def build(self) -> None:
        """Create the laboratory palette and optical materials."""
        self.create(MaterialName.WHITE, (0.78, 0.83, 0.82), 0.27)
        self.create(MaterialName.TEAL, (0.027, 0.145, 0.154), 0.28, 0.18)
        self.create(MaterialName.PALE_TEAL, (0.33, 0.54, 0.52), 0.28, 0.15)
        self.create(MaterialName.STEEL, (0.46, 0.52, 0.53), 0.26, 0.9)
        self.create(MaterialName.DARK, (0.017, 0.025, 0.027), 0.44)
        self.create(MaterialName.TOP, (0.80, 0.85, 0.83), 0.23)
        self.create(MaterialName.FLOOR, (0.44, 0.47, 0.45), 0.63)
        self.create(MaterialName.WALL, (0.67, 0.73, 0.70), 0.73)
        self.create(MaterialName.GLASS, (0.94, 0.985, 0.99), 0.045, 0, 1, 1.47)
        self.create(MaterialName.AMBER, (0.82, 0.32, 0.055), 0.045, 0, 0.92, 1.333)
        self.create(MaterialName.LIQUID_TEAL, (0.12, 0.56, 0.52), 0.04, 0, 0.93, 1.333)
        self.create(MaterialName.LABEL, (0.63, 0.70, 0.69), 0.45)
        self.create(MaterialName.PAPER, (0.84, 0.87, 0.84), 0.78)
        self.create(MaterialName.TEAL_PAPER, (0.33, 0.54, 0.52), 0.78)
        diffuser = self.create(MaterialName.LIGHT, (0.94, 0.98, 1), 0.28)
        shader = diffuser.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Emission Color"].default_value = (0.75, 0.9, 1, 1)
        shader.inputs["Emission Strength"].default_value = 1.5


# %% Modelling operations
@dataclass
class LaboratoryBuilder:
    """Construct measured meshes, exported rigid bodies and reference cameras."""

    options: BuildOptions
    """Generation and render settings."""
    library: MaterialLibrary = field(default_factory=MaterialLibrary)
    """Reusable laboratory finishes."""
    objects: dict[AssetName, list[bpy.types.Object]] = field(default_factory=dict)
    """Meshes belonging to each independently exported body."""
    current: AssetName = AssetName.ENVIRONMENT
    """Body receiving newly created meshes."""
    measurements: dict[str, dict] = field(default_factory=dict)
    """Bounds captured before the bodies are posed for rendering."""

    def register(
        self, scene_object: bpy.types.Object, name: str, material: MaterialName
    ) -> bpy.types.Object:
        """Name and register a mesh with one shared material."""
        scene_object.name = name
        scene_object.data.materials.append(self.library.materials[material])
        self.objects.setdefault(self.current, []).append(scene_object)
        return scene_object

    def box(
        self,
        name: str,
        centre: tuple[float, float, float],
        size: tuple[float, float, float],
        material: MaterialName,
        bevel: float = 0.002,
    ) -> bpy.types.Object:
        """Create a manufactured rectangular body with rounded edge highlights."""
        bpy.ops.mesh.primitive_cube_add(size=1, location=centre)
        scene_object = self.register(bpy.context.object, name, material)
        scene_object.dimensions = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        if bevel:
            modifier = scene_object.modifiers.new("Machined edges", "BEVEL")
            modifier.width = bevel
            modifier.segments = 3
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            scene_object.data.use_auto_smooth = True
            modifier = scene_object.modifiers.new("Surface normals", "WEIGHTED_NORMAL")
            modifier.keep_sharp = True
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        return scene_object

    def cylinder(
        self,
        name: str,
        centre: tuple[float, float, float],
        radius: float,
        height: float,
        material: MaterialName,
    ) -> bpy.types.Object:
        """Create a cylindrical component with clean side normals."""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64, radius=radius, depth=height, location=centre
        )
        scene_object = self.register(bpy.context.object, name, material)
        for face in scene_object.data.polygons:
            face.use_smooth = len(face.vertices) == 4
        return scene_object

    def lathe(
        self,
        name: str,
        profile: tuple[ProfilePoint, ...],
        material: MaterialName,
        location: tuple[float, float, float] = (0, 0, 0),
        segments: int = 96,
    ) -> bpy.types.Object:
        """Create a smooth vessel from an explicitly dimensioned cross section."""
        surface = LatheMesh.from_profile(profile, segments)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(surface.vertices, [], surface.faces)
        mesh.update()
        scene_object = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(scene_object)
        scene_object.location = location
        for face in mesh.polygons:
            face.use_smooth = True
        return self.register(scene_object, name, material)

    def text(
        self,
        name: str,
        content: str,
        location: tuple[float, float, float],
        size: float,
        material: MaterialName,
        rotation: tuple[float, float, float] = (pi / 2, 0, 0),
    ) -> bpy.types.Object:
        """Add physical lettering which remains visible after GLB export."""
        bpy.ops.object.text_add(location=location, rotation=rotation)
        scene_object = bpy.context.object
        scene_object.data.body = content
        scene_object.data.size = size
        scene_object.data.extrude = 0.00003
        scene_object.data.space_character = 1.1
        bpy.ops.object.convert(target="MESH")
        return self.register(bpy.context.object, name, material)

    def logo(self) -> None:
        """Print teal AICOR lettering directly on each instrument surface."""
        image = bpy.data.images.load(
            str(Path(__file__).parents[2] / "src/cramera/web/img" / BrandingName.IMAGE),
            check_existing=True,
        )
        image.pack()
        material = bpy.data.materials.new(BrandingName.MATERIAL)
        material.use_nodes = True
        material.blend_method = "BLEND"
        material.use_screen_refraction = False
        color = self.library.materials[MaterialName.TEAL].diffuse_color
        material.diffuse_color = color
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Roughness"].default_value = 0.48
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        tint = material.node_tree.nodes.new("ShaderNodeMix")
        tint.data_type = "RGBA"
        tint.blend_type = "MULTIPLY"
        tint.inputs[0].default_value = 1
        tint.inputs[7].default_value = color
        material.node_tree.links.new(texture.outputs["Color"], tint.inputs[6])
        material.node_tree.links.new(tint.outputs[2], shader.inputs["Base Color"])
        material.node_tree.links.new(texture.outputs["Alpha"], shader.inputs["Alpha"])
        for placement in LogoPlacement.for_asset(self.current):
            self.mount_logo(placement, image, material)

    def mount_logo(
        self,
        placement: LogoPlacement,
        image: bpy.types.Image,
        material: bpy.types.Material,
    ) -> None:
        """Apply the original artwork to a flat or curved housing surface."""
        height = placement.width * image.size[1] / image.size[0]
        segments = placement.segments if placement.surface else 1
        vertices = []
        coordinates = []
        bpy.context.view_layer.update()
        for row in (0, 1):
            for column in range(segments + 1):
                horizontal = column / segments
                vertex = Vector(
                    (
                        placement.center[0] + (horizontal - 0.5) * placement.width,
                        placement.center[1],
                        placement.center[2] + (row - 0.5) * height,
                    )
                )
                if placement.surface:
                    surface = bpy.data.objects[placement.surface]
                    inverse = surface.matrix_world.inverted()
                    found, hit, _, _ = surface.ray_cast(
                        inverse @ vertex,
                        inverse.to_3x3() @ Vector((0, 1, 0)),
                    )
                    if not found:
                        raise LogoProjectionError(placement.name)
                    vertex = surface.matrix_world @ hit
                    vertex.y -= placement.clearance
                vertices.append(tuple(vertex))
                coordinates.append((horizontal, row))
        faces = [
            (column, column + 1, column + segments + 2, column + segments + 1)
            for column in range(segments)
        ]
        mesh = bpy.data.meshes.new(placement.name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        artwork = mesh.uv_layers.new()
        for polygon in mesh.polygons:
            for index in polygon.loop_indices:
                artwork.data[index].uv = coordinates[mesh.loops[index].vertex_index]
        scene_object = bpy.data.objects.new(placement.name, mesh)
        bpy.context.collection.objects.link(scene_object)
        self.register(scene_object, placement.name, MaterialName.WHITE)
        scene_object.data.materials.clear()
        scene_object.data.materials.append(material)

    def handle(self, name: str, centre: tuple[float, float, float]) -> None:
        """Build a U-shaped horizontal cabinet pull."""
        x, y, z = centre
        self.box(
            name, (x, y - 0.02, z), (0.14, 0.014, 0.014), MaterialName.STEEL, 0.004
        )
        for offset in (-0.061, 0.061):
            self.box(
                name + " mounting",
                (x + offset, y - 0.007, z),
                (0.012, 0.025, 0.014),
                MaterialName.STEEL,
            )

    def surface_lettering(
        self,
        lettering: bpy.types.Object,
        surface: bpy.types.Object,
        clearance: float = 0.0002,
        subdivisions: int = 0,
    ) -> None:
        """Center lettering and conform its printed faces to a curved housing."""
        if subdivisions:
            mesh = bmesh.new()
            mesh.from_mesh(lettering.data)
            bmesh.ops.subdivide_edges(
                mesh, edges=list(mesh.edges), cuts=subdivisions, use_grid_fill=True
            )
            mesh.to_mesh(lettering.data)
            mesh.free()
        horizontal = [vertex.co.x for vertex in lettering.data.vertices]
        lettering.location.x -= (min(horizontal) + max(horizontal)) / 2
        bpy.context.view_layer.update()
        lettering_inverse = lettering.matrix_world.inverted()
        surface_inverse = surface.matrix_world.inverted()
        support = BVHTree.FromPolygons(
            [vertex.co for vertex in surface.data.vertices],
            [tuple(face.vertices) for face in surface.data.polygons],
        )
        for vertex in lettering.data.vertices:
            position = lettering.matrix_world @ vertex.co
            printing_depth = lettering.location.y - position.y
            hit, _, _, _ = support.ray_cast(
                surface_inverse @ position,
                surface_inverse.to_3x3() @ Vector((0, 1, 0)),
            )
            if hit is None:
                raise LogoProjectionError(lettering.name)
            position = surface.matrix_world @ hit
            position.y -= clearance + printing_depth
            vertex.co = lettering_inverse @ position
        lettering.data.update()

    def adhesive_label(
        self,
        name: str,
        surface: bpy.types.Object,
        center: tuple[float, float, float],
        width: float,
        height: float,
        material: MaterialName,
    ) -> bpy.types.Object:
        """Fit a thin, rounded paper label against the actual bottle surface."""
        segments = 96
        corner_radius = min(width, height) * 0.03
        adhesive_clearance, paper_thickness = 0.00004, 0.00008
        vertices, faces = [], []
        bpy.context.view_layer.update()
        inverse = surface.matrix_world.inverted()
        for column in range(segments + 1):
            horizontal = width * (column / segments - 0.5)
            corner = max(0, abs(horizontal) - (width / 2 - corner_radius))
            extent = (
                height / 2 - corner_radius + max(0, corner_radius**2 - corner**2) ** 0.5
            )
            for clearance in (adhesive_clearance + paper_thickness, adhesive_clearance):
                for vertical in (-extent, extent):
                    start = Vector(
                        (center[0] + horizontal, center[1], center[2] + vertical)
                    )
                    found, hit, _, _ = surface.ray_cast(
                        inverse @ start, inverse.to_3x3() @ Vector((0, 1, 0))
                    )
                    if not found:
                        raise LogoProjectionError(name)
                    position = surface.matrix_world @ hit
                    position.y -= clearance
                    vertices.append(tuple(position))
        for column in range(segments):
            left, right = column * 4, (column + 1) * 4
            faces.extend(
                (
                    (left, right, right + 1, left + 1),
                    (left + 2, left + 3, right + 3, right + 2),
                    (left, left + 2, right + 2, right),
                    (left + 1, right + 1, right + 3, left + 3),
                )
            )
        end = segments * 4
        faces.extend(((0, 1, 3, 2), (end, end + 2, end + 3, end + 1)))
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        label = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(label)
        return self.register(label, name, material)

    def centrifuge_control_enclosure(
        self, surface: bpy.types.Object, front: float
    ) -> None:
        """Seat a rounded control enclosure against the cylindrical housing."""
        width, height, center_height = 0.216, 0.067, 0.079
        corner_radius, bevel, mounting_depth = 0.008, 0.003, 0.0001
        segments = 96
        vertices = []
        faces = []
        bpy.context.view_layer.update()
        inverse = surface.matrix_world.inverted()
        for column in range(segments + 1):
            horizontal = width * (column / segments - 0.5)
            inner_horizontal = horizontal * (width - 2 * bevel) / width
            profiles = []
            for coordinate, profile_width, profile_height, radius in (
                (
                    inner_horizontal,
                    width - 2 * bevel,
                    height - 2 * bevel,
                    corner_radius - bevel,
                ),
                (horizontal, width, height, corner_radius),
            ):
                corner_offset = max(0, abs(coordinate) - (profile_width / 2 - radius))
                extent = (
                    profile_height / 2
                    - radius
                    + (max(0, radius**2 - corner_offset**2)) ** 0.5
                )
                profiles.append((center_height - extent, center_height + extent))
            for level in profiles[0]:
                vertices.append((inner_horizontal, front, level))
            for level in profiles[1]:
                vertices.append((horizontal, front + bevel, level))
            for level in profiles[1]:
                start = Vector((horizontal, front, level))
                found, hit, _, _ = surface.ray_cast(
                    inverse @ start, inverse.to_3x3() @ Vector((0, 1, 0))
                )
                if not found:
                    raise LogoProjectionError("Centrifuge control bezel")
                position = surface.matrix_world @ hit
                position.y += mounting_depth
                vertices.append(tuple(position))
        for column in range(segments):
            left = column * 6
            right = left + 6
            faces.extend(
                (
                    (left, right, right + 1, left + 1),
                    (left + 2, right + 2, right, left),
                    (left + 1, right + 1, right + 3, left + 3),
                    (left + 4, right + 4, right + 2, left + 2),
                    (left + 3, right + 3, right + 5, left + 5),
                    (left + 4, left + 5, right + 5, right + 4),
                )
            )
        end = segments * 6
        faces.extend(
            (
                (0, 1, 3, 2),
                (2, 3, 5, 4),
                (end + 2, end + 3, end + 1, end),
                (end + 4, end + 5, end + 3, end + 2),
            )
        )
        name = "Centrifuge control bezel"
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        scene_object = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(scene_object)
        self.register(scene_object, name, MaterialName.TEAL)

    def drill(
        self, plate: bpy.types.Object, centre: tuple[float, float, float], radius: float
    ) -> None:
        """Cut a true vertical through-hole into a rack plate."""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64, radius=radius, depth=0.03, location=centre
        )
        cutter = bpy.context.object
        bpy.context.view_layer.objects.active = plate
        modifier = plate.modifiers.new("Open tube socket", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)

    def build_rack(self) -> None:
        """Build and export the independent six-socket carrier and its grasp bar."""
        previous = self.current
        self.current = AssetName.TUBE_RACK
        dimensions = RackDimensions()
        x, y, z = 0.0, 0.0, 0.0
        self.box(
            "Rack base", (x, y, z + 0.003), (0.19, 0.11, 0.006), MaterialName.PALE_TEAL
        )
        plate = self.box(
            "Rack perforated plate",
            (x, y, z + 0.065),
            (0.19, 0.11, 0.006),
            MaterialName.TEAL,
        )
        for column, offset_x in enumerate(dimensions.columns):
            for row, offset_y in enumerate(dimensions.rows):
                self.drill(plate, (x + offset_x, y + offset_y, z + 0.065), 0.0115)
                self.lathe(
                    f"Rack lower guide {column}_{row}",
                    dimensions.guide_profile(),
                    MaterialName.PALE_TEAL,
                    (offset_x, offset_y, 0),
                )
                self.text(
                    "Socket identifier",
                    f"{chr(65 + row)}{column + 1}",
                    (x + offset_x - 0.009, y + offset_y - 0.022, z + 0.0681),
                    0.006,
                    MaterialName.WHITE,
                    (0, 0, 0),
                )
        for offset_x in (-0.083, 0.083):
            for offset_y in (-0.045, 0.045):
                self.cylinder(
                    "Rack standoff",
                    (x + offset_x, y + offset_y, z + 0.034),
                    0.004,
                    0.056,
                    MaterialName.STEEL,
                )
        self.box(
            "Rack carry handle",
            dimensions.handle_center,
            dimensions.handle_size,
            MaterialName.STEEL,
            0.004,
        )
        for horizontal in (-0.052, 0.052):
            self.box(
                "Rack carry mounting",
                (horizontal, -0.0875, 0.055),
                (0.012, 0.075, 0.014),
                MaterialName.STEEL,
            )
        self.logo()
        self.export_asset(AssetName.TUBE_RACK, dimensions.origin)
        self.current = previous

    def build_bench(self) -> None:
        """Build a worktop and cabinets around the open sliding-drawer bay."""
        self.box(
            "Porcelain worktop",
            (0, 0, 0.876),
            (1.8, 0.75, 0.048),
            MaterialName.TOP,
            0.006,
        )
        self.box(
            "Rear splashback",
            (0, 0.352, 0.958),
            (1.78, 0.025, 0.116),
            MaterialName.TOP,
            0.003,
        )
        self.box(
            "Recessed plinth", (0, 0.027, 0.083), (1.66, 0.59, 0.12), MaterialName.DARK
        )
        for x in (-0.75, 0.75):
            for y in (-0.23, 0.23):
                self.cylinder(
                    "Adjustable foot", (x, y, 0.03), 0.025, 0.06, MaterialName.DARK
                )
        for x in (-0.81, 0.81):
            self.box(
                "Cabinet side",
                (x, 0.02, 0.468),
                (0.026, 0.654, 0.684),
                MaterialName.WHITE,
            )
        self.box(
            "Cabinet rear", (0, 0.33, 0.47), (1.62, 0.023, 0.7), MaterialName.WHITE
        )
        self.box(
            "Upper cross member",
            (0, 0.02, 0.833),
            (1.62, 0.64, 0.05),
            MaterialName.WHITE,
        )
        for x in (-0.225, 0.24):
            self.box(
                "Cabinet partition",
                (x, 0.015, 0.464),
                (0.021, 0.63, 0.675),
                MaterialName.WHITE,
            )
        self.box(
            "Drawer lower shelf",
            (-0.52, 0.006, 0.597),
            (0.558, 0.616, 0.018),
            MaterialName.WHITE,
        )
        self.box(
            "Lower left cabinet",
            (-0.52, -0.313, 0.352),
            (0.558, 0.023, 0.451),
            MaterialName.TEAL,
        )
        self.handle("Lower cabinet pull", (-0.52, -0.326, 0.546))
        for x in (0.005, 0.523):
            self.box(
                "Cabinet door",
                (x, -0.313, 0.469),
                (0.436 if x < 0.1 else 0.552, 0.024, 0.681),
                MaterialName.WHITE,
            )
            self.handle("Cabinet door pull", (x, -0.326, 0.744))
        for x in (-0.79, -0.25):
            self.box(
                "Drawer guide",
                (x, 0.015, 0.669),
                (0.009, 0.582, 0.014),
                MaterialName.STEEL,
            )
        self.text(
            "Workstation number",
            "LAB / 01",
            (0.536, -0.327, 0.17),
            0.023,
            MaterialName.TEAL,
        )

    def build_room(self) -> None:
        """Dress the bench with a restrained architectural laboratory setting."""
        self.box("Floor", (0, 0.2, -0.035), (4.6, 4.4, 0.07), MaterialName.FLOOR)
        self.box("Back wall", (0, 1.15, 1.4), (4.6, 0.08, 2.8), MaterialName.WALL)
        self.box("Left wall", (-1.85, 0.05, 1.4), (0.08, 2.2, 2.8), MaterialName.WALL)
        for x in (-1.5, -0.75, 0, 0.75, 1.5):
            self.box(
                "Wall panel seam",
                (x, 1.107, 1.16),
                (0.003, 0.004, 2.3),
                MaterialName.WHITE,
                0,
            )
        self.box(
            "Wall skirting", (0, 1.093, 0.074), (3.66, 0.016, 0.12), MaterialName.STEEL
        )
        self.box("Shelf", (-0.26, 0.93, 1.5), (1.69, 0.29, 0.022), MaterialName.WHITE)
        self.box(
            "Shelf light",
            (-0.26, 0.854, 1.487),
            (1.49, 0.033, 0.006),
            MaterialName.LIGHT,
        )
        for x in (-0.95, 0.40):
            self.box(
                "Shelf bracket",
                (x, 1.035, 1.406),
                (0.012, 0.16, 0.17),
                MaterialName.STEEL,
            )
        self.text(
            "Laboratory subtitle",
            "L A B O R A T O R Y   /   0 1",
            (-0.852, 1.101, 1.897),
            0.033,
            MaterialName.TEAL,
        )
        for x, height in ((-0.86, 0.16), (-0.69, 0.21), (-0.49, 0.16)):
            self.build_bottle(x, 0.955, 1.512, height)
        self.box(
            "Storage box",
            (0.02, 0.95, 1.595),
            (0.38, 0.235, 0.16),
            MaterialName.PALE_TEAL,
        )
        self.box(
            "Storage box lid",
            (0.02, 0.95, 1.68),
            (0.392, 0.245, 0.017),
            MaterialName.TEAL,
        )
        self.text(
            "Storage label",
            "GLASSWARE",
            (-0.1, 0.831, 1.583),
            0.021,
            MaterialName.WHITE,
        )
        self.box(
            "Ceiling task light",
            (0, 0.05, 2.42),
            (1.5, 0.22, 0.036),
            MaterialName.WHITE,
        )
        self.box(
            "Task light diffuser",
            (0, 0.05, 2.399),
            (1.44, 0.19, 0.004),
            MaterialName.LIGHT,
        )
        for x in (-0.6, 0.6):
            self.cylinder(
                "Light suspension", (x, 0.05, 2.61), 0.003, 0.35, MaterialName.STEEL
            )
        self.build_rack()
        self.build_bottle(-0.69, 0.21, 0.900, 0.175)
        self.build_bottle(-0.53, 0.22, 0.900, 0.135)
        self.box(
            "Instrument base",
            (0.65, 0.19, 0.931),
            (0.30, 0.22, 0.058),
            MaterialName.WHITE,
            0.014,
        )
        self.cylinder(
            "Balance weighing pan",
            (0.65, 0.21, 0.968),
            0.085,
            0.012,
            MaterialName.STEEL,
        )
        self.box(
            "Balance display",
            (0.65, 0.078, 0.94),
            (0.125, 0.003, 0.023),
            MaterialName.DARK,
        )
        self.text(
            "Balance digits",
            "0.000 g",
            (0.6, 0.075, 0.933),
            0.015,
            MaterialName.PALE_TEAL,
        )
        self.text(
            "Worktop station label",
            "01  /  SAMPLE PREPARATION",
            (-0.76, -0.376, 0.863),
            0.018,
            MaterialName.TEAL,
        )
        self.logo()

    def build_bottle(self, x: float, y: float, base: float, height: float) -> None:
        """Create capped reagent bottles for static room dressing."""
        radius = height * 0.23
        profile = (
            ProfilePoint(0, 0),
            ProfilePoint(radius * 0.9, 0),
            ProfilePoint(radius, 0.006),
            ProfilePoint(radius, height * 0.68),
            ProfilePoint(radius * 0.55, height * 0.81),
            ProfilePoint(radius * 0.55, height * 0.91),
            ProfilePoint(0, height * 0.91),
        )
        bottle = self.lathe("Reagent bottle", profile, MaterialName.WHITE, (x, y, base))
        self.cylinder(
            "Bottle screw cap",
            (x, y, base + height * 0.935),
            radius * 0.64,
            height * 0.13,
            MaterialName.TEAL,
        )
        self.label_reagent_bottle(bottle, x, y, base, height)

    def label_reagent_bottle(
        self, bottle: bpy.types.Object, x: float, y: float, base: float, height: float
    ) -> None:
        """Apply a fitted paper label and printed identification to a stored reagent."""
        radius = height * 0.23
        suffix = bottle.name.removeprefix("Reagent bottle")
        label = self.adhesive_label(
            f"Bottle label{suffix}",
            bottle,
            (x, y - radius - 0.0004, base + height * 0.41),
            radius * 1.33,
            height * 0.32,
            MaterialName.TEAL_PAPER,
        )
        title = self.text(
            f"Bottle lettering{suffix}",
            "BUFFER",
            (x, y - radius - 0.001, base + height * 0.45),
            height * 0.071,
            MaterialName.WHITE,
        )
        specification = self.text(
            f"Bottle specification{suffix}",
            "pH 7.4",
            (x, y - radius - 0.001, base + height * 0.32),
            height * 0.057,
            MaterialName.WHITE,
        )
        self.surface_lettering(title, label, clearance=0.00008, subdivisions=3)
        self.surface_lettering(specification, label, clearance=0.00008, subdivisions=3)

    def build_drawer(self) -> None:
        """Build a hollow tray centred on the prismatic joint origin."""
        self.current = AssetName.DRAWER
        self.box(
            "Drawer floor", (0, 0, -0.064), (0.526, 0.596, 0.012), MaterialName.STEEL
        )
        for x in (-0.269, 0.269):
            self.box("Drawer side", (x, 0, 0), (0.012, 0.62, 0.14), MaterialName.STEEL)
        self.box("Drawer back", (0, 0.304, 0), (0.55, 0.012, 0.14), MaterialName.STEEL)
        self.box("Drawer face", (0, -0.304, 0), (0.55, 0.012, 0.14), MaterialName.TEAL)
        self.handle("Drawer pull", (0, -0.316, 0.015))

    def build_tube(self, asset: AssetName, liquid: MaterialName | None) -> None:
        """Create one hollow glass vessel with separate optional contents."""
        self.current = asset
        dimensions = TubeDimensions()
        self.lathe("Hollow borosilicate tube", dimensions.profile(), MaterialName.GLASS)
        for number in range(1, 7):
            height = 0.03 + number * 0.014
            self.box(
                "Volume graduation",
                (-0.001, -0.00903, height),
                (0.004 if number % 2 else 0.006, 0.00008, 0.00045),
                MaterialName.LABEL,
                0,
            )
        if liquid is None:
            return
        height = 0.074 if liquid == MaterialName.AMBER else 0.095
        radius = dimensions.inner_radius - 0.0001
        profile = [ProfilePoint(0, 0.0014)]
        for index in range(1, 17):
            angle = -pi / 2 + pi / 2 * index / 16
            profile.append(
                ProfilePoint(radius * cos(angle), 0.0091 + radius * sin(angle))
            )
        profile.extend(
            (
                ProfilePoint(radius, height + 0.0007),
                ProfilePoint(radius * 0.9, height),
                ProfilePoint(0, height),
            )
        )
        self.lathe("Visible liquid fill", tuple(profile), liquid)

    def build_stopper(self) -> None:
        """Create a removable stem which fits the glass inner diameter."""
        self.current = AssetName.STOPPER
        profile = (
            ProfilePoint(0, 0),
            ProfilePoint(0.00735, 0),
            ProfilePoint(0.0076, 0.0115),
            ProfilePoint(0.0098, 0.0115),
            ProfilePoint(0.0105, 0.0122),
            ProfilePoint(0.0105, 0.020),
            ProfilePoint(0.0098, 0.021),
            ProfilePoint(0, 0.021),
        )
        self.lathe("Silicone stopper", profile, MaterialName.TEAL)

    def build_centrifuge(self) -> None:
        """Model a four-place bench centrifuge with an open bowl and hinged lid."""
        dimensions = CentrifugeDimensions()
        self.current = AssetName.CENTRIFUGE_BASE
        for horizontal in (-0.115, 0.115):
            for depth in (-0.11, 0.11):
                self.cylinder(
                    "Centrifuge isolation foot",
                    (horizontal, depth, 0.005),
                    0.018,
                    0.01,
                    MaterialName.DARK,
                )
        self.box(
            "Centrifuge enamel chassis",
            (0, 0, 0.027),
            (dimensions.width, dimensions.depth, 0.042),
            MaterialName.WHITE,
            0.019,
        )
        self.lathe(
            "Centrifuge chamber shell",
            (
                ProfilePoint(0.139, 0.042),
                ProfilePoint(0.156, 0.042),
                ProfilePoint(0.158, 0.055),
                ProfilePoint(0.152, 0.194),
                ProfilePoint(0.150, 0.210),
                ProfilePoint(0.145, 0.216),
                ProfilePoint(dimensions.chamber_radius, 0.216),
                ProfilePoint(dimensions.chamber_radius, 0.046),
                ProfilePoint(0.139, 0.042),
            ),
            MaterialName.WHITE,
        )
        self.lathe(
            "Centrifuge stainless bowl",
            (
                ProfilePoint(0, 0.033),
                ProfilePoint(0.1365, 0.033),
                ProfilePoint(0.137, 0.04),
                ProfilePoint(0.137, 0.211),
                ProfilePoint(0.1355, 0.211),
                ProfilePoint(0.1355, 0.041),
                ProfilePoint(0.13, 0.036),
                ProfilePoint(0, 0.036),
            ),
            MaterialName.STEEL,
        )
        self.lathe(
            "Centrifuge rim seal",
            (
                ProfilePoint(0.137, 0.214),
                ProfilePoint(0.147, 0.214),
                ProfilePoint(0.147, 0.220),
                ProfilePoint(0.137, 0.220),
                ProfilePoint(0.137, 0.214),
            ),
            MaterialName.DARK,
        )
        chamber = bpy.data.objects[BrandingName.CHAMBER]
        control_front = -0.1685
        self.centrifuge_control_enclosure(chamber, control_front)
        self.box(
            "Centrifuge display",
            (-0.025, control_front + 0.0013, 0.081),
            (0.123, 0.003, 0.034),
            MaterialName.DARK,
            0.0005,
        )
        self.text(
            "Centrifuge status lettering",
            "READY",
            (-0.075, control_front - 0.00025, 0.085),
            0.010,
            MaterialName.PALE_TEAL,
        )
        self.text(
            "Centrifuge capacity lettering",
            "4 x 15 ml",
            (-0.075, control_front - 0.00025, 0.072),
            0.008,
            MaterialName.WHITE,
        )
        self.box(
            "Centrifuge run key",
            (0.078, control_front, 0.083),
            (0.023, 0.006, 0.024),
            MaterialName.PALE_TEAL,
            0.005,
        )
        identity = self.text(
            "Centrifuge engraved identity",
            "CENTRIFUGE",
            (0, -0.2, 0.129),
            0.010,
            MaterialName.TEAL,
        )
        self.surface_lettering(identity, chamber)
        for side in (-1, 1):
            for index in range(6):
                depth = -0.035 + index * 0.014
                horizontal = side * ((0.157**2 - depth**2) ** 0.5 + 0.0005)
                self.box(
                    "Centrifuge ventilation slot",
                    (horizontal, depth, 0.068),
                    (0.004, 0.004, 0.026),
                    MaterialName.DARK,
                    0.001,
                )
        self.box(
            "Centrifuge hinge pedestal",
            (0, 0.147, 0.205),
            (0.094, 0.026, 0.037),
            MaterialName.TEAL,
            0.004,
        )
        self.logo()
        self.export_asset(AssetName.CENTRIFUGE_BASE, dimensions.origin)

        self.current = AssetName.CENTRIFUGE_ROTOR
        self.lathe(
            "Centrifuge balanced rotor plate",
            (
                ProfilePoint(0, 0),
                ProfilePoint(0.108, 0),
                ProfilePoint(0.111, 0.002),
                ProfilePoint(0.111, 0.004),
                ProfilePoint(0.108, 0.006),
                ProfilePoint(0, 0.006),
            ),
            MaterialName.TEAL,
        )
        self.lathe(
            "Centrifuge central hub",
            (
                ProfilePoint(0, 0.005),
                ProfilePoint(0.022, 0.005),
                ProfilePoint(0.024, 0.01),
                ProfilePoint(0.022, 0.025),
                ProfilePoint(0.016, 0.032),
                ProfilePoint(0, 0.032),
            ),
            MaterialName.STEEL,
        )
        self.cylinder(
            "Centrifuge hub fastener", (0, 0, 0.034), 0.010, 0.007, MaterialName.DARK
        )
        self.box(
            "Centrifuge fastener slot",
            (0, 0, 0.0376),
            (0.011, 0.0014, 0.0004),
            MaterialName.STEEL,
            0,
        )
        for index in range(dimensions.slot_count):
            horizontal, depth, bottom = dimensions.slot_position(index)
            height = bottom + dimensions.slot_height
            self.lathe(
                f"Centrifuge C{index + 1} hollow sleeve",
                (
                    ProfilePoint(0, bottom - 0.006),
                    ProfilePoint(dimensions.slot_outer_radius, bottom - 0.006),
                    ProfilePoint(dimensions.slot_outer_radius, height - 0.003),
                    ProfilePoint(dimensions.slot_outer_radius + 0.0015, height - 0.002),
                    ProfilePoint(dimensions.slot_outer_radius + 0.0015, height + 0.001),
                    ProfilePoint(dimensions.slot_inner_radius, height + 0.001),
                    ProfilePoint(dimensions.slot_inner_radius, bottom),
                    ProfilePoint(0, bottom),
                ),
                MaterialName.DARK,
                (horizontal, depth, 0),
            )
            self.text(
                f"Centrifuge C{index + 1} marker",
                str(index + 1),
                (horizontal * 1.33 - 0.004, depth * 1.33 - 0.004, 0.0062),
                0.010,
                MaterialName.WHITE,
                (0, 0, 0),
            )
        self.export_asset(
            AssetName.CENTRIFUGE_ROTOR,
            tuple(
                dimensions.origin[index] + dimensions.rotor_origin[index]
                for index in range(3)
            ),
        )
        rotor = bpy.data.objects[AssetName.CENTRIFUGE_ROTOR.value]
        rotor_constraint = rotor.constraints.new("LIMIT_ROTATION")
        rotor_constraint.name = "Vertical drive axis"
        rotor_constraint.owner_space = "LOCAL"
        rotor_constraint.use_limit_x = rotor_constraint.use_limit_y = True
        rotor_constraint.use_transform_limit = True

        self.current = AssetName.CENTRIFUGE_LID
        center = (0, -dimensions.lid_origin[1], 0)
        self.lathe(
            "Centrifuge lid enamel annulus",
            (
                ProfilePoint(0.105, 0.001),
                ProfilePoint(0.147, 0.001),
                ProfilePoint(0.152, 0.006),
                ProfilePoint(0.151, 0.018),
                ProfilePoint(0.145, 0.024),
                ProfilePoint(0.11, 0.027),
                ProfilePoint(0.105, 0.023),
                ProfilePoint(0.105, 0.001),
            ),
            MaterialName.WHITE,
            center,
        )
        self.lathe(
            "Centrifuge viewing window gasket",
            (
                ProfilePoint(0.1015, 0.012),
                ProfilePoint(0.1065, 0.012),
                ProfilePoint(0.1065, 0.024),
                ProfilePoint(0.1015, 0.024),
                ProfilePoint(0.1015, 0.012),
            ),
            MaterialName.DARK,
            center,
        )
        self.cylinder(
            "Centrifuge protective viewing window",
            (0, center[1], 0.019),
            0.102,
            0.006,
            MaterialName.GLASS,
        )
        self.box(
            "Centrifuge lid hinge",
            (0, 0, 0),
            (0.081, 0.024, 0.027),
            MaterialName.STEEL,
            0.004,
        )
        self.box(
            "Centrifuge lid grip",
            dimensions.handle_center,
            dimensions.handle_size,
            MaterialName.TEAL,
            0.006,
        )
        for position in dimensions.handle_mounts:
            self.box(
                "Centrifuge lid grip support",
                position,
                dimensions.handle_mount_size,
                MaterialName.TEAL,
                0.004,
            )
        self.export_asset(
            AssetName.CENTRIFUGE_LID,
            tuple(
                dimensions.origin[index] + dimensions.lid_origin[index]
                for index in range(3)
            ),
        )
        lid = bpy.data.objects[AssetName.CENTRIFUGE_LID.value]
        lid.rotation_euler.x = -dimensions.open_angle
        lid_constraint = lid.constraints.new("LIMIT_ROTATION")
        lid_constraint.name = "Rear cover hinge"
        lid_constraint.owner_space = "LOCAL"
        lid_constraint.use_limit_x = True
        lid_constraint.min_x = -dimensions.open_angle
        lid_constraint.max_x = 0
        lid_constraint.use_limit_y = lid_constraint.use_limit_z = True
        lid_constraint.use_transform_limit = True

    def pipetting_tubing(
        self,
        name: str,
        points: tuple[tuple[float, float, float], ...],
        material: MaterialName,
    ) -> None:
        """Create a rounded reagent feed line along its supporting hardware."""
        curve = bpy.data.curves.new(name, "CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 16
        curve.bevel_depth = 0.0017
        curve.bevel_resolution = 4
        spline = curve.splines.new("BEZIER")
        spline.bezier_points.add(len(points) - 1)
        for vertex, position in zip(spline.bezier_points, points):
            vertex.co = position
            vertex.handle_left_type = vertex.handle_right_type = "AUTO"
        scene_object = bpy.data.objects.new(name, curve)
        bpy.context.collection.objects.link(scene_object)
        bpy.ops.object.select_all(action="DESELECT")
        scene_object.select_set(True)
        bpy.context.view_layer.objects.active = scene_object
        bpy.ops.object.convert(target="MESH")
        self.register(bpy.context.object, name, material)

    def build_pipetting(self) -> None:
        """Model the bench dispenser with two reservoirs and an open receiving bay."""
        dimensions = PipettingDimensions()
        cover_center = (0, -0.032, 0.233)
        cover_size = (0.051, 0.187, 0.006)
        cover_top = cover_center[2] + cover_size[2] / 2
        inlet_length = 0.012
        inlet_insertion = 0.002
        inlet_height = cover_top - inlet_insertion + inlet_length / 2
        inlet_radius = 0.0037
        self.current = AssetName.PIPETTING
        self.box(
            "Pipetting enamel platform",
            dimensions.platform_center,
            dimensions.platform_size,
            MaterialName.WHITE,
            0.005,
        )
        self.box(
            "Pipetting rear housing",
            dimensions.column_center,
            dimensions.column_size,
            MaterialName.WHITE,
            0.009,
        )
        self.box(
            "Pipetting lower plinth",
            (0, 0.085, 0.026),
            (0.217, 0.089, 0.018),
            MaterialName.TEAL,
            0.004,
        )
        for horizontal in (-0.106, 0.106):
            self.box(
                "Pipetting side trim",
                (horizontal, 0.085, 0.137),
                (0.006, 0.068, 0.204),
                MaterialName.TEAL,
                0.003,
            )
        self.box(
            "Pipetting head",
            dimensions.head_center,
            dimensions.head_size,
            MaterialName.TEAL,
            0.005,
        )
        self.box(
            "Pipetting head cover",
            cover_center,
            cover_size,
            MaterialName.WHITE,
            0.002,
        )
        self.box(
            "Pipetting head nose",
            (0, -0.115, 0.209),
            (0.030, 0.028, 0.006),
            MaterialName.STEEL,
            0.002,
        )
        self.box(
            "Pipetting receiving inset",
            (0, -0.097, 0.0155),
            (0.115, 0.101, 0.001),
            MaterialName.STEEL,
            0.004,
        )
        for horizontal in (-0.042, 0.042):
            for depth in (-0.13, -0.08):
                self.cylinder(
                    "Pipetting platform screw",
                    (horizontal, depth, 0.01615),
                    0.0018,
                    0.0003,
                    MaterialName.DARK,
                )
        for guide in dimensions.holder_guides():
            self.box(
                guide.name, guide.center, guide.size, MaterialName.PALE_TEAL, 0.001
            )
        self.text(
            "Pipetting loading label",
            "SAMPLE",
            (-0.019, -0.151, 0.0162),
            0.005,
            MaterialName.TEAL,
            (0, 0, 0),
        )
        self.logo()
        self.text(
            "Pipetting instrument label",
            "DUAL DISPENSER",
            (-0.095, 0.0391, 0.202),
            0.0065,
            MaterialName.TEAL,
        )
        self.box(
            "Pipetting screen bezel",
            (
                dimensions.display[0],
                dimensions.display[1] + 0.001,
                dimensions.display[2],
            ),
            (
                dimensions.display_size[0] + 0.006,
                0.0015,
                dimensions.display_size[1] + 0.006,
            ),
            MaterialName.TEAL,
            0.002,
        )
        self.box(
            PipettingPart.DISPLAY,
            dimensions.display,
            (dimensions.display_size[0], 0.0008, dimensions.display_size[1]),
            MaterialName.DARK,
            0.0005,
        )
        self.text(
            "Pipetting channel label",
            "A  /  B",
            (0.038, 0.0391, 0.166),
            0.009,
            MaterialName.TEAL,
        )
        for index, channel in enumerate(PipettingChannel):
            sign = -1 if channel is PipettingChannel.AMBER else 1
            horizontal = sign * dimensions.reservoir_horizontal_offset
            material = (
                MaterialName.AMBER
                if channel is PipettingChannel.AMBER
                else MaterialName.LIQUID_TEAL
            )
            self.cylinder(
                f"Pipetting {channel} reservoir seat",
                (horizontal, 0.085, 0.252),
                0.030,
                0.004,
                MaterialName.TEAL,
            )
            bottle = self.lathe(
                f"Pipetting {channel} bottle",
                (
                    ProfilePoint(0, 0.254),
                    ProfilePoint(0.021, 0.254),
                    ProfilePoint(0.025, 0.258),
                    ProfilePoint(0.025, 0.343),
                    ProfilePoint(0.022, 0.352),
                    ProfilePoint(0.013, 0.365),
                    ProfilePoint(0.013, 0.379),
                    ProfilePoint(0.010, 0.379),
                    ProfilePoint(0.010, 0.365),
                    ProfilePoint(0.019, 0.350),
                    ProfilePoint(0.023, 0.342),
                    ProfilePoint(0.023, 0.259),
                    ProfilePoint(0, 0.259),
                ),
                MaterialName.GLASS,
                (horizontal, 0.085, 0),
            )
            self.cylinder(
                f"Pipetting {channel} liquid",
                (
                    horizontal,
                    0.085,
                    dimensions.reservoir_liquid_bottom
                    + dimensions.reservoir_liquid_height / 2,
                ),
                0.0228,
                dimensions.reservoir_liquid_height,
                material,
            )
            self.cylinder(
                f"Pipetting {channel} cap",
                (horizontal, 0.085, 0.379),
                0.016,
                0.013,
                MaterialName.DARK,
            )
            self.cylinder(
                f"Pipetting {channel} cap band",
                (horizontal, 0.085, 0.373),
                0.0162,
                0.002,
                MaterialName.PALE_TEAL,
            )
            for rib in range(24):
                angle = 2 * pi * rib / 24
                self.cylinder(
                    f"Pipetting {channel} cap knurl",
                    (
                        horizontal + 0.0157 * cos(angle),
                        0.085 + 0.0157 * sin(angle),
                        0.379,
                    ),
                    0.0006,
                    0.010,
                    MaterialName.DARK,
                )
            self.cylinder(
                f"Pipetting {channel} feed coupling",
                (horizontal, 0.085, 0.390),
                0.0037,
                0.010,
                MaterialName.STEEL,
            )
            label = self.adhesive_label(
                f"Pipetting {channel} bottle label",
                bottle,
                (horizontal, 0.0598, 0.291),
                0.036,
                0.024,
                MaterialName.PAPER,
            )
            title = self.text(
                f"Pipetting {channel} label text",
                f"REAGENT {chr(65 + index)}",
                (horizontal, 0.0592, 0.294),
                0.0038,
                MaterialName.TEAL,
            )
            identification = self.text(
                f"Pipetting {channel} identification",
                str(channel).upper(),
                (horizontal, 0.0592, 0.284),
                0.0045,
                MaterialName.TEAL,
            )
            self.surface_lettering(title, label, clearance=0.00008, subdivisions=3)
            self.surface_lettering(
                identification, label, clearance=0.00008, subdivisions=3
            )
            for graduation in range(5):
                self.box(
                    f"Pipetting {channel} graduation",
                    (horizontal + 0.016, 0.0647, 0.317 + graduation * 0.005),
                    (0.005 if graduation % 2 else 0.008, 0.0005, 0.0006),
                    MaterialName.LABEL,
                    0,
                )
            inlet_position = (sign * 0.009, -0.101, inlet_height)
            self.cylinder(
                f"Pipetting {channel} head inlet coupling",
                inlet_position,
                inlet_radius,
                inlet_length,
                MaterialName.STEEL,
            )
            self.pipetting_tubing(
                f"Pipetting {channel} feed tube",
                (
                    (horizontal, 0.085, 0.390),
                    (horizontal, 0.040, 0.408),
                    (sign * 0.019, 0.018, 0.354),
                    (sign * 0.011, -0.021, 0.279),
                    (
                        inlet_position[0],
                        inlet_position[1],
                        inlet_position[2] + inlet_length * 1.5,
                    ),
                    inlet_position,
                ),
                MaterialName.PALE_TEAL if index else MaterialName.LABEL,
            )
            nozzle = dimensions.channel_nozzle(channel)
            self.cylinder(
                f"Pipetting {channel} nozzle fitting",
                (nozzle[0], nozzle[1], 0.206),
                0.00165,
                0.007,
                MaterialName.DARK,
            )
            self.cylinder(
                f"Pipetting {channel} nozzle",
                (nozzle[0], nozzle[1], nozzle[2] + dimensions.nozzle_length / 2),
                dimensions.nozzle_radius,
                dimensions.nozzle_length,
                MaterialName.STEEL,
            )
            self.box(
                f"Pipetting {channel} indicator bezel",
                (0.037 + index * 0.026, 0.0387, 0.145),
                (0.016, 0.002, 0.008),
                MaterialName.TEAL,
                0.002,
            )
            self.box(
                f"Pipetting {channel} indicator",
                (0.037 + index * 0.026, 0.0375, 0.145),
                (0.010, 0.0005, 0.003),
                MaterialName.PALE_TEAL if index else MaterialName.LABEL,
                0.0007,
            )
        for horizontal in (-0.077, 0.077):
            for vent in range(6):
                self.box(
                    "Pipetting ventilation slot",
                    (horizontal, 0.0396, 0.078 + vent * 0.006),
                    (0.033, 0.001, 0.002),
                    MaterialName.DARK,
                    0.0007,
                )
        self.text(
            "Pipetting nozzle label",
            "A  B",
            (-0.010, -0.1292, 0.217),
            0.005,
            MaterialName.WHITE,
        )
        self.export_asset(AssetName.PIPETTING, dimensions.origin)

    def build_refrigerator(self) -> None:
        """Author an insulated cabinet, six open sample supports and glass door."""
        dimensions = RefrigeratorDimensions()
        self.current = AssetName.REFRIGERATOR_BASE
        for box in dimensions.cabinet_boxes():
            shelf = "shelf" in box.name
            self.box(
                box.name.replace("refrigerator_", "Refrigerator ").replace("_", " "),
                box.center,
                box.size,
                MaterialName.STEEL if shelf else MaterialName.WHITE,
                0.002 if shelf else 0.006,
            )
        self.box(
            "Refrigerator interior liner",
            (0, 0.2485, 0.43),
            (0.468, 0.003, 0.575),
            MaterialName.PALE_TEAL,
            0.001,
        )
        for horizontal in (-0.231, 0.231):
            self.box(
                "Refrigerator shelf rail",
                (horizontal, -0.005, 0.23),
                (0.008, 0.465, 0.018),
                MaterialName.WHITE,
            )
            self.box(
                "Refrigerator shelf rail",
                (horizontal, -0.005, 0.47),
                (0.008, 0.465, 0.018),
                MaterialName.WHITE,
            )
            self.box(
                "Refrigerator interior light",
                (horizontal, -0.23, 0.44),
                (0.007, 0.008, 0.48),
                MaterialName.LIGHT,
                0.002,
            )
        for horizontal in (-0.22, 0.22):
            for depth in (-0.235, 0.235):
                self.cylinder(
                    "Refrigerator leveling foot",
                    (horizontal, depth, 0.01),
                    0.026,
                    0.02,
                    MaterialName.DARK,
                )
        self.box(
            "Refrigerator ventilation plinth",
            (0, -0.292, 0.08),
            (0.462, 0.009, 0.092),
            MaterialName.TEAL,
            0.004,
        )
        for level in range(5):
            self.box(
                "Refrigerator ventilation slot",
                (0, -0.297, 0.052 + level * 0.014),
                (0.397, 0.0015, 0.0045),
                MaterialName.DARK,
                0.0005,
            )
        for elevation in (-0.18, 0.18):
            self.box(
                "Refrigerator hinge mounting",
                (
                    dimensions.door_origin[0],
                    -0.299,
                    dimensions.door_origin[2] + elevation,
                ),
                (0.022, 0.035, 0.056),
                MaterialName.STEEL,
                0.003,
            )
        self.logo()
        self.text(
            "Refrigerator instrument label",
            "SAMPLE COOLER",
            (-0.183, -0.2904, 0.753),
            0.014,
            MaterialName.TEAL,
        )
        self.box(
            "Refrigerator display bezel",
            (dimensions.display[0], -0.291, dimensions.display[2]),
            (
                dimensions.display_size[0] + 0.013,
                0.006,
                dimensions.display_size[1] + 0.012,
            ),
            MaterialName.TEAL,
            0.003,
        )
        self.box(
            RefrigeratorPart.DISPLAY,
            dimensions.display,
            (dimensions.display_size[0], 0.002, dimensions.display_size[1]),
            MaterialName.DARK,
            0.0005,
        )
        for index, position in enumerate(dimensions.slot_positions()):
            inner = dimensions.collar_inner_radius
            outer = dimensions.collar_outer_radius
            height = dimensions.collar_height
            self.lathe(
                f"Refrigerator R{index + 1} sample collar",
                (
                    ProfilePoint(inner, 0),
                    ProfilePoint(outer, 0),
                    ProfilePoint(outer, height - 0.002),
                    ProfilePoint(outer - 0.002, height),
                    ProfilePoint(inner + 0.001, height),
                    ProfilePoint(inner, height - 0.002),
                    ProfilePoint(inner, 0),
                ),
                MaterialName.TEAL,
                position,
            )
            self.text(
                f"Refrigerator R{index + 1} label",
                f"R{index + 1}",
                (position[0] - 0.008, position[1] - 0.041, position[2] + 0.0002),
                0.013,
                MaterialName.TEAL,
                (0, 0, 0),
            )
        self.export_asset(AssetName.REFRIGERATOR_BASE, dimensions.origin)

        self.current = AssetName.REFRIGERATOR_DOOR
        for box in dimensions.door_boxes():
            is_glass = box.name == RefrigeratorPart.GLASS
            name = (
                "Refrigerator door glass"
                if is_glass
                else box.name.replace("refrigerator_", "Refrigerator ").replace(
                    "_", " "
                )
            )
            self.box(
                name,
                box.center,
                box.size,
                MaterialName.GLASS if is_glass else MaterialName.TEAL,
                0.001 if is_glass else 0.004,
            )
        for horizontal in (0.038, 0.462):
            self.box(
                "Refrigerator glazing gasket",
                (horizontal, -0.001, 0),
                (0.006, 0.012, 0.492),
                MaterialName.DARK,
                0.001,
            )
        for elevation in (-0.244, 0.244):
            self.box(
                "Refrigerator glazing gasket",
                (0.25, -0.001, elevation),
                (0.425, 0.012, 0.006),
                MaterialName.DARK,
                0.001,
            )
        for elevation in (-0.18, 0.18):
            self.cylinder(
                "Refrigerator door hinge",
                (0, 0, elevation),
                0.012,
                0.062,
                MaterialName.STEEL,
            )
        for elevation in (-0.085, 0.085):
            self.box(
                "Refrigerator handle mounting",
                (0.482, -0.026, elevation),
                (0.016, 0.035, 0.016),
                MaterialName.STEEL,
                0.004,
            )
        self.box(
            "Refrigerator handle",
            dimensions.handle_center,
            (0.018, 0.018, 0.203),
            MaterialName.STEEL,
            0.006,
        )
        self.export_asset(
            AssetName.REFRIGERATOR_DOOR,
            tuple(dimensions.world_position(dimensions.door_origin)),
        )
        door = bpy.data.objects[AssetName.REFRIGERATOR_DOOR]
        constraint = door.constraints.new("LIMIT_ROTATION")
        constraint.name = "Front-left door hinge"
        constraint.owner_space = "LOCAL"
        constraint.use_limit_x = constraint.use_limit_y = constraint.use_limit_z = True
        constraint.min_z = -dimensions.door_open
        constraint.max_z = 0
        constraint.use_transform_limit = True

    def build_refrigerator_assets(self) -> None:
        """Update the editable cold-storage model and preserve every room object."""
        source = self.options.output / "laboratory.blend"
        existing_source = source.exists()
        if existing_source:
            bpy.ops.wm.open_mainfile(filepath=str(source))
        else:
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.delete(use_global=False)
        for scene_object in list(bpy.data.objects):
            if scene_object.name.startswith(("Refrigerator", "refrigerator_")):
                bpy.data.objects.remove(scene_object, do_unlink=True)
        (self.options.output / "assets").mkdir(parents=True, exist_ok=True)
        self.library.build()
        self.build_refrigerator()
        if not existing_source:
            self.setup_render()
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        position = RefrigeratorDimensions().origin
        scene = bpy.context.scene
        scene.camera = self.camera(
            "Refrigerator detail",
            (position[0] + 0.72, -1.32, 1.07),
            (position[0], 0, 0.43),
            53,
            11,
        )
        scene.cycles.samples = self.options.samples
        scene.cycles.use_denoising = False
        scene.render.resolution_x = self.options.width
        scene.render.resolution_y = round(self.options.width * 1.12)
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(self.options.output / "render_refrigerator.png")
        bpy.ops.wm.save_as_mainfile(
            filepath=str(self.options.output / "refrigerator.blend")
        )
        (self.options.output / "assets" / "refrigerator_measurements.json").write_text(
            json.dumps(self.measurements, indent=2) + "\n"
        )
        if self.options.render:
            bpy.ops.render.render(write_still=True)

    def build_pipetting_assets(self) -> None:
        """Add an editable dispenser and detail render without rebuilding the lab."""
        source = self.options.output / "laboratory.blend"
        existing_source = source.exists()
        if existing_source:
            bpy.ops.wm.open_mainfile(filepath=str(source))
        else:
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.delete(use_global=False)
        for scene_object in list(bpy.data.objects):
            if scene_object.name.startswith(
                ("Pipetting", "pipetting_", BrandingName.DISPENSER)
            ):
                bpy.data.objects.remove(scene_object, do_unlink=True)
        (self.options.output / "assets").mkdir(parents=True, exist_ok=True)
        self.library.build()
        self.build_pipetting()
        if not existing_source:
            self.setup_render()
        scene = bpy.context.scene
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        position = PipettingDimensions().origin
        scene.camera = self.camera(
            "Pipetting detail",
            (position[0] + 0.52, -0.77, position[2] + 0.51),
            (position[0], -0.015, position[2] + 0.20),
            58,
            11,
        )
        scene.cycles.samples = self.options.samples
        scene.cycles.use_denoising = False
        scene.render.resolution_x = self.options.width
        scene.render.resolution_y = round(self.options.width * 0.90)
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(self.options.output / "render_pipetting.png")
        bpy.ops.wm.save_as_mainfile(
            filepath=str(self.options.output / "pipetting.blend")
        )
        (self.options.output / "assets" / "pipetting_measurements.json").write_text(
            json.dumps(self.measurements, indent=2) + "\n"
        )
        if self.options.render:
            bpy.ops.render.render(write_still=True)

    def build_centrifuge_assets(self) -> None:
        """Add instrument assets and an editable source without replacing the lab."""
        source = self.options.output / "laboratory.blend"
        if source.exists():
            bpy.ops.wm.open_mainfile(filepath=str(source))
        else:
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.delete(use_global=False)
        for scene_object in list(bpy.data.objects):
            if scene_object.name.startswith(("Centrifuge", "centrifuge_")):
                bpy.data.objects.remove(scene_object, do_unlink=True)
        (self.options.output / "assets").mkdir(parents=True, exist_ok=True)
        self.library.build()
        self.build_centrifuge()
        if not source.exists():
            self.setup_render()
        dimensions = CentrifugeDimensions()
        position = dimensions.origin
        camera = self.camera(
            "Centrifuge detail",
            (position[0] + 0.58, position[1] - 0.82, position[2] + 0.68),
            (position[0], position[1], position[2] + 0.21),
            40,
            11,
        )
        scene = bpy.context.scene
        scene.camera = camera
        scene.cycles.samples = self.options.samples
        scene.cycles.use_denoising = False
        scene.render.resolution_x = self.options.width
        scene.render.resolution_y = round(self.options.width * 0.75)
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(self.options.output / "render_centrifuge.png")
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(
            filepath=str(self.options.output / "centrifuge.blend")
        )
        (self.options.output / "assets" / "centrifuge_measurements.json").write_text(
            json.dumps(self.measurements, indent=2) + "\n"
        )
        if self.options.render:
            bpy.ops.render.render(write_still=True)

    def refresh_branding_assets(self) -> None:
        """Re-export only branded bodies, preserving all other authored meshes."""
        source = self.options.output / "laboratory.blend"
        bpy.ops.wm.open_mainfile(filepath=str(source))
        self.library.build()
        if AssetName.TUBE_RACK not in bpy.data.objects:
            room = bpy.data.objects[AssetName.ENVIRONMENT]
            for scene_object in list(room.children):
                if scene_object.name.startswith(
                    ("Rack", "Socket identifier", "Tube rack")
                ):
                    bpy.data.objects.remove(scene_object, do_unlink=True)
            self.build_rack()
        cross_member = bpy.data.objects["Upper cross member"]
        heights = [
            (cross_member.matrix_world @ Vector(point)).z
            for point in cross_member.bound_box
        ]
        cross_member.location.z += 0.833 - (min(heights) + max(heights)) / 2
        retired = (
            BrandingName.ROOM_TITLE,
            BrandingName.INSTRUMENT_TITLE,
            BrandingName.TRAY,
            BrandingName.PIPETTE,
            BrandingName.DRAWER_MAT,
            BrandingName.DRAWER_DIVIDER,
        )
        for asset in (
            AssetName.ENVIRONMENT,
            AssetName.TUBE_RACK,
            AssetName.DRAWER,
            AssetName.CENTRIFUGE_BASE,
            AssetName.PIPETTING,
        ):
            body = bpy.data.objects[asset.value]
            placement = tuple(body.location)
            objects = []
            for scene_object in list(body.children):
                if scene_object.name.startswith(retired) or any(
                    name in scene_object.name
                    for name in (BrandingName.LOGO, BrandingName.PLAQUE)
                ):
                    bpy.data.objects.remove(scene_object, do_unlink=True)
                    continue
                transform = scene_object.matrix_local.copy()
                scene_object.parent = None
                scene_object.matrix_world = transform
                objects.append(scene_object)
            bpy.data.objects.remove(body, do_unlink=True)
            self.current = asset
            self.objects[asset] = objects
            self.logo()
            self.export_asset(asset, placement)
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(source))

    def export_asset(
        self, asset: AssetName, placement: tuple[float, float, float]
    ) -> None:
        """Bake mesh transforms and export the local Z-up rigid body."""
        bpy.ops.object.select_all(action="DESELECT")
        objects = self.objects[asset]
        for scene_object in objects:
            scene_object.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        vertices = [
            scene_object.matrix_world @ vertex.co
            for scene_object in objects
            for vertex in scene_object.data.vertices
        ]
        lower = [min(point[axis] for point in vertices) for axis in range(3)]
        upper = [max(point[axis] for point in vertices) for axis in range(3)]
        self.measurements[asset.value] = {
            "minimum": lower,
            "maximum": upper,
            "dimensions": [upper[axis] - lower[axis] for axis in range(3)],
            "meshes": len(objects),
            "vertices": len(vertices),
        }
        bpy.ops.export_scene.gltf(
            filepath=str(self.options.output / "assets" / f"{asset.value}.glb"),
            export_format="GLB",
            use_selection=True,
            export_yup=False,
            export_apply=True,
            export_animations=False,
            export_cameras=False,
            export_lights=False,
            export_materials="EXPORT",
        )
        body = bpy.data.objects.new(asset.value, None)
        bpy.context.collection.objects.link(body)
        body.empty_display_size = 0.035
        body.location = placement
        for scene_object in objects:
            scene_object.parent = body
        if asset == AssetName.DRAWER:
            constraint = body.constraints.new("LIMIT_LOCATION")
            constraint.name = "Prismatic drawer travel"
            constraint.use_min_x = constraint.use_max_x = True
            constraint.use_min_y = constraint.use_max_y = True
            constraint.use_min_z = constraint.use_max_z = True
            constraint.min_x = constraint.max_x = placement[0]
            constraint.min_y, constraint.max_y = -0.36, 0
            constraint.min_z = constraint.max_z = placement[2]
            constraint.use_transform_limit = True

    def instance_asset(
        self,
        source: AssetName,
        asset: AssetName,
        placement: tuple[float, float, float],
    ) -> None:
        """Place an independent rigid body sharing an existing asset's mesh data."""
        body = bpy.data.objects.new(asset.value, None)
        bpy.context.collection.objects.link(body)
        body.empty_display_size = 0.035
        body.location = placement
        self.objects[asset] = []
        for original in self.objects[source]:
            instance = original.copy()
            bpy.context.collection.objects.link(instance)
            instance.parent = body
            self.objects[asset].append(instance)
        self.measurements[asset.value] = self.measurements[source.value].copy()

    def area_light(
        self,
        name: str,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
        energy: float,
        size: float,
        colour: tuple[float, float, float] = (1, 1, 1),
    ) -> None:
        """Place a soft studio luminaire aimed at the work surface."""
        light = bpy.data.lights.new(name, "AREA")
        light.energy = energy
        light.shape = "DISK"
        light.size = size
        light.color = colour
        scene_object = bpy.data.objects.new(name, light)
        bpy.context.collection.objects.link(scene_object)
        scene_object.location = position
        scene_object.rotation_euler = (
            (Vector(target) - scene_object.location).to_track_quat("-Z", "Y").to_euler()
        )

    def camera(
        self,
        name: str,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
        lens: float,
        aperture: float,
    ) -> bpy.types.Object:
        """Create a perspective camera focused on the relevant work area."""
        camera = bpy.data.cameras.new(name)
        scene_object = bpy.data.objects.new(name, camera)
        bpy.context.collection.objects.link(scene_object)
        scene_object.location = position
        scene_object.rotation_euler = (
            (Vector(target) - scene_object.location).to_track_quat("-Z", "Y").to_euler()
        )
        camera.lens = lens
        camera.clip_start = 0.005
        camera.dof.use_dof = True
        camera.dof.focus_distance = (Vector(target) - scene_object.location).length
        camera.dof.aperture_fstop = aperture
        return scene_object

    def setup_render(self) -> tuple[bpy.types.Object, bpy.types.Object]:
        """Configure Cycles, broad window light and detail photography."""
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "CUDA"
        preferences.get_devices()
        accelerators = [
            device for device in preferences.devices if device.type == "CUDA"
        ]
        if accelerators:
            for device in preferences.devices:
                device.use = device.type == "CUDA"
            scene.cycles.device = "GPU"
        scene.cycles.samples = self.options.samples
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 16
        scene.cycles.transmission_bounces = 12
        scene.cycles.transparent_max_bounces = 12
        scene.cycles.adaptive_threshold = 0.015
        scene.render.resolution_x = self.options.width
        scene.render.resolution_y = round(self.options.width * 0.68)
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.world.use_nodes = True
        scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (
            0.72,
            0.82,
            0.92,
            1,
        )
        scene.world.node_tree.nodes["Background"].inputs[
            "Strength"
        ].default_value = 0.28
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -1.8
        self.area_light(
            "Window key", (-1.35, -1.05, 2.5), (0, 0, 0.8), 360, 2.2, (0.88, 0.95, 1)
        )
        self.area_light(
            "Right reflection",
            (1.35, 0.25, 2),
            (0.1, 0, 0.95),
            170,
            1.2,
            (1, 0.90, 0.79),
        )
        self.area_light(
            "Front glass reflection", (0.2, -1.9, 1.4), (0.2, 0, 1), 90, 1.0
        )
        self.area_light("Overhead practical", (0, 0.02, 2.37), (0, 0, 0.9), 100, 1.2)
        overview = self.camera(
            "Overview", (2.65, -3.6, 2.42), (-0.04, 0.12, 1.02), 44, 8
        )
        detail = self.camera(
            "Glassware detail", (0.69, -0.77, 1.40), (0.275, -0.055, 0.946), 65, 11
        )
        return overview, detail

    def build(self) -> None:
        """Generate every body, save the source and render both reference views."""
        self._build()
        LaboratoryAssetRights().install(self.options.output)

    def _build(self) -> None:
        """Generate either selected instruments or the complete laboratory scene."""
        if self.options.branding_only:
            self.refresh_branding_assets()
            return
        if self.options.centrifuge_only:
            self.build_centrifuge_assets()
            return
        if self.options.pipetting_only:
            self.build_pipetting_assets()
            return
        if self.options.refrigerator_only:
            self.build_refrigerator_assets()
            return
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.context.scene.unit_settings.system = "METRIC"
        bpy.context.scene.unit_settings.scale_length = 1
        (self.options.output / "assets").mkdir(parents=True, exist_ok=True)
        self.library.build()
        self.build_bench()
        self.build_room()
        self.export_asset(AssetName.ENVIRONMENT, (0, 0, 0))
        self.build_drawer()
        self.export_asset(AssetName.DRAWER, (-0.52, -0.18, 0.69))
        for asset, material, position in (
            (AssetName.TUBE_CLEAR, None, (0.16, -0.065, 0.906)),
            (AssetName.TUBE_AMBER, MaterialName.AMBER, (0.22, -0.065, 0.906)),
            (AssetName.TUBE_TEAL, MaterialName.LIQUID_TEAL, (0.16, -0.015, 0.906)),
        ):
            self.build_tube(asset, material)
            self.export_asset(asset, position)
        self.instance_asset(
            AssetName.TUBE_CLEAR, AssetName.TUBE_RESERVE, (0.28, -0.015, 0.906)
        )
        self.build_stopper()
        self.export_asset(AssetName.STOPPER, (0.50, -0.12, 0.900))
        self.build_centrifuge()
        self.build_pipetting()
        self.build_refrigerator()
        cameras = self.setup_render()
        bpy.context.scene.camera = cameras[0]
        bpy.context.scene.render.filepath = str(
            self.options.output / "render_overview.png"
        )
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(
            filepath=str(self.options.output / "laboratory.blend")
        )
        (self.options.output / "assets" / "measurements.json").write_text(
            json.dumps(self.measurements, indent=2) + "\n"
        )
        if self.options.render:
            for camera, view in zip(cameras, (RenderView.OVERVIEW, RenderView.CLOSEUP)):
                if self.options.view not in (RenderView.ALL, view):
                    continue
                bpy.context.scene.camera = camera
                bpy.context.scene.render.filepath = str(
                    self.options.output / f"render_{view.value}.png"
                )
                bpy.ops.render.render(write_still=True)


# %% Command line
def main() -> None:
    """Read Blender's trailing arguments and build the laboratory bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / ".cramera" / "scenes" / "aicor_medical_lab",
    )
    parser.add_argument("--samples", type=int, default=BuildOptions.samples)
    parser.add_argument("--width", type=int, default=BuildOptions.width)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--centrifuge-only", action="store_true")
    parser.add_argument("--pipetting-only", action="store_true")
    parser.add_argument("--branding-only", action="store_true")
    parser.add_argument("--refrigerator-only", action="store_true")
    parser.add_argument(
        "--view", type=RenderView, choices=list(RenderView), default=RenderView.ALL
    )
    arguments = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    LaboratoryBuilder(
        BuildOptions(
            arguments.output,
            arguments.samples,
            arguments.width,
            not arguments.skip_render,
            arguments.view,
            arguments.centrifuge_only,
            arguments.pipetting_only,
            arguments.branding_only,
            arguments.refrigerator_only,
        )
    ).build()


if __name__ == "__main__":
    main()
