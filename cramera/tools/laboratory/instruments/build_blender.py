"""
Build editable, articulated AICOR instruments and self-contained URDF packages.

Run with Blender's Python: ``blender -b -t 8 --python build_blender.py -- --output
DIR``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

from description import InstrumentDescription
from catalog import build_devices
from rights import LaboratoryAssetRights
from spec import DeviceSpec, Finish, JointKind, ShapeKind, ShapeSpec

# %% material and primitive authoring


@dataclass
class InstrumentBuilder:
    """
    Construct one independent device in an isolated Blender scene.
    """

    model: DeviceSpec
    """Dimensioned device and interaction targets."""
    output: Path
    """
    Portable instrument directory.
    """

    samples: int = 64
    """
    Cycles samples for each verification image.
    """

    width: int = 1200
    """
    Preview width in pixels.
    """

    materials: dict[Finish | ShapeKind, bpy.types.Material] = field(
        default_factory=dict
    )
    """
    Shared Principled surface materials.
    """

    objects: dict[str, list[bpy.types.Object]] = field(default_factory=dict)
    """
    Visual meshes owned by each local body name.
    """

    bodies: dict[str, bpy.types.Object] = field(default_factory=dict)
    """
    Editable empties following the URDF hierarchy.
    """

    def prepare(self) -> None:
        """
        Reset Blender and create the consistent laboratory material palette.
        """
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.preferences.filepaths.save_version = 0
        self.model.validate()
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "assets").mkdir(exist_ok=True)
        palette = {
            Finish.WHITE: ((0.78, 0.83, 0.82), 0.3, 0.08),
            Finish.TEAL: ((0.027, 0.145, 0.154), 0.29, 0.18),
            Finish.PALE_TEAL: ((0.33, 0.54, 0.52), 0.34, 0.12),
            Finish.STEEL: ((0.46, 0.52, 0.53), 0.3, 0.82),
            Finish.DARK: ((0.017, 0.025, 0.027), 0.5, 0),
            Finish.GLASS: ((0.68, 0.88, 0.9), 0.08, 0),
            Finish.SCREEN: ((0.009, 0.045, 0.049), 0.3, 0.05),
            Finish.INK: ((0.45, 0.85, 0.79), 0.48, 0),
            Finish.PAPER: ((0.9, 0.93, 0.91), 0.78, 0),
            Finish.BLUE: ((0.035, 0.23, 0.5), 0.34, 0.05),
            Finish.AMBER: ((0.8, 0.35, 0.04), 0.32, 0.05),
            Finish.RED: ((0.62, 0.055, 0.04), 0.4, 0.05),
        }
        for finish, (color, roughness, metalness) in palette.items():
            material = bpy.data.materials.new(finish.value)
            material.use_nodes = True
            material.diffuse_color = (*color, 1)
            shader = material.node_tree.nodes.get("Principled BSDF")
            shader.inputs["Base Color"].default_value = (*color, 1)
            shader.inputs["Roughness"].default_value = roughness
            shader.inputs["Metallic"].default_value = metalness
            if finish == Finish.GLASS:
                shader.inputs["Transmission Weight"].default_value = 0.82
                shader.inputs["IOR"].default_value = 1.46
            self.materials[finish] = material
        self._logo_material()

    def _logo_material(self) -> None:
        """
        Pack the existing transparent artwork, tinted to the laboratory teal.
        """
        path = (
            Path(__file__).resolve().parents[3] / "src/cramera/web/img/aicor-logo.png"
        )
        image = bpy.data.images.load(str(path), check_existing=True)
        image.pack()
        material = bpy.data.materials.new("AICOR printed teal artwork")
        material.use_nodes = True
        material.blend_method = "CLIP"
        material.alpha_threshold = 0.25
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = self.materials[
            Finish.TEAL
        ].diffuse_color
        shader.inputs["Roughness"].default_value = 0.5
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        material.node_tree.links.new(texture.outputs["Alpha"], shader.inputs["Alpha"])
        self.materials[ShapeKind.LOGO] = material

    def create_shape(self, shape: ShapeSpec) -> bpy.types.Object:
        """
        Construct one local surface, preserving hollow sleeves and thin markings.
        """
        if shape.kind == ShapeKind.BOX:
            bpy.ops.mesh.primitive_cube_add(size=1)
            obj = bpy.context.object
            obj.dimensions = shape.dimensions
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            if shape.bevel:
                bevel = obj.modifiers.new("Manufactured edge radius", "BEVEL")
                bevel.width = min(shape.bevel, min(shape.dimensions) * 0.3)
                bevel.segments = 5
                bpy.ops.object.modifier_apply(modifier=bevel.name)
                for polygon in obj.data.polygons:
                    polygon.use_smooth = True
                obj.data.use_auto_smooth = True
                normal = obj.modifiers.new(
                    "Weighted surface normals", "WEIGHTED_NORMAL"
                )
                normal.keep_sharp = True
                bpy.ops.object.modifier_apply(modifier=normal.name)
        elif shape.kind in (ShapeKind.CYLINDER, ShapeKind.CONE):
            if shape.kind == ShapeKind.CYLINDER:
                radius, depth = shape.dimensions
                bpy.ops.mesh.primitive_cylinder_add(
                    vertices=64, radius=radius, depth=depth
                )
            else:
                bottom, top, depth = shape.dimensions
                bpy.ops.mesh.primitive_cone_add(
                    vertices=64, radius1=bottom, radius2=top, depth=depth
                )
            obj = bpy.context.object
            for polygon in obj.data.polygons:
                polygon.use_smooth = len(polygon.vertices) == 4
        elif shape.kind == ShapeKind.SPHERE:
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=32, ring_count=16, radius=shape.dimensions[0]
            )
            obj = bpy.context.object
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
        elif shape.kind == ShapeKind.RING:
            obj = self._ring(shape)
        elif shape.kind == ShapeKind.TUBE:
            curve = bpy.data.curves.new(shape.name, "CURVE")
            curve.dimensions = "3D"
            curve.bevel_depth = shape.dimensions[0]
            curve.bevel_resolution = 3
            curve.resolution_u = 16
            spline = curve.splines.new("BEZIER")
            spline.bezier_points.add(len(shape.points) - 1)
            for point, coordinate in zip(spline.bezier_points, shape.points):
                point.co = coordinate
                point.handle_left_type = point.handle_right_type = "AUTO"
            obj = bpy.data.objects.new(shape.name, curve)
            bpy.context.collection.objects.link(obj)
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.convert(target="MESH")
            obj = bpy.context.object
        elif shape.kind == ShapeKind.LABEL:
            bpy.ops.object.text_add()
            obj = bpy.context.object
            obj.data.body = shape.text
            obj.data.align_x = "CENTER"
            obj.data.align_y = "CENTER"
            obj.data.size = shape.dimensions[0]
            obj.data.extrude = 0.000015
            obj.data.resolution_u = 8
            bpy.context.view_layer.update()
            if shape.dimensions[1] and obj.dimensions.x > shape.dimensions[1]:
                obj.scale *= shape.dimensions[1] / obj.dimensions.x
            bpy.ops.object.convert(target="MESH")
            obj = bpy.context.object
        else:
            image = next(
                node.image
                for node in self.materials[ShapeKind.LOGO].node_tree.nodes
                if node.type == "TEX_IMAGE"
            )
            bpy.ops.mesh.primitive_plane_add(size=1)
            obj = bpy.context.object
            obj.dimensions = (
                shape.dimensions[0],
                shape.dimensions[0] * image.size[1] / image.size[0],
                0,
            )
        obj.name = shape.name
        obj.location = shape.center
        obj.rotation_euler = shape.rotation
        obj.data.materials.clear()
        obj.data.materials.append(
            self.materials[
                ShapeKind.LOGO if shape.kind == ShapeKind.LOGO else shape.material
            ]
        )
        return obj

    @staticmethod
    def _ring(shape: ShapeSpec) -> bpy.types.Object:
        """
        Generate a closed annular wall with an unobstructed bore.
        """
        outer, inner, depth = shape.dimensions
        count = 64
        vertices = [
            (
                radius * math.cos(math.tau * index / count),
                radius * math.sin(math.tau * index / count),
                height,
            )
            for radius, height in (
                (outer, -depth / 2),
                (outer, depth / 2),
                (inner, -depth / 2),
                (inner, depth / 2),
            )
            for index in range(count)
        ]
        faces = []
        for index in range(count):
            following = (index + 1) % count
            faces.extend(
                (
                    (index, following, count + following, count + index),
                    (
                        2 * count + following,
                        2 * count + index,
                        3 * count + index,
                        3 * count + following,
                    ),
                    (
                        count + index,
                        count + following,
                        3 * count + following,
                        3 * count + index,
                    ),
                    (following, index, 2 * count + index, 2 * count + following),
                )
            )
        mesh = bpy.data.meshes.new(shape.name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        for index, polygon in enumerate(mesh.polygons):
            polygon.use_smooth = index % 4 < 2
        obj = bpy.data.objects.new(shape.name, mesh)
        bpy.context.collection.objects.link(obj)
        return obj

    def export_parts(self) -> None:
        """
        Export each link in its local Z-up frame before assembling the hierarchy.
        """
        for part in self.model.parts:
            bpy.ops.object.select_all(action="DESELECT")
            objects = []
            for shape in part.shapes:
                bpy.ops.object.select_all(action="DESELECT")
                objects.append(self.create_shape(shape))
            self.objects[part.name] = objects
            if objects:
                bpy.ops.object.select_all(action="DESELECT")
                for obj in objects:
                    obj.select_set(True)
                bpy.context.view_layer.objects.active = objects[0]
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                bpy.ops.export_scene.gltf(
                    filepath=str(self.output / "assets" / f"{part.name}.glb"),
                    export_format="GLB",
                    use_selection=True,
                    export_yup=False,
                    export_apply=True,
                    export_animations=False,
                    export_cameras=False,
                    export_lights=False,
                    export_materials="EXPORT",
                )
            body = bpy.data.objects.new(f"{self.model.identifier}_{part.name}", None)
            bpy.context.collection.objects.link(body)
            body.empty_display_size = 0.02
            body.location = part.origin
            body.rotation_euler = part.rotation
            if part.parent:
                body.parent = self.bodies[part.parent]
            body["joint_type"] = part.joint.value
            body["joint_axis"] = part.axis
            body["joint_lower"] = part.lower
            body["joint_upper"] = part.upper
            body["inspection_position"] = part.open_value
            for obj in objects:
                obj.parent = body
            self.bodies[part.name] = body
        for frame in self.model.frames:
            obj = bpy.data.objects.new(f"{self.model.identifier}_{frame.name}", None)
            bpy.context.collection.objects.link(obj)
            obj.parent = self.bodies[frame.parent]
            obj.location = frame.position
            obj.rotation_euler = frame.rotation
            obj.empty_display_type = "ARROWS"
            obj.empty_display_size = 0.015
            obj["interaction_role"] = frame.role
        for frame, opened in ((1, False), (60, True), (120, False)):
            self.pose(opened)
            for part in self.model.parts:
                body = self.bodies[part.name]
                body.keyframe_insert(data_path="location", frame=frame)
                body.keyframe_insert(data_path="rotation_euler", frame=frame)
        scene = bpy.context.scene
        scene.frame_start, scene.frame_end = 1, 120
        scene.frame_set(1)
        scene["principal_author"] = "Vanessa Hassouna"
        scene["institution"] = "AICOR Institute"
        scene["asset_license"] = "LicenseRef-AICOR-Laboratory-Assets"

    def pose(self, opened: bool) -> None:
        """
        Set a rest or inspection pose using the actual local joint axes.
        """
        for part in self.model.parts:
            body = self.bodies[part.name]
            rotation = Euler(part.rotation, "XYZ").to_matrix()
            value = part.open_value if opened else 0
            location = Vector(part.origin)
            if part.joint == JointKind.PRISMATIC:
                location += rotation @ Vector(part.axis) * value
            elif part.joint in (JointKind.REVOLUTE, JointKind.CONTINUOUS):
                rotation = rotation @ Matrix.Rotation(value, 3, Vector(part.axis))
            body.location = location
            body.rotation_euler = rotation.to_euler("XYZ")
        bpy.context.view_layer.update()

    def bounds(self) -> tuple[Vector, Vector]:
        """
        Measure only device geometry, excluding the photographic studio.
        """
        points = [
            obj.matrix_world @ Vector(corner)
            for objects in self.objects.values()
            for obj in objects
            for corner in obj.bound_box
        ]
        return Vector(
            tuple(min(point[axis] for point in points) for axis in range(3))
        ), Vector(tuple(max(point[axis] for point in points) for axis in range(3)))

    def studio(self) -> None:
        """
        Frame both articulation states in a neutral product-photography setup.
        """
        scene = bpy.context.scene
        scene.frame_set(1)
        lower, upper = self.bounds()
        scene.frame_set(60)
        opened_lower, opened_upper = self.bounds()
        lower = Vector(
            tuple(min(lower[index], opened_lower[index]) for index in range(3))
        )
        upper = Vector(
            tuple(max(upper[index], opened_upper[index]) for index in range(3))
        )
        scene.frame_set(1)
        target = (lower + upper) / 2
        span = max(upper - lower)
        scene.render.engine = "CYCLES"
        scene.cycles.samples = self.samples
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 6
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 8
        scene.render.resolution_x = self.width
        scene.render.resolution_y = round(self.width * 0.8)
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.world = bpy.data.worlds.new("Neutral studio")
        scene.world.use_nodes = True
        scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (
            0.78,
            0.83,
            0.86,
            1,
        )
        scene.world.node_tree.nodes["Background"].inputs[
            "Strength"
        ].default_value = 0.35
        scene.view_settings.view_transform = "AgX"
        camera_data = bpy.data.cameras.new("Product camera")
        camera = bpy.data.objects.new("Product camera", camera_data)
        scene.collection.objects.link(camera)
        camera.location = target + Vector((1.25, -1.8, 1.2)).normalized() * span * 2.35
        camera.rotation_euler = (
            (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        )
        camera_data.lens = 47
        camera_data.clip_start = 0.001
        camera_data.clip_end = 100
        scene.camera = camera
        for name, offset, energy, color in (
            ("Large softbox", (-1.4, -1.7, 2.4), 100, (1, 0.95, 0.88)),
            ("Cool fill", (1.8, -0.3, 1.2), 65, (0.84, 0.94, 1)),
            ("Rim softbox", (0.1, 1.4, 2), 120, (1, 1, 1)),
        ):
            data = bpy.data.lights.new(name, "AREA")
            data.energy = energy * span * span
            data.shape = "DISK"
            data.size = span * 1.8
            data.color = color
            obj = bpy.data.objects.new(name, data)
            scene.collection.objects.link(obj)
            obj.location = target + Vector(offset) * span
            obj.rotation_euler = (
                (target - obj.location).to_track_quat("-Z", "Y").to_euler()
            )
        bpy.ops.mesh.primitive_plane_add(
            size=span * 200, location=(0, 0, lower.z - 0.001)
        )
        floor = bpy.context.object
        floor.name = "STUDIO - excluded from model exports"
        floor.data.materials.append(self.materials[Finish.PAPER])
        scene.render.film_transparent = False

    def build(self, render: bool = True) -> None:
        """
        Package Blender, local meshes, URDF, notices and inspection renders.
        """
        self.prepare()
        self.export_parts()
        InstrumentDescription(self.model).write(self.output)
        self.studio()
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(self.output / "model.blend"))
        LaboratoryAssetRights().install(self.output)
        if render:
            for frame, filename in ((1, "preview.png"), (60, "preview_open.png")):
                if frame == 60 and not any(
                    part.open_value for part in self.model.parts
                ):
                    shutil.copyfile(self.output / "preview.png", self.output / filename)
                    continue
                bpy.context.scene.frame_set(frame)
                bpy.context.scene.render.filepath = str(self.output / filename)
                bpy.ops.render.render(write_still=True)
        print(
            json.dumps(
                {
                    "built": self.model.identifier,
                    "parts": len(self.model.parts),
                    "frames": len(self.model.frames),
                    "output": str(self.output),
                }
            ),
            flush=True,
        )


# %% selection and command line


def main() -> None:
    """
    Build all devices or a named subset into an existing resource catalog.
    """
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--output", type=Path, required=True)
    arguments.add_argument("--device", action="append")
    arguments.add_argument("--samples", type=int, default=64)
    arguments.add_argument("--width", type=int, default=1200)
    arguments.add_argument("--no-render", action="store_true")
    options = arguments.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    devices = build_devices()
    if options.device:
        devices = [model for model in devices if model.identifier in options.device]
        unknown = set(options.device) - {model.identifier for model in devices}
        if unknown:
            raise ValueError(f"Unknown instrument identifiers: {sorted(unknown)}")
    for model in devices:
        InstrumentBuilder(
            model, options.output / model.identifier, options.samples, options.width
        ).build(not options.no_render)


if __name__ == "__main__":
    main()
