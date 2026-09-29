"""
Render and save the assembled expanded laboratory from its portable URDFs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


# %% Native metric URDF assembly
@dataclass
class WorldPreview:
    """
    Assemble exported visual meshes at their actual joint and free-object poses.
    """

    directory: Path
    """Self-contained expanded laboratory bundle."""
    samples: int = 96
    """
    Cycles sampling quality for the overview.
    """

    width: int = 1600
    """
    Output preview width in pixels.
    """

    @staticmethod
    def values(text: str) -> tuple[float, ...]:
        """
        Read one serialized metric URDF vector.
        """
        return tuple(float(value) for value in text.split())

    def origin(self, element: ElementTree.Element | None) -> Matrix:
        """
        Convert an optional URDF origin to a homogeneous transform.
        """
        if element is None:
            return Matrix.Identity(4)
        position = self.values(element.get("xyz", "0 0 0"))
        rotation = (
            Euler(self.values(element.get("rpy", "0 0 0")), "XYZ").to_matrix().to_4x4()
        )
        return Matrix.Translation(Vector(position)) @ rotation

    def model(self, path: Path, joints: dict[str, float], placement: Matrix) -> None:
        """
        Read one complete URDF with metric Z-up exported mesh coordinates.
        """
        root = ElementTree.parse(path).getroot()
        bodies = {}
        children = {joint.find("child").get("link") for joint in root.findall("joint")}
        for link in root.findall("link"):
            name = link.get("name")
            body = bpy.data.objects.new(name, None)
            bpy.context.collection.objects.link(body)
            body.empty_display_size = 0.03
            body["urdf_link"] = name
            bodies[name] = body
            if name not in children:
                body.matrix_world = placement
            for visual in link.findall("visual"):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    continue
                existing = set(bpy.data.objects)
                bpy.ops.import_scene.gltf(
                    filepath=str((path.parent / mesh.get("filename")).resolve())
                )
                imported = set(bpy.data.objects) - existing
                correction = Matrix.Rotation(-math.pi / 2, 4, "X")
                transform = self.origin(visual.find("origin"))
                for obj in imported:
                    if obj.parent is None:
                        local = correction @ obj.matrix_world
                        obj.parent = body
                        obj.matrix_parent_inverse = Matrix.Identity(4)
                        obj.matrix_basis = transform @ local
        for joint in root.findall("joint"):
            parent, child = joint.find("parent").get("link"), joint.find("child").get(
                "link"
            )
            transform = self.origin(joint.find("origin"))
            kind = joint.get("type")
            position = joints.get(joint.get("name"), 0)
            if kind in ("prismatic", "revolute", "continuous"):
                axis = Vector(self.values(joint.find("axis").get("xyz")))
                if kind == "prismatic":
                    transform @= Matrix.Translation(axis * position)
                else:
                    transform @= Matrix.Rotation(position, 4, axis)
            body = bodies[child]
            body.parent = bodies[parent]
            body.matrix_parent_inverse = Matrix.Identity(4)
            body.matrix_basis = transform
        bpy.context.view_layer.update()

    def build(self) -> None:
        """
        Save a packed editable assembled world and its overview image.
        """
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.preferences.filepaths.save_version = 0
        manifest = json.loads((self.directory / "scene.json").read_text())
        trajectory = json.loads((self.directory / "trajectory.json").read_text())
        joints = trajectory["frames"][0]
        self.model(self.directory / "environment.urdf", joints, Matrix.Identity(4))
        for item in manifest["objects"]:
            position, orientation = item["spawn"][:3], item["spawn"][3:]
            quaternion = Quaternion((orientation[3], *orientation[:3]))
            transform = (
                Matrix.Translation(Vector(position)) @ quaternion.to_matrix().to_4x4()
            )
            self.model(self.directory / item["urdf"], joints, transform)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.samples = self.samples
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 6
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 8
        scene.render.resolution_x = self.width
        scene.render.resolution_y = round(self.width * 0.67)
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.world = bpy.data.worlds.new("Laboratory ambient")
        scene.world.use_nodes = True
        background = scene.world.node_tree.nodes["Background"]
        background.inputs["Color"].default_value = (0.70, 0.76, 0.80, 1)
        background.inputs["Strength"].default_value = 0.55
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.7
        camera_data = bpy.data.cameras.new("Laboratory overview")
        camera_data.lens = 34
        camera_data.clip_start = 0.01
        camera_data.clip_end = 100
        camera = bpy.data.objects.new("Laboratory overview", camera_data)
        scene.collection.objects.link(camera)
        camera.location = (2.80, -5.2, 4.0)
        target = Vector((0, 0.9, 1.0))
        camera.rotation_euler = (
            (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        )
        scene.camera = camera
        for name, position, power, size in (
            ("Front softbox", (0, -2.5, 5), 1400, 5),
            ("Rear workbank lighting", (0, 1.8, 2.65), 700, 5),
            ("Left fill", (-2.8, -0.5, 3.8), 600, 3),
        ):
            data = bpy.data.lights.new(name, "AREA")
            data.energy = power
            data.shape = "DISK"
            data.size = size
            light = bpy.data.objects.new(name, data)
            scene.collection.objects.link(light)
            light.location = position
            light.rotation_euler = (
                (target - light.location).to_track_quat("-Z", "Y").to_euler()
            )
        scene["principal_author"] = "Vanessa Hassouna"
        scene["institution"] = "AICOR Institute"
        scene["asset_license"] = "LicenseRef-AICOR-Laboratory-Assets"
        scene["coordinate_system"] = (
            "metres; Z up; calibrated preparation poses preserved"
        )
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(
            filepath=str(self.directory / "laboratory_large.blend")
        )
        scene.render.filepath = str(self.directory / "preview.png")
        bpy.ops.render.render(write_still=True)


# %% Command line


def main() -> None:
    """
    Render an expanded portable world without changing its URDF files.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--width", type=int, default=1600)
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    WorldPreview(options.directory, options.samples, options.width).build()


if __name__ == "__main__":
    main()
