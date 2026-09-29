"""
Reopen saved instrument projects and compare their animation to exported URDFs.

Run with ``blender -b --python-exit-code 1 --python validate_blender.py -- --root
RESOURCE_DIRECTORY``. The command only reads saved packages and does not save or rebuild
any project.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ElementTree
from dataclasses import asdict, dataclass
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog import build_devices
from spec import JointKind, ShapeKind


# %% Saved Blender project validation
@dataclass(frozen=True)
class BlenderReport:
    """
    Read-back results from an independently reopened Blender project.
    """

    device: str
    """
    Identifier of the reopened resource.
    """

    bodies: int
    """
    Physical rigid-body empties with correctly parented editable meshes.
    """

    targets: int
    """
    Semantic interaction empties matching the exported URDF.
    """

    movable_bodies: int
    """
    Animated physical bodies checked at rest and inspection positions.
    """

    packed_images: int
    """
    File-backed textures embedded in the saved project.
    """

    checked_frames: tuple[int, ...]
    """
    Animation samples matched to the exported joint positions.
    """


@dataclass(frozen=True)
class BlenderValidator:
    """
    Check persisted Blender structure independently of the construction script.
    """

    directory: Path
    """
    Resource package containing model.blend and its exported descriptions.
    """

    def require(self, condition: bool, message: str) -> None:
        """
        Identify an invalid saved project with its owning device.
        """
        if not condition:
            raise ValueError(f"{self.directory.name}: {message}")

    @staticmethod
    def values(text: str) -> tuple[float, ...]:
        """
        Read spatial coordinates from an exported URDF attribute.
        """
        return tuple(float(value) for value in text.split())

    @staticmethod
    def close(left: Matrix, right: Matrix) -> bool:
        """
        Compare local transforms with tolerance for Blender single precision.
        """
        return all(
            abs(left[row][column] - right[row][column]) < 2e-6
            for row in range(4)
            for column in range(4)
        )

    def expected_pose(self, joint: ElementTree.Element, position: float) -> Matrix:
        """
        Compute the URDF origin followed by the exported axis motion.
        """
        origin = joint.find("origin")
        translation = Vector(self.values(origin.get("xyz", "0 0 0")))
        rotation = (
            Euler(self.values(origin.get("rpy", "0 0 0")), "XYZ").to_matrix().to_4x4()
        )
        transform = Matrix.Translation(translation) @ rotation
        kind = joint.get("type")
        if kind == JointKind.FIXED:
            return transform
        axis = Vector(self.values(joint.find("axis").get("xyz")))
        if kind == JointKind.PRISMATIC:
            return transform @ Matrix.Translation(axis * position)
        return transform @ Matrix.Rotation(position, 4, axis)

    def validate(self) -> BlenderReport:
        """
        Reopen a project and verify hierarchy, packed artwork and joint animation.
        """
        metadata = json.loads((self.directory / "metadata.json").read_text())
        definition = json.loads((self.directory / "definition.json").read_text())
        root = ElementTree.parse(self.directory / "model.urdf").getroot()
        connections = {
            joint.find("child").get("link"): joint for joint in root.findall("joint")
        }
        inspections = {
            joint["name"]: joint["inspectionPosition"] for joint in metadata["joints"]
        }
        bpy.ops.wm.open_mainfile(
            filepath=str((self.directory / metadata["blender"]).resolve())
        )
        prefix = metadata["device"] + "_"
        physical_names = {prefix + part["name"] for part in definition["parts"]}
        target_names = {frame["name"] for frame in metadata["frames"]}
        for name in physical_names | target_names:
            self.require(name in bpy.data.objects, f"missing body or target {name}")
            obj = bpy.data.objects[name]
            self.require(obj.type == "EMPTY", f"frame is not an editable empty: {name}")
            if name == metadata["rootLink"]:
                self.require(obj.parent is None, "root body has an external parent")
            else:
                self.require(
                    obj.parent is not None
                    and obj.parent.name == connections[name].find("parent").get("link"),
                    f"wrong parent of {name}",
                )
        for part in definition["parts"]:
            body = bpy.data.objects[prefix + part["name"]]
            meshes = [child for child in body.children if child.type == "MESH"]
            self.require(
                len(meshes) == len(part["shapes"]),
                f"lost editable surfaces under {body.name}",
            )
            for mesh in meshes:
                self.require(
                    self.close(mesh.matrix_basis, Matrix.Identity(4)),
                    f"unapplied exported mesh coordinates on {mesh.name}",
                )
                self.require(
                    self.close(mesh.matrix_parent_inverse, Matrix.Identity(4)),
                    f"unexpected parent inverse on {mesh.name}",
                )
        for obj in bpy.context.scene.objects:
            if obj.type in ("LIGHT", "CAMERA"):
                self.require(
                    obj.parent is None, "studio equipment belongs to a physical body"
                )
            elif obj.type == "MESH" and obj.parent is None:
                self.require(
                    obj.name.startswith("STUDIO -"),
                    f"unattached instrument mesh {obj.name}",
                )
        images = [image for image in bpy.data.images if image.source == "FILE"]
        for image in images:
            self.require(
                image.packed_file is not None or bool(image.packed_files),
                f"unpacked image dependency {image.name}",
            )
            self.require(
                image.size[0] > 0 and image.size[1] > 0,
                f"unreadable packed image {image.name}",
            )
        authored_logo = any(
            shape["kind"] == ShapeKind.LOGO
            for part in definition["parts"]
            for shape in part["shapes"]
        )
        if authored_logo:
            self.require(bool(images), "missing AICOR artwork")
            logo = bpy.data.materials.get("AICOR printed teal artwork")
            self.require(
                logo is not None and logo.use_nodes, "missing printed-logo material"
            )
            alpha_links = [
                link for link in logo.node_tree.links if link.to_socket.name == "Alpha"
            ]
            self.require(bool(alpha_links), "logo transparency is not connected")
        for frame, inspection in ((1, False), (60, True), (120, False)):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            for name in physical_names | target_names:
                if name == metadata["rootLink"]:
                    continue
                joint = connections[name]
                value = inspections.get(joint.get("name"), 0) if inspection else 0
                expected = self.expected_pose(joint, value)
                self.require(
                    self.close(bpy.data.objects[name].matrix_local, expected),
                    f"animation disagrees with URDF at frame {frame}: {name}",
                )
        return BlenderReport(
            metadata["device"],
            len(physical_names),
            len(target_names),
            len(inspections),
            len(images),
            (1, 60, 120),
        )


# %% Read-only command line


def main() -> None:
    """
    Read back selected or all saved projects without altering their files.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", action="append")
    parser.add_argument("--report", type=Path)
    options = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    known = {model.identifier for model in build_devices()}
    requested = set(options.device or known)
    if requested - known:
        raise ValueError(f"Unknown devices: {sorted(requested - known)}")
    reports = [
        BlenderValidator(options.root / name).validate() for name in sorted(requested)
    ]
    content = json.dumps([asdict(report) for report in reports], indent=2) + "\n"
    if options.report:
        options.report.write_text(content)
    print(content, flush=True)


if __name__ == "__main__":
    main()
