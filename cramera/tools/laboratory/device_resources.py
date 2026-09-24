"""Export independent laboratory device URDFs and compact embedded-texture assets."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import struct
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any

from bundle import BalancePart
from cramera.laboratory_devices import (
    DeviceResourceFile,
    LaboratoryDevice,
    LaboratoryDeviceResource,
)

# %% external document vocabulary


class ModelField(StrEnum):
    """URDF tags and attributes relevant to extracting a fixed device subtree."""

    ROBOT = "robot"
    LINK = "link"
    JOINT = "joint"
    PARENT = "parent"
    CHILD = "child"
    ORIGIN = "origin"
    XYZ = "xyz"
    NAME = "name"
    VISUAL = "visual"
    GEOMETRY = "geometry"
    MESH = "mesh"
    FILENAME = "filename"
    ENVIRONMENT = "environment.urdf"
    COLLISION = "collision"
    MATERIAL = "material"
    COLOR = "color"
    SUPPORT_FINISH = "Satin stainless steel"


class GlbField(StrEnum):
    """Standard glTF fields required by static authored mesh extraction."""

    NODES = "nodes"
    MESH = "mesh"
    NAME = "name"
    MESHES = "meshes"
    PRIMITIVES = "primitives"
    ATTRIBUTES = "attributes"
    INDICES = "indices"
    ACCESSORS = "accessors"
    BUFFER_VIEW = "bufferView"
    BUFFER_VIEWS = "bufferViews"
    BUFFER = "buffer"
    BUFFERS = "buffers"
    BYTE_OFFSET = "byteOffset"
    BYTE_LENGTH = "byteLength"
    IMAGES = "images"
    TRANSLATION = "translation"
    SCENES = "scenes"
    SCENE = "scene"
    CHILDREN = "children"
    MATRIX = "matrix"
    ROTATION = "rotation"
    SCALE = "scale"
    SPARSE = "sparse"
    URI = "uri"


@dataclass(frozen=True)
class DeviceDefinition:
    """Selection of an authored rigid-body subtree and its embedded room meshes."""

    device: LaboratoryDevice
    """Portable catalog identifier."""
    root_link: str
    """Existing root body whose laboratory attachment is removed."""
    mesh_prefixes: tuple[str, ...] = ()
    """Room mesh prefixes to extract when the source root has no visual."""
    visual_filename: str | None = None
    """Filename for the extracted device-local room meshes."""

    @classmethod
    def catalog(cls) -> tuple[DeviceDefinition, ...]:
        """Return stable device selections matching the authored laboratory bodies."""
        return (
            cls(LaboratoryDevice.CENTRIFUGE, "laboratory_centrifuge_base"),
            cls(LaboratoryDevice.DUAL_DISPENSER, "laboratory_pipetting_station"),
            cls(LaboratoryDevice.REFRIGERATOR, "laboratory_refrigerator_base"),
            cls(
                LaboratoryDevice.SCALE,
                "laboratory_balance",
                ("Instrument base", "Balance", "Scale"),
                "scale.glb",
            ),
            cls(
                LaboratoryDevice.TUBE_RACK,
                "laboratory_rack",
                ("Rack", "Socket identifier", "Tube rack"),
                "tube_rack.glb",
            ),
        )


@dataclass
class StaticGlb:
    """A static glTF document with one embedded binary buffer."""

    document: dict[str, Any]
    """Decoded glTF JSON using the standard schema."""
    binary: bytes
    """Embedded vertex, index, and image bytes."""

    @classmethod
    def read(cls, path: Path) -> StaticGlb:
        """Read a version-two binary glTF without modifying material definitions."""
        content = path.read_bytes()
        magic, version, length = struct.unpack_from("<4sII", content)
        if magic != b"glTF" or version != 2 or length != len(content):
            raise ValueError(f"Unsupported binary glTF: {path}")
        json_length, json_type = struct.unpack_from("<I4s", content, 12)
        if json_type != b"JSON":
            raise ValueError(f"Missing glTF JSON chunk: {path}")
        document = json.loads(content[20 : 20 + json_length])
        offset = 20 + json_length
        binary_length, binary_type = struct.unpack_from("<I4s", content, offset)
        if binary_type != b"BIN\0":
            raise ValueError(f"Missing embedded glTF buffer: {path}")
        return cls(document, content[offset + 8 : offset + 8 + binary_length])

    def write(self, path: Path) -> None:
        """Write aligned glTF chunks with the original embedded material images."""
        encoded = json.dumps(self.document, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 4)
        binary = self.binary + b"\0" * (-len(self.binary) % 4)
        length = 12 + 8 + len(encoded) + 8 + len(binary)
        path.write_bytes(
            struct.pack("<4sII", b"glTF", 2, length)
            + struct.pack("<I4s", len(encoded), b"JSON")
            + encoded
            + struct.pack("<I4s", len(binary), b"BIN\0")
            + binary
        )

    def extract(
        self, prefixes: tuple[str, ...], origin: tuple[float, ...]
    ) -> StaticGlb:
        """Select baked room meshes and express them in a device-local frame."""
        document = copy.deepcopy(self.document)
        nodes = [
            node
            for node in document[GlbField.NODES]
            if node.get(GlbField.NAME, "").startswith(prefixes)
        ]
        if not nodes:
            raise ValueError(f"No instrument meshes match {prefixes}")
        for node in nodes:
            if any(
                field in node
                for field in (
                    GlbField.CHILDREN,
                    GlbField.MATRIX,
                    GlbField.ROTATION,
                    GlbField.SCALE,
                )
            ):
                raise ValueError(
                    "Instrument extraction requires baked static mesh transforms"
                )
            translation = node.get(GlbField.TRANSLATION, (0.0, 0.0, 0.0))
            node[GlbField.TRANSLATION] = [
                translation[axis] - origin[axis] for axis in range(3)
            ]
        mesh_indices = sorted({node[GlbField.MESH] for node in nodes})
        meshes = [document[GlbField.MESHES][index] for index in mesh_indices]
        for node in nodes:
            node[GlbField.MESH] = mesh_indices.index(node[GlbField.MESH])
        accessor_indices = sorted(
            {
                index
                for mesh in meshes
                for primitive in mesh[GlbField.PRIMITIVES]
                for index in (
                    *primitive[GlbField.ATTRIBUTES].values(),
                    primitive[GlbField.INDICES],
                )
            }
        )
        accessors = [document[GlbField.ACCESSORS][index] for index in accessor_indices]
        for mesh in meshes:
            for primitive in mesh[GlbField.PRIMITIVES]:
                primitive[GlbField.INDICES] = accessor_indices.index(
                    primitive[GlbField.INDICES]
                )
                primitive[GlbField.ATTRIBUTES] = {
                    name: accessor_indices.index(index)
                    for name, index in primitive[GlbField.ATTRIBUTES].items()
                }
        for accessor in accessors:
            if GlbField.SPARSE in accessor:
                raise ValueError(
                    "Sparse glTF accessors are not part of the authored device format"
                )
        images = document.get(GlbField.IMAGES, [])
        if any(GlbField.URI in image for image in images):
            raise ValueError("Portable instrument artwork must be embedded")
        view_indices = sorted(
            {item[GlbField.BUFFER_VIEW] for item in (*accessors, *images)}
        )
        views = [document[GlbField.BUFFER_VIEWS][index] for index in view_indices]
        binary = bytearray()
        for view in views:
            start = view.get(GlbField.BYTE_OFFSET, 0)
            length = view[GlbField.BYTE_LENGTH]
            binary.extend(b"\0" * (-len(binary) % 4))
            view[GlbField.BUFFER] = 0
            view[GlbField.BYTE_OFFSET] = len(binary)
            binary.extend(self.binary[start : start + length])
        for item in (*accessors, *images):
            item[GlbField.BUFFER_VIEW] = view_indices.index(item[GlbField.BUFFER_VIEW])
        document[GlbField.NODES] = nodes
        document[GlbField.MESHES] = meshes
        document[GlbField.ACCESSORS] = accessors
        document[GlbField.BUFFER_VIEWS] = views
        document[GlbField.BUFFERS] = [{GlbField.BYTE_LENGTH: len(binary)}]
        document[GlbField.SCENES] = [{GlbField.NODES: list(range(len(nodes)))}]
        document[GlbField.SCENE] = 0
        return StaticGlb(document, bytes(binary))


# %% independent URDF subtrees


@dataclass
class DeviceResourceExporter:
    """Generate standalone instrument folders from the authored laboratory bundle."""

    source: Path
    """Laboratory bundle with URDF and material-bearing GLB assets."""
    output: Path
    """Parent folder receiving independently portable instrument resources."""

    def export(self) -> None:
        """Copy device subtrees while keeping all local collision and joint values."""
        source = ElementTree.parse(self.source / ModelField.ENVIRONMENT)
        room = StaticGlb.read(
            self.source / DeviceResourceFile.ASSETS / "environment.glb"
        )
        for definition in DeviceDefinition.catalog():
            self.export_device(source, room, definition)
        shutil.copyfile(
            Path(__file__).with_name("device_resources_readme.md"),
            self.output / DeviceResourceFile.README,
        )

    def export_device(
        self,
        source: ElementTree.ElementTree,
        room: StaticGlb,
        definition: DeviceDefinition,
    ) -> None:
        """Detach one root from the bench and retain all of its articulated children."""
        resource = LaboratoryDeviceResource(definition.device, self.output)
        assets = resource.directory / DeviceResourceFile.ASSETS
        assets.mkdir(parents=True, exist_ok=True)
        root = ElementTree.Element(ModelField.ROBOT, name=definition.device.value)
        names = {definition.root_link}
        selected_joints = []
        remaining = list(source.findall(ModelField.JOINT))
        while remaining:
            attached = [
                joint
                for joint in remaining
                if joint.find(ModelField.PARENT).attrib[ModelField.LINK] in names
            ]
            if not attached:
                break
            for joint in attached:
                names.add(joint.find(ModelField.CHILD).attrib[ModelField.LINK])
                selected_joints.append(copy.deepcopy(joint))
                remaining.remove(joint)
        for link in source.findall(ModelField.LINK):
            if link.attrib[ModelField.NAME] in names:
                root.append(copy.deepcopy(link))
        root.extend(selected_joints)
        attachment = next(
            joint
            for joint in source.findall(ModelField.JOINT)
            if joint.find(ModelField.CHILD).attrib[ModelField.LINK]
            == definition.root_link
        )
        origin = tuple(
            float(value)
            for value in attachment.find(ModelField.ORIGIN)
            .attrib[ModelField.XYZ]
            .split()
        )
        if definition.visual_filename:
            room.extract(definition.mesh_prefixes, origin).write(
                assets / definition.visual_filename
            )
            link = root.find(f"link[@name='{definition.root_link}']")
            visual = ElementTree.SubElement(link, ModelField.VISUAL)
            geometry = ElementTree.SubElement(visual, ModelField.GEOMETRY)
            ElementTree.SubElement(
                geometry,
                ModelField.MESH,
                filename=f"{DeviceResourceFile.ASSETS}/{definition.visual_filename}",
            )
        if definition.device is LaboratoryDevice.SCALE:
            self.add_scale_support(root)
        for mesh in root.findall(".//mesh"):
            relative = Path(mesh.attrib[ModelField.FILENAME])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Device assets must be relative: {relative}")
            destination = resource.directory / relative
            if (
                definition.visual_filename
                and relative.name == definition.visual_filename
            ):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.source / relative, destination)
        ElementTree.indent(root)
        ElementTree.ElementTree(root).write(
            resource.urdf, encoding="utf-8", xml_declaration=True
        )
        metadata = {
            "device": definition.device,
            "rootLink": definition.root_link,
            "units": "metres",
            "upAxis": "Z",
            "laboratoryOrigin": origin,
            "joints": [joint.attrib[ModelField.NAME] for joint in selected_joints],
        }
        (resource.directory / DeviceResourceFile.METADATA).write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        shutil.copyfile(
            Path(__file__).with_name("device_descriptions") / f"{definition.device}.md",
            resource.directory / DeviceResourceFile.README,
        )

    @staticmethod
    def add_scale_support(root: ElementTree.Element) -> None:
        """Include visible collar segments matching the scale's contact surfaces."""
        link = root.find(f"link[@name='{BalancePart.BODY}']")
        for collision in link.findall(ModelField.COLLISION):
            name = collision.attrib[ModelField.NAME]
            if not name.startswith(f"{BalancePart.HOLDER}_"):
                continue
            visual = ElementTree.SubElement(link, ModelField.VISUAL, name=name)
            visual.append(copy.deepcopy(collision.find(ModelField.ORIGIN)))
            visual.append(copy.deepcopy(collision.find(ModelField.GEOMETRY)))
            material = ElementTree.SubElement(
                visual, ModelField.MATERIAL, name=ModelField.SUPPORT_FINISH
            )
            ElementTree.SubElement(
                material, ModelField.COLOR, rgba="0.721568627 0.768627451 0.768627451 1"
            )


def main() -> None:
    """Export reusable models from an authored laboratory bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    resources = Path(__file__).resolve().parents[2] / "src/cramera/resources"
    parser.add_argument("--source", type=Path, default=resources / "precision_lab")
    parser.add_argument("--output", type=Path, default=resources / "laboratory_devices")
    arguments = parser.parse_args()
    DeviceResourceExporter(arguments.source, arguments.output).export()


if __name__ == "__main__":
    main()
