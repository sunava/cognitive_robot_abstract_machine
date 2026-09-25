"""Export independent laboratory device URDFs and compact embedded-texture assets."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import struct
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path

from typing_extensions import Any

import numpy as np
import trimesh

from bundle import BalancePart
from geometry import RackDimensions
from rights import LaboratoryAssetRights
from cramera.laboratory_devices import (
    DeviceResourceFile,
    LaboratoryDevice,
    LaboratoryDeviceResource,
)
from cramera.laboratory_world import CabinetPart, CabinetStructure

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
    BOX = "box"
    """Axis-aligned contact primitive."""
    SIZE = "size"
    """Full extents of a contact box."""


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
    MATERIALS = "materials"
    """Shared material definitions retained from the source asset."""
    MATERIAL = "material"
    """Material index on one mesh primitive."""
    POSITION = "POSITION"
    """Vertex position accessor."""
    NORMAL = "NORMAL"
    """Vertex normal accessor."""
    COMPONENT_TYPE = "componentType"
    """Numeric storage format of an accessor."""
    COUNT = "count"
    """Number of accessor elements."""
    TYPE = "type"
    """Number of components per accessor element."""
    MINIMUM = "min"
    """Lower accessor bounds per component."""
    MAXIMUM = "max"
    """Upper accessor bounds per component."""


class GlbComponent(IntEnum):
    """Standard glTF numeric storage formats for appended geometry."""

    FLOAT = 5126
    """Single-precision positions and normals."""
    UNSIGNED_INT = 5125
    """Unsigned 32-bit triangle indices."""


class GlbShape(StrEnum):
    """Standard accessor dimensions for appended mesh attributes."""

    VECTOR = "VEC3"
    """Three spatial components per vertex."""
    SCALAR = "SCALAR"
    """One index per element."""


class CarrierArtwork(StrEnum):
    """Authored carrier and drawer meshes used by asset preparation."""

    ROOM = "environment.glb"
    """Stationary room asset with retained surface materials."""
    DRAWER = "drawer.glb"
    """Moving drawer body containing the original handle mesh."""
    HANDLE_PREFIX = "Rack carry"
    """Generated carry bar and mounting meshes."""
    HANDLE = "Rack carry handle"
    """Visible front carry bar."""
    LEFT_MOUNT = "Rack carry mounting"
    """First carry bar support."""
    RIGHT_MOUNT = "Rack carry mounting.001"
    """Second carry bar support."""
    HANDLE_COLLISION = "rack_handle"
    """Authored contact box determining the carry bar bounds."""
    LEFT_COLLISION = "rack_handle_mount_0"
    """First authored mounting contact box."""
    RIGHT_COLLISION = "rack_handle_mount_1"
    """Second authored mounting contact box."""
    DRAWER_HANDLE = "Drawer pull"
    """Bevelled handle used as the carry bar shape."""
    DRAWER_MOUNT = "Drawer pull mounting"
    """Bevelled support used as the carrier mounting shape."""
    MAT = "Drawer mat"
    """Decorative layer above the actual drawer floor."""
    DIVIDER = "Tray divider"
    """Interior separators occupying the sample carrier footprint."""
    GUIDE_PREFIX = "Rack lower guide"
    """Hollow lower sample guides with widened insertion mouths."""
    GUIDE_FINISH = "Pale teal powder coat"
    """Powder coat shared with the carrier's lower plate."""


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
        self,
        prefixes: tuple[str, ...],
        origin: tuple[float, ...],
        *,
        exclude: bool = False,
    ) -> StaticGlb:
        """Select baked room meshes and express them in a device-local frame."""
        document = copy.deepcopy(self.document)
        nodes = [
            node
            for node in document[GlbField.NODES]
            if node.get(GlbField.NAME, "").startswith(prefixes) != exclude
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

    def set_center(self, name: str, position: np.ndarray) -> None:
        """Place a baked mesh at a measured center without rewriting its vertices."""
        node = next(
            item
            for item in self.document[GlbField.NODES]
            if item[GlbField.NAME] == name
        )
        mesh = self.document[GlbField.MESHES][node[GlbField.MESH]]
        accessors = [
            self.document[GlbField.ACCESSORS][
                primitive[GlbField.ATTRIBUTES][GlbField.POSITION]
            ]
            for primitive in mesh[GlbField.PRIMITIVES]
        ]
        lower = np.min([item[GlbField.MINIMUM] for item in accessors], axis=0)
        upper = np.max([item[GlbField.MAXIMUM] for item in accessors], axis=0)
        node[GlbField.TRANSLATION] = (position - (lower + upper) / 2).tolist()

    def add_mesh(self, name: str, mesh: trimesh.Trimesh, material: str) -> None:
        """Append a static mesh while retaining all existing artwork buffers."""
        if any(
            node.get(GlbField.NAME) == name for node in self.document[GlbField.NODES]
        ):
            return
        material_index = next(
            index
            for index, item in enumerate(self.document[GlbField.MATERIALS])
            if item.get(GlbField.NAME) == material
        )
        positions = self.add_accessor(
            np.asarray(mesh.vertices, dtype="<f4"), GlbComponent.FLOAT, GlbShape.VECTOR
        )
        normals = self.add_accessor(
            np.asarray(mesh.vertex_normals, dtype="<f4"),
            GlbComponent.FLOAT,
            GlbShape.VECTOR,
        )
        indices = self.add_accessor(
            np.asarray(mesh.faces, dtype="<u4").reshape(-1),
            GlbComponent.UNSIGNED_INT,
            GlbShape.SCALAR,
        )
        self.document[GlbField.MESHES].append(
            {
                GlbField.NAME: name,
                GlbField.PRIMITIVES: [
                    {
                        GlbField.ATTRIBUTES: {
                            GlbField.POSITION: positions,
                            GlbField.NORMAL: normals,
                        },
                        GlbField.INDICES: indices,
                        GlbField.MATERIAL: material_index,
                    }
                ],
            }
        )
        self.document[GlbField.NODES].append(
            {
                GlbField.NAME: name,
                GlbField.MESH: len(self.document[GlbField.MESHES]) - 1,
            }
        )
        self.document[GlbField.SCENES][self.document.get(GlbField.SCENE, 0)][
            GlbField.NODES
        ].append(len(self.document[GlbField.NODES]) - 1)

    def add_accessor(
        self, values: np.ndarray, component_type: GlbComponent, shape: GlbShape
    ) -> int:
        """Append one aligned standard glTF numeric array and return its accessor."""
        padding = b"\0" * (-len(self.binary) % 4)
        offset = len(self.binary) + len(padding)
        content = values.tobytes()
        self.binary += padding + content
        views = self.document[GlbField.BUFFER_VIEWS]
        views.append(
            {
                GlbField.BUFFER: 0,
                GlbField.BYTE_OFFSET: offset,
                GlbField.BYTE_LENGTH: len(content),
            }
        )
        accessors = self.document[GlbField.ACCESSORS]
        bounds = values.reshape(len(values), -1)
        accessors.append(
            {
                GlbField.BUFFER_VIEW: len(views) - 1,
                GlbField.COMPONENT_TYPE: component_type,
                GlbField.COUNT: len(values),
                GlbField.TYPE: shape,
                GlbField.MINIMUM: bounds.min(axis=0).tolist(),
                GlbField.MAXIMUM: bounds.max(axis=0).tolist(),
            }
        )
        self.document[GlbField.BUFFERS][0][GlbField.BYTE_LENGTH] = len(self.binary)
        return len(accessors) - 1


# %% independent URDF subtrees


@dataclass
class DeviceResourceExporter:
    """Generate standalone instrument folders from the authored laboratory bundle."""

    source: Path
    """Laboratory bundle with URDF and material-bearing GLB assets."""
    output: Path
    """Parent folder receiving independently portable instrument resources."""

    def prepare_rack(self) -> None:
        """Separate the movable rack and align drawer artwork with contact geometry."""
        definition = next(
            item
            for item in DeviceDefinition.catalog()
            if item.device is LaboratoryDevice.TUBE_RACK
        )
        assets = self.source / DeviceResourceFile.ASSETS
        description = ElementTree.parse(self.source / ModelField.ENVIRONMENT)
        attachment = next(
            joint
            for joint in description.findall(ModelField.JOINT)
            if joint.find(ModelField.CHILD).attrib[ModelField.LINK]
            == definition.root_link
        )
        origin = tuple(
            float(value)
            for value in attachment.find(ModelField.ORIGIN)
            .attrib[ModelField.XYZ]
            .split()
        )
        room = StaticGlb.read(assets / CarrierArtwork.ROOM)
        if any(
            node.get(GlbField.NAME, "").startswith(definition.mesh_prefixes)
            for node in room.document[GlbField.NODES]
        ):
            rack = room.extract(definition.mesh_prefixes, origin)
            room = room.extract(definition.mesh_prefixes, (0, 0, 0), exclude=True)
        else:
            rack = StaticGlb.read(assets / definition.visual_filename)
        rack = rack.extract(
            (CarrierArtwork.HANDLE_PREFIX, CarrierArtwork.GUIDE_PREFIX),
            (0, 0, 0),
            exclude=True,
        )
        cross_member = next(
            panel
            for panel in CabinetStructure().panels()
            if panel.part is CabinetPart.CROSS_MEMBER
        )
        room.set_center(
            CabinetPart.CROSS_MEMBER.value, cross_member.origin.to_np()[:3, 3]
        )
        drawer = trimesh.load_scene(
            assets / CarrierArtwork.DRAWER, process=False, merge_primitives=True
        )
        for collision_name, visual_name, template in (
            (
                CarrierArtwork.HANDLE_COLLISION,
                CarrierArtwork.HANDLE,
                CarrierArtwork.DRAWER_HANDLE,
            ),
            (
                CarrierArtwork.LEFT_COLLISION,
                CarrierArtwork.LEFT_MOUNT,
                CarrierArtwork.DRAWER_MOUNT,
            ),
            (
                CarrierArtwork.RIGHT_COLLISION,
                CarrierArtwork.RIGHT_MOUNT,
                CarrierArtwork.DRAWER_MOUNT,
            ),
        ):
            collision = description.find(
                f"link[@name='{definition.root_link}']/collision[@name='{collision_name}']"
            )
            center = np.fromstring(
                collision.find(ModelField.ORIGIN).attrib[ModelField.XYZ], sep=" "
            )
            size = np.fromstring(
                collision.find(f"{ModelField.GEOMETRY}/{ModelField.BOX}").attrib[
                    ModelField.SIZE
                ],
                sep=" ",
            )
            transform, geometry = drawer.graph[template]
            mesh = drawer.geometry[geometry].copy()
            mesh.apply_transform(transform)
            mesh.apply_translation(-mesh.bounds.mean(axis=0))
            mesh.apply_scale(size / mesh.extents)
            mesh.apply_translation(center)
            rack.add_mesh(visual_name, mesh, ModelField.SUPPORT_FINISH)
        dimensions = RackDimensions()
        profile = np.array(
            [(point.radius, point.height) for point in dimensions.guide_profile()]
        )
        guide = trimesh.creation.revolve(profile, sections=96)
        for column, horizontal in enumerate(dimensions.columns):
            for row, depth in enumerate(dimensions.rows):
                mesh = guide.copy()
                mesh.apply_translation((horizontal, depth, 0))
                rack.add_mesh(
                    f"{CarrierArtwork.GUIDE_PREFIX} {column}_{row}",
                    mesh,
                    CarrierArtwork.GUIDE_FINISH,
                )
        rack.write(assets / definition.visual_filename)
        room.write(assets / CarrierArtwork.ROOM)
        StaticGlb.read(assets / CarrierArtwork.DRAWER).extract(
            (CarrierArtwork.MAT, CarrierArtwork.DIVIDER), (0, 0, 0), exclude=True
        ).write(assets / CarrierArtwork.DRAWER)

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
        LaboratoryAssetRights().install(self.output)

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
        link = root.find(f"link[@name='{definition.root_link}']")
        extracted = bool(
            definition.visual_filename and not link.findall(ModelField.VISUAL)
        )
        if extracted:
            room.extract(definition.mesh_prefixes, origin).write(
                assets / definition.visual_filename
            )
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
            if extracted and relative.name == definition.visual_filename:
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
        LaboratoryAssetRights().install(resource.directory)

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
    parser.add_argument("--source", type=Path, default=resources / "aicor_medical_lab")
    parser.add_argument("--output", type=Path, default=resources / "laboratory_devices")
    parser.add_argument(
        "--prepare-rack",
        action="store_true",
        help="Separate rack artwork and clear the drawer for its loaded carrier",
    )
    arguments = parser.parse_args()
    exporter = DeviceResourceExporter(arguments.source, arguments.output)
    if arguments.prepare_rack:
        exporter.prepare_rack()
    exporter.export()


if __name__ == "__main__":
    main()
