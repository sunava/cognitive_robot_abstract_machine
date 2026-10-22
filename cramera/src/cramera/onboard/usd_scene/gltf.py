"""
A binary glTF 2.0 (GLB) writer for the visual geometry of one link: meshes merged per
material, physically based materials with their extensions, and PNG or JPEG textures
embedded in the binary chunk.
"""

from __future__ import annotations

import io
import json
import os
import struct
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path

import numpy
from PIL import Image
from typing_extensions import Any, Self

from cramera.onboard.usd_scene.materials import (
    AlphaMode,
    PhysicallyBasedMaterial,
    RoughnessMetallicTexture,
    TextureTransform,
    TransmissionBlending,
)
from cramera.onboard.usd_scene.scene_description import (
    ImportedLink,
    ImportedMesh,
    TextureFile,
)

# %% format constants


class ComponentType(IntEnum):
    """
    The glTF accessor component types the writer uses.
    """

    UNSIGNED_SHORT = 5123
    """
    16-bit unsigned integer indices.
    """

    UNSIGNED_INT = 5125
    """
    32-bit unsigned integer indices.
    """

    FLOAT = 5126
    """
    32-bit float attributes.
    """

    @classmethod
    def indices_for(cls, vertex_count: int) -> ComponentType:
        """
        The smallest index type addressing a number of vertices; glTF reserves the
        largest value of each type, so it never indexes a vertex.

        :param vertex_count: The number of vertices.
        """
        if vertex_count <= numpy.iinfo(numpy.uint16).max:
            return cls.UNSIGNED_SHORT
        return cls.UNSIGNED_INT

    @property
    def numpy_type(self) -> type[numpy.generic]:
        """
        The numpy type holding one component.
        """
        return {
            ComponentType.UNSIGNED_SHORT: numpy.uint16,
            ComponentType.UNSIGNED_INT: numpy.uint32,
            ComponentType.FLOAT: numpy.float32,
        }[self]


class BufferTarget(IntEnum):
    """
    What a buffer view holds, as WebGL binds it.
    """

    ARRAY_BUFFER = 34962
    """
    Vertex attributes.
    """

    ELEMENT_ARRAY_BUFFER = 34963
    """
    Triangle indices.
    """


class SamplerSetting(IntEnum):
    """
    The texture sampler settings the writer uses.
    """

    LINEAR = 9729
    """
    Linear magnification.
    """

    LINEAR_MIPMAP_LINEAR = 9987
    """
    Trilinear minification.
    """

    REPEAT = 10497
    """
    Tiling outside the unit square, which baked-in tiling relies on.
    """


class GlbChunk(IntEnum):
    """
    The magic numbers of the GLB container.
    """

    MAGIC = 0x46546C67
    """
    ``glTF`` in little endian, at the start of the file.
    """

    VERSION = 2
    """
    The container version.
    """

    JSON = 0x4E4F534A
    """
    The type of the JSON chunk.
    """

    BINARY = 0x004E4942
    """
    The type of the binary chunk.
    """


class GltfExtension(StrEnum):
    """
    The glTF extensions the writer may use.
    """

    INDEX_OF_REFRACTION = "KHR_materials_ior"
    """
    The index of refraction.
    """

    CLEARCOAT = "KHR_materials_clearcoat"
    """
    A clear coat layer.
    """

    EMISSIVE_STRENGTH = "KHR_materials_emissive_strength"
    """
    Emission brighter than the unit range.
    """

    ANISOTROPY = "KHR_materials_anisotropy"
    """
    Anisotropic reflection.
    """

    TEXTURE_TRANSFORM = "KHR_texture_transform"
    """
    Offset and scale of texture coordinates.
    """


class AttributeName(StrEnum):
    """
    The vertex attributes the writer emits.
    """

    POSITION = "POSITION"
    """
    Vertex positions.
    """

    NORMAL = "NORMAL"
    """
    Vertex normals.
    """

    TEXTURE_COORDINATES = "TEXCOORD_0"
    """
    The first texture coordinate set.
    """


class ImageType(StrEnum):
    """
    The image encodings glTF allows without extensions.
    """

    PNG = "image/png"
    """
    PNG images.
    """

    JPEG = "image/jpeg"
    """
    JPEG images.
    """


class GltfKey(StrEnum):
    """
    The property names of the glTF document structure the writer emits, as glTF 2.0
    names them.
    """

    ASSET = "asset"
    """
    The document's metadata.
    """

    VERSION = "version"
    """
    The glTF version the document follows.
    """

    GENERATOR = "generator"
    """
    The tool that wrote the document.
    """

    SCENE = "scene"
    """
    The index of the scene shown by default.
    """

    SCENES = "scenes"
    """
    The scenes, each a list of root nodes.
    """

    NODES = "nodes"
    """
    The nodes; in a scene, the indices of its root nodes.
    """

    MESHES = "meshes"
    """
    The meshes.
    """

    MATERIALS = "materials"
    """
    The materials.
    """

    ACCESSORS = "accessors"
    """
    The typed views of buffer data.
    """

    BUFFER_VIEWS = "bufferViews"
    """
    The byte ranges of the buffers.
    """

    BUFFERS = "buffers"
    """
    The binary buffers.
    """

    SAMPLERS = "samplers"
    """
    The texture samplers.
    """

    IMAGES = "images"
    """
    The embedded images.
    """

    TEXTURES = "textures"
    """
    The textures, each an image with a sampler.
    """

    EXTENSIONS_USED = "extensionsUsed"
    """
    The extensions the document uses.
    """

    EXTENSIONS = "extensions"
    """
    The extension objects of a property.
    """

    EXTRAS = "extras"
    """
    Application data stored on a property.
    """

    NAME = "name"
    """
    The name of a node, mesh or material.
    """

    MAGNIFICATION_FILTER = "magFilter"
    """
    How a sampler magnifies a texture.
    """

    MINIFICATION_FILTER = "minFilter"
    """
    How a sampler minifies a texture.
    """

    WRAP_S = "wrapS"
    """
    How a sampler wraps the first texture coordinate.
    """

    WRAP_T = "wrapT"
    """
    How a sampler wraps the second texture coordinate.
    """

    BUFFER = "buffer"
    """
    The buffer a buffer view lies in.
    """

    BYTE_OFFSET = "byteOffset"
    """
    Where a buffer view starts in its buffer.
    """

    BYTE_LENGTH = "byteLength"
    """
    The number of bytes of a buffer or buffer view.
    """

    TARGET = "target"
    """
    What a buffer view holds, as WebGL binds it.
    """

    BUFFER_VIEW = "bufferView"
    """
    The buffer view an accessor or image reads.
    """

    COMPONENT_TYPE = "componentType"
    """
    The type of an accessor's components.
    """

    COUNT = "count"
    """
    The number of elements of an accessor.
    """

    TYPE = "type"
    """
    Whether an accessor's elements are scalars or vectors.
    """

    MINIMUM = "min"
    """
    The smallest value of each component of an accessor.
    """

    MAXIMUM = "max"
    """
    The largest value of each component of an accessor.
    """

    MIME_TYPE = "mimeType"
    """
    The encoding of an embedded image.
    """

    SOURCE = "source"
    """
    The image of a texture.
    """

    SAMPLER = "sampler"
    """
    The sampler of a texture.
    """

    INDEX = "index"
    """
    The texture a texture reference points at.
    """

    OFFSET = "offset"
    """
    The shift of texture coordinates.
    """

    SCALE = "scale"
    """
    The scale of texture coordinates, or the strength of a normal map.
    """

    PRIMITIVES = "primitives"
    """
    The drawable parts of a mesh.
    """

    ATTRIBUTES = "attributes"
    """
    The vertex attributes of a primitive.
    """

    INDICES = "indices"
    """
    The triangle indices of a primitive.
    """

    MATERIAL = "material"
    """
    The material of a primitive.
    """

    MESH = "mesh"
    """
    The mesh of a node.
    """


class MaterialKey(StrEnum):
    """
    The property names of a glTF material and its extensions the writer emits.
    """

    METALLIC_ROUGHNESS = "pbrMetallicRoughness"
    """
    The metallic-roughness parameters.
    """

    BASE_COLOR_FACTOR = "baseColorFactor"
    """
    The linear base colour and opacity.
    """

    BASE_COLOR_TEXTURE = "baseColorTexture"
    """
    The base colour texture.
    """

    METALLIC_FACTOR = "metallicFactor"
    """
    The metalness.
    """

    ROUGHNESS_FACTOR = "roughnessFactor"
    """
    The roughness.
    """

    METALLIC_ROUGHNESS_TEXTURE = "metallicRoughnessTexture"
    """
    The texture holding roughness in green and metalness in blue.
    """

    DOUBLE_SIDED = "doubleSided"
    """
    Whether both sides of each triangle are drawn.
    """

    NORMAL_TEXTURE = "normalTexture"
    """
    The normal map.
    """

    EMISSIVE_FACTOR = "emissiveFactor"
    """
    The linear emitted colour.
    """

    EMISSIVE_TEXTURE = "emissiveTexture"
    """
    The emitted colour texture.
    """

    ALPHA_MODE = "alphaMode"
    """
    How the opacity is used.
    """

    ALPHA_CUTOFF = "alphaCutoff"
    """
    The opacity below which a masked surface is cut out.
    """

    TRANSMISSION_FACTOR = "transmissionFactor"
    """
    The share of light passing through the surface, named as
    ``KHR_materials_transmission`` names it.
    """

    INDEX_OF_REFRACTION = "ior"
    """
    The index of refraction.
    """

    CLEARCOAT_FACTOR = "clearcoatFactor"
    """
    The strength of the clear coat.
    """

    CLEARCOAT_ROUGHNESS_FACTOR = "clearcoatRoughnessFactor"
    """
    The roughness of the clear coat.
    """

    EMISSIVE_STRENGTH = "emissiveStrength"
    """
    The factor scaling the emitted colour.
    """

    ANISOTROPY_STRENGTH = "anisotropyStrength"
    """
    The strength of the anisotropic reflection.
    """


class NodeExtra(StrEnum):
    """
    The application data the writer stores on each node besides the semantic
    attributes.
    """

    USD_PATH = "usdPath"
    """
    The prim the node stands for: the common ancestor of the merged prims.
    """

    USD_PATHS = "usdPaths"
    """
    Every prim merged into the node.
    """


class AssetInformation(StrEnum):
    """
    What the writer states about the document in its ``asset`` property.
    """

    VERSION = "2.0"
    """
    The glTF version.
    """

    GENERATOR = "cramera-import"
    """
    The tool that wrote the document.
    """


class AccessorType(StrEnum):
    """
    The shapes of an accessor's elements the writer uses.
    """

    SCALAR = "SCALAR"
    """
    One component.
    """

    VECTOR_2 = "VEC2"
    """
    Two components, such as texture coordinates.
    """

    VECTOR_3 = "VEC3"
    """
    Three components, such as positions and normals.
    """

    @classmethod
    def of_width(cls, width: int) -> AccessorType:
        """
        The element shape with a number of components.

        :param width: The number of components.
        """
        return next(member for member in cls if member.width == width)

    @property
    def width(self) -> int:
        """
        The number of components.
        """
        return {
            AccessorType.SCALAR: 1,
            AccessorType.VECTOR_2: 2,
            AccessorType.VECTOR_3: 3,
        }[self]


class GlbSize(IntEnum):
    """
    The byte sizes of the parts of the GLB container.
    """

    ALIGNMENT = 4
    """
    Every chunk starts and ends on a multiple of this many bytes.
    """

    CHUNK_HEADER = 8
    """
    The length and type preceding each chunk.
    """

    HEADER = 12
    """
    The magic number, version and total length at the start of the file.
    """


# %% merged geometry


@dataclass(eq=False)
class MergedMesh:
    """
    The meshes of one link that share a material, concatenated into one primitive.
    """

    points: numpy.ndarray
    """
    ``(N, 3)`` float32 positions.
    """

    triangles: numpy.ndarray
    """
    ``(M, 3)`` uint32 indices.
    """

    normals: numpy.ndarray | None
    """
    ``(N, 3)`` float32 normals, or None when no source mesh has any.
    """

    texture_coordinates: numpy.ndarray | None
    """
    ``(N, 2)`` float32 glTF texture coordinates (V flipped), or None.
    """

    @classmethod
    def of_meshes(cls, meshes: list[ImportedMesh], textured: bool) -> Self:
        """
        Concatenate meshes, flipping USD's texture V axis into glTF's.

        A mesh without normals gets area-weighted vertex normals when others have
        normals; a mesh without texture coordinates gets zeros when the material is
        textured.

        :param meshes: The meshes sharing one material.
        :param textured: Whether the material reads texture coordinates.
        """
        offsets = numpy.cumsum([0] + [len(mesh.points) for mesh in meshes])
        with_normals = any(mesh.normals is not None for mesh in meshes)
        return cls(
            points=numpy.concatenate([mesh.points for mesh in meshes]).astype(
                numpy.float32
            ),
            triangles=numpy.concatenate(
                [mesh.triangles + offset for mesh, offset in zip(meshes, offsets)]
            ).astype(numpy.uint32),
            normals=(
                numpy.concatenate([cls.normals_of(mesh) for mesh in meshes]).astype(
                    numpy.float32
                )
                if with_normals
                else None
            ),
            texture_coordinates=(
                numpy.concatenate(
                    [cls.flipped_coordinates(mesh) for mesh in meshes]
                ).astype(numpy.float32)
                if textured
                else None
            ),
        )

    @staticmethod
    def normals_of(mesh: ImportedMesh) -> numpy.ndarray:
        """
        The mesh's normals, or area-weighted vertex normals computed from its faces.

        :param mesh: The mesh.
        """
        if mesh.normals is not None:
            return mesh.normals
        corners = mesh.points[mesh.triangles]
        face_normals = numpy.cross(
            corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        )
        normals = numpy.zeros_like(mesh.points, dtype=numpy.float64)
        for corner in range(3):
            numpy.add.at(normals, mesh.triangles[:, corner], face_normals)
        lengths = numpy.linalg.norm(normals, axis=1, keepdims=True)
        return normals / numpy.where(lengths > 0.0, lengths, 1.0)

    @staticmethod
    def flipped_coordinates(mesh: ImportedMesh) -> numpy.ndarray:
        """
        The mesh's texture coordinates with V flipped (USD's origin is bottom left,
        glTF's top left), or zeros when the mesh has none.

        :param mesh: The mesh.
        """
        if mesh.texture_coordinates is None:
            return numpy.zeros((len(mesh.points), 2), dtype=numpy.float32)
        flipped = mesh.texture_coordinates.astype(numpy.float32).copy()
        flipped[:, 1] = 1.0 - flipped[:, 1]
        return flipped


# %% document


@dataclass
class GltfDocument:
    """
    A glTF document under construction, with one binary buffer.
    """

    json: dict[str, Any] = field(
        default_factory=lambda: {
            GltfKey.ASSET: {
                GltfKey.VERSION: AssetInformation.VERSION,
                GltfKey.GENERATOR: AssetInformation.GENERATOR,
            },
            GltfKey.SCENE: 0,
            GltfKey.SCENES: [{GltfKey.NODES: []}],
            GltfKey.NODES: [],
            GltfKey.MESHES: [],
            GltfKey.MATERIALS: [],
            GltfKey.ACCESSORS: [],
            GltfKey.BUFFER_VIEWS: [],
            GltfKey.BUFFERS: [],
            GltfKey.SAMPLERS: [
                {
                    GltfKey.MAGNIFICATION_FILTER: SamplerSetting.LINEAR,
                    GltfKey.MINIFICATION_FILTER: SamplerSetting.LINEAR_MIPMAP_LINEAR,
                    GltfKey.WRAP_S: SamplerSetting.REPEAT,
                    GltfKey.WRAP_T: SamplerSetting.REPEAT,
                }
            ],
            GltfKey.IMAGES: [],
            GltfKey.TEXTURES: [],
        }
    )
    """
    The JSON part of the document.
    """

    binary: bytearray = field(default_factory=bytearray)
    """
    The binary buffer.
    """

    material_indices: dict[PhysicallyBasedMaterial, int] = field(default_factory=dict)
    """
    The index of each material written so far.
    """

    texture_indices: dict[TextureFile | RoughnessMetallicTexture, int] = field(
        default_factory=dict
    )
    """
    The index of each texture written so far.
    """

    extensions: set[GltfExtension] = field(default_factory=set)
    """
    The extensions the document uses.
    """

    blending: TransmissionBlending = field(default_factory=TransmissionBlending)
    """
    How transmissive surfaces are drawn, for viewers that cannot refract.
    """

    def add_buffer_view(self, data: bytes, target: BufferTarget | None = None) -> int:
        """
        Append bytes to the buffer, aligned to four bytes, and describe them.

        :param data: The bytes.
        :param target: What the bytes hold, or None for images.
        """
        self.binary.extend(b"\0" * (-len(self.binary) % GlbSize.ALIGNMENT))
        view = {
            GltfKey.BUFFER: 0,
            GltfKey.BYTE_OFFSET: len(self.binary),
            GltfKey.BYTE_LENGTH: len(data),
        }
        if target is not None:
            view[GltfKey.TARGET] = target
        self.binary.extend(data)
        self.json[GltfKey.BUFFER_VIEWS].append(view)
        return len(self.json[GltfKey.BUFFER_VIEWS]) - 1

    def add_accessor(
        self,
        values: numpy.ndarray,
        component_type: ComponentType,
        target: BufferTarget,
        with_bounds: bool = False,
    ) -> int:
        """
        Store an array and describe it as an accessor.

        :param values: A one-dimensional (scalars) or two-dimensional (vectors) array
            already of the component type.
        :param component_type: The type of each component.
        :param target: Whether the values are vertex attributes or indices.
        :param with_bounds: Whether to record the per-component minimum and maximum,
            which glTF requires for positions.
        """
        values = numpy.ascontiguousarray(values)
        width = 1 if values.ndim == 1 else values.shape[1]
        accessor = {
            GltfKey.BUFFER_VIEW: self.add_buffer_view(values.tobytes(), target),
            GltfKey.COMPONENT_TYPE: component_type,
            GltfKey.COUNT: int(values.shape[0]),
            GltfKey.TYPE: AccessorType.of_width(width),
        }
        if with_bounds:
            accessor[GltfKey.MINIMUM] = [float(value) for value in values.min(axis=0)]
            accessor[GltfKey.MAXIMUM] = [float(value) for value in values.max(axis=0)]
        self.json[GltfKey.ACCESSORS].append(accessor)
        return len(self.json[GltfKey.ACCESSORS]) - 1

    def add_texture(self, source: TextureFile | RoughnessMetallicTexture) -> int:
        """
        Embed a texture once and return its index.

        :param source: An image file, or a texture packed from greyscale sources.
        """
        if source in self.texture_indices:
            return self.texture_indices[source]
        data, image_type = (
            (source.encode(), ImageType.PNG)
            if isinstance(source, RoughnessMetallicTexture)
            else EmbeddedImage(source.path).encoded()
        )
        self.json[GltfKey.IMAGES].append(
            {
                GltfKey.BUFFER_VIEW: self.add_buffer_view(data),
                GltfKey.MIME_TYPE: image_type,
            }
        )
        self.json[GltfKey.TEXTURES].append(
            {GltfKey.SOURCE: len(self.json[GltfKey.IMAGES]) - 1, GltfKey.SAMPLER: 0}
        )
        self.texture_indices[source] = len(self.json[GltfKey.TEXTURES]) - 1
        return self.texture_indices[source]

    def texture_reference(
        self,
        source: TextureFile | RoughnessMetallicTexture,
        transform: TextureTransform | None,
    ) -> dict[str, Any]:
        """
        A glTF texture info pointing at an embedded texture.

        :param source: The texture.
        :param transform: The material's change of texture coordinates, if any.
        """
        reference: dict[str, Any] = {GltfKey.INDEX: self.add_texture(source)}
        if transform is not None:
            self.extensions.add(GltfExtension.TEXTURE_TRANSFORM)
            reference[GltfKey.EXTENSIONS] = {
                GltfExtension.TEXTURE_TRANSFORM: {
                    GltfKey.OFFSET: list(transform.offset),
                    GltfKey.SCALE: list(transform.scale),
                }
            }
        return reference

    def add_material(self, material: PhysicallyBasedMaterial) -> int:
        """
        Write a material once and return its index.

        :param material: The material.
        """
        if material in self.material_indices:
            return self.material_indices[material]
        self.json[GltfKey.MATERIALS].append(GltfMaterial(material, self).to_json())
        self.material_indices[material] = len(self.json[GltfKey.MATERIALS]) - 1
        return self.material_indices[material]

    def add_mesh_node(
        self,
        name: str,
        extras: dict[str, Any],
        mesh: MergedMesh,
        material: PhysicallyBasedMaterial,
    ) -> None:
        """
        Add one node holding one primitive.

        :param name: The node name.
        :param extras: Application data stored on the node.
        :param mesh: The geometry.
        :param material: The material of the geometry.
        """
        attributes = {
            AttributeName.POSITION: self.add_accessor(
                mesh.points, ComponentType.FLOAT, BufferTarget.ARRAY_BUFFER, True
            )
        }
        if mesh.normals is not None:
            attributes[AttributeName.NORMAL] = self.add_accessor(
                mesh.normals, ComponentType.FLOAT, BufferTarget.ARRAY_BUFFER
            )
        if mesh.texture_coordinates is not None:
            attributes[AttributeName.TEXTURE_COORDINATES] = self.add_accessor(
                mesh.texture_coordinates,
                ComponentType.FLOAT,
                BufferTarget.ARRAY_BUFFER,
            )
        index_type = ComponentType.indices_for(len(mesh.points))
        indices = self.add_accessor(
            mesh.triangles.ravel().astype(index_type.numpy_type),
            index_type,
            BufferTarget.ELEMENT_ARRAY_BUFFER,
        )
        self.json[GltfKey.MESHES].append(
            {
                GltfKey.NAME: name,
                GltfKey.PRIMITIVES: [
                    {
                        GltfKey.ATTRIBUTES: attributes,
                        GltfKey.INDICES: indices,
                        GltfKey.MATERIAL: self.add_material(material),
                    }
                ],
            }
        )
        self.json[GltfKey.NODES].append(
            {
                GltfKey.NAME: name,
                GltfKey.MESH: len(self.json[GltfKey.MESHES]) - 1,
                GltfKey.EXTRAS: extras,
            }
        )
        self.json[GltfKey.SCENES][0][GltfKey.NODES].append(
            len(self.json[GltfKey.NODES]) - 1
        )

    def encode(self) -> bytes:
        """
        The document as a GLB file.
        """
        self.binary.extend(b"\0" * (-len(self.binary) % GlbSize.ALIGNMENT))
        document = dict(self.json)
        document[GltfKey.BUFFERS] = [{GltfKey.BYTE_LENGTH: len(self.binary)}]
        if self.extensions:
            document[GltfKey.EXTENSIONS_USED] = sorted(self.extensions)
        for key in (GltfKey.IMAGES, GltfKey.TEXTURES, GltfKey.MATERIALS):
            if not document[key]:
                del document[key]
        if GltfKey.IMAGES not in document:
            del document[GltfKey.SAMPLERS]
        text = json.dumps(document, separators=(",", ":")).encode("utf-8")
        text += b" " * (-len(text) % GlbSize.ALIGNMENT)
        return b"".join(
            [
                struct.pack(
                    "<III",
                    GlbChunk.MAGIC,
                    GlbChunk.VERSION,
                    GlbSize.HEADER
                    + GlbSize.CHUNK_HEADER
                    + len(text)
                    + GlbSize.CHUNK_HEADER
                    + len(self.binary),
                ),
                struct.pack("<II", len(text), GlbChunk.JSON),
                text,
                struct.pack("<II", len(self.binary), GlbChunk.BINARY),
                bytes(self.binary),
            ]
        )


@dataclass
class EmbeddedImage:
    """
    An image file as glTF can embed it: PNG and JPEG as they are, anything else
    re-encoded as PNG.
    """

    path: Path
    """
    The image file.
    """

    def encoded(self) -> tuple[bytes, ImageType]:
        """
        The image bytes and their type.
        """
        data = self.path.read_bytes()
        if data.startswith(b"\x89PNG"):
            return data, ImageType.PNG
        if data.startswith(b"\xff\xd8"):
            return data, ImageType.JPEG
        stream = io.BytesIO()
        Image.open(self.path).save(stream, format="PNG")
        return stream.getvalue(), ImageType.PNG


@dataclass
class GltfMaterial:
    """
    The glTF JSON of one material.
    """

    material: PhysicallyBasedMaterial
    """
    The material to write, as authored.
    """

    document: GltfDocument
    """
    The document receiving its textures and extensions.
    """

    def to_json(self) -> dict[str, Any]:
        """
        The material as glTF JSON; a transmissive surface is written as its blended
        approximation, with its :meth:`physical_properties` in the ``extras``.
        """
        material = self.document.blending.approximate(self.material)
        transform = material.texture_transform
        metallic_roughness: dict[str, Any] = {
            MaterialKey.BASE_COLOR_FACTOR: [
                float(value) for value in material.base_color_factor
            ],
            MaterialKey.METALLIC_FACTOR: float(material.metallic_factor),
            MaterialKey.ROUGHNESS_FACTOR: float(material.roughness_factor),
        }
        if material.base_color_texture is not None:
            metallic_roughness[MaterialKey.BASE_COLOR_TEXTURE] = (
                self.document.texture_reference(material.base_color_texture, transform)
            )
        if material.metallic_roughness_texture is not None:
            metallic_roughness[MaterialKey.METALLIC_ROUGHNESS_TEXTURE] = (
                self.document.texture_reference(
                    material.metallic_roughness_texture, transform
                )
            )
        result: dict[str, Any] = {
            GltfKey.NAME: material.name,
            MaterialKey.METALLIC_ROUGHNESS: metallic_roughness,
            MaterialKey.DOUBLE_SIDED: material.double_sided,
        }
        if material.normal_texture is not None:
            result[MaterialKey.NORMAL_TEXTURE] = self.document.texture_reference(
                material.normal_texture, transform
            ) | {GltfKey.SCALE: float(material.normal_scale)}
        if any(material.emissive_factor):
            result[MaterialKey.EMISSIVE_FACTOR] = [
                float(value) for value in material.emissive_factor
            ]
        if material.emissive_texture is not None:
            result[MaterialKey.EMISSIVE_TEXTURE] = self.document.texture_reference(
                material.emissive_texture, transform
            )
        if material.alpha_mode is not AlphaMode.OPAQUE:
            result[MaterialKey.ALPHA_MODE] = material.alpha_mode.value
        if material.alpha_mode is AlphaMode.MASK:
            result[MaterialKey.ALPHA_CUTOFF] = float(material.alpha_cutoff)
        if material.transmissive:
            result[GltfKey.EXTRAS] = self.physical_properties()
        extensions = self.extensions()
        if extensions:
            result[GltfKey.EXTENSIONS] = extensions
            self.document.extensions.update(extensions)
        return result

    def physical_properties(self) -> dict[MaterialKey, float | list[float]]:
        """
        The authored values of a transmissive surface that a renderer which refracts
        needs and the blended approximation replaces: transmission, index of
        refraction, base colour and metalness, named as glTF and its
        ``KHR_materials_transmission`` and ``KHR_materials_ior`` extensions name them.
        """
        material = self.material
        return {
            MaterialKey.TRANSMISSION_FACTOR: float(material.transmission),
            MaterialKey.INDEX_OF_REFRACTION: float(material.index_of_refraction),
            MaterialKey.BASE_COLOR_FACTOR: [
                float(value) for value in material.base_color_factor
            ],
            MaterialKey.METALLIC_FACTOR: float(material.metallic_factor),
        }

    def extensions(self) -> dict[GltfExtension, dict[str, float]]:
        """
        The material extensions whose values differ from glTF's defaults; the index of
        refraction of a transmissive surface is kept with its
        :meth:`physical_properties`.
        """
        material = self.material
        extensions: dict[GltfExtension, dict[str, float]] = {}
        if not material.transmissive and material.index_of_refraction != 1.5:
            extensions[GltfExtension.INDEX_OF_REFRACTION] = {
                MaterialKey.INDEX_OF_REFRACTION: float(material.index_of_refraction)
            }
        if material.clearcoat > 0.0:
            extensions[GltfExtension.CLEARCOAT] = {
                MaterialKey.CLEARCOAT_FACTOR: float(material.clearcoat),
                MaterialKey.CLEARCOAT_ROUGHNESS_FACTOR: float(
                    material.clearcoat_roughness
                ),
            }
        if material.emissive_strength != 1.0:
            extensions[GltfExtension.EMISSIVE_STRENGTH] = {
                MaterialKey.EMISSIVE_STRENGTH: float(material.emissive_strength)
            }
        if material.anisotropy > 0.0:
            extensions[GltfExtension.ANISOTROPY] = {
                MaterialKey.ANISOTROPY_STRENGTH: float(material.anisotropy)
            }
        return extensions


# %% one link


@dataclass
class LinkGltf:
    """
    The GLB file of one link: its meshes merged per material, one node per material.
    """

    link: ImportedLink
    """
    The link whose meshes are written, in the link frame.
    """

    materials: dict[str, PhysicallyBasedMaterial]
    """
    The mapped material for every material name the meshes use.
    """

    semantics: dict[str, dict[str, Any]] = field(default_factory=dict)
    """
    Semantic attributes by prim path, stored on the nodes as ``extras``.
    """

    def groups(self) -> dict[PhysicallyBasedMaterial, list[ImportedMesh]]:
        """
        The link's meshes grouped by the material they are drawn with, in the order
        the materials first appear.
        """
        groups: dict[PhysicallyBasedMaterial, list[ImportedMesh]] = {}
        for mesh in self.link.meshes:
            material = self.materials[mesh.material]
            if mesh.double_sided:
                material = material.with_double_sides()
            groups.setdefault(material, []).append(mesh)
        return groups

    def encode(self) -> bytes:
        """
        The link's GLB file.
        """
        document = GltfDocument()
        for material, meshes in self.groups().items():
            paths = [mesh.path for mesh in meshes]
            name = self.common_prim_path(paths)
            extras: dict[str, Any] = {
                NodeExtra.USD_PATH: name,
                NodeExtra.USD_PATHS: sorted(set(paths)),
            }
            extras.update(self.semantics.get(name, {}))
            document.add_mesh_node(
                name, extras, MergedMesh.of_meshes(meshes, material.textured), material
            )
        return document.encode()

    def write(self, path: Path) -> None:
        """
        Write the link's GLB file.

        :param path: The destination file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.encode())

    @staticmethod
    def common_prim_path(paths: list[str]) -> str:
        """
        The deepest prim path that is an ancestor of (or equal to) every given path,
        which names a node merged from several prims.

        :param paths: Absolute prim paths.
        """
        common = os.path.commonpath(paths)
        return common if common != os.sep else paths[0]
