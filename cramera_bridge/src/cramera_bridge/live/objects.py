"""
The free objects of the live world with their native shapes, and the mesh files the
viewer loads them from.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlencode

from typing_extensions import Dict, List, Optional

from cramera.confinement import ConfinedDirectory
from cramera_bridge.exceptions import MissingMeshFile, UnknownMesh, UnsupportedShape
from cramera_bridge.geometry import (
    SIZE_PRECISION,
    NumericPose,
    PublishedColor,
    PublishedScale,
)
from cramera_bridge.live.world_model import ObjectKey, WorldModelView
from cramera_protocol.live import LiveRoute
from cramera_protocol.live_messages import (
    LiveObject,
    LiveParameter,
    ObjectCatalog,
    ShapeDescription,
    ShapeKind,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Mesh,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.world_entity import Body

MESH_KEY_SEPARATOR = "#"
"""
Separates an object's key from the position of the shape in a served mesh's key.
"""


class ObjFileStatement(StrEnum):
    """
    The statements of an OBJ file the publisher reads.
    """

    MATERIAL_LIBRARY = "mtllib"
    """
    Names the material library of the mesh.
    """


# %% served meshes


@dataclass(frozen=True)
class ServedMesh:
    """
    One mesh file the publisher serves, with the files beside it it may serve too.
    """

    path: Path
    """
    The mesh file.
    """

    def material_library(self) -> Optional[str]:
        """
        :return: The material library an OBJ file names, relative to its directory, or
            None for another format or an OBJ file without one.
        """
        if self.path.suffix.lower() != ".obj":
            return None
        for line in self.path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            statement, _, argument = line.strip().partition(" ")
            if statement == ObjFileStatement.MATERIAL_LIBRARY and argument.strip():
                return argument.strip()
        return None

    def side_file(self, requested: str) -> Path:
        """
        :param requested: A file beside the mesh, relative to the mesh's directory.
        :return: The file.
        :raises PathOutsideDirectory: When the path leaves the mesh's directory.
        :raises FileNotInDirectory: When there is no such file.
        """
        return ConfinedDirectory(self.path.parent).file(requested)


@dataclass(frozen=True)
class MeshRoute:
    """
    The route serving one mesh file and its side files.
    """

    key: str
    """
    The served mesh's key.
    """

    def url(self, side: Optional[str] = None) -> str:
        """
        :param side: A side file of the mesh, or None for the mesh itself.
        :return: The route, relative to the publisher.
        """
        parameters = {LiveParameter.KEY.value: self.key}
        if side is not None:
            parameters[LiveParameter.SIDE.value] = side
        return f"{LiveRoute.MESH}?{urlencode(parameters)}"


# %% shapes


@dataclass(frozen=True)
class ShapePublication:
    """
    Describes one native shape as the viewer builds it.
    """

    shape: Shape
    """
    The shape.
    """

    def description(self, mesh_key: str) -> ShapeDescription:
        """
        :param mesh_key: The key a mesh shape's file is served under.
        :return: The shape's geometry, local pose and native colour.
        :raises MissingMeshFile: For a mesh whose file does not exist.
        :raises UnsupportedShape: For a shape without a published form.
        """
        pose = NumericPose.of_matrix(self.shape.origin.to_np())
        color = PublishedColor(self.shape.color)
        description = ShapeDescription(
            kind=ShapeKind.BOX,
            position=pose.published_position(),
            quaternion=pose.published_orientation(),
            color=color.hexadecimal(),
            opacity=color.opacity(),
        )
        if isinstance(self.shape, Box):
            description.size = PublishedScale(self.shape.scale).published()
            return description
        if isinstance(self.shape, Cylinder):
            description.kind = ShapeKind.CYLINDER
            description.radius = round(self.shape.radius, SIZE_PRECISION)
            description.height = round(self.shape.height, SIZE_PRECISION)
            return description
        if isinstance(self.shape, Sphere):
            description.kind = ShapeKind.SPHERE
            description.radius = round(self.shape.radius, SIZE_PRECISION)
            return description
        if isinstance(self.shape, Mesh):
            return self.mesh_description(description, mesh_key)
        raise UnsupportedShape(type(self.shape).__name__)

    def mesh_description(
        self, description: ShapeDescription, mesh_key: str
    ) -> ShapeDescription:
        """
        :param description: The shape's pose and colour.
        :param mesh_key: The key the mesh file is served under.
        :return: The description completed with the mesh's route and scale.
        :raises MissingMeshFile: When the mesh's file does not exist.
        """
        served = self.served_mesh()
        route = MeshRoute(mesh_key)
        library = served.material_library()
        description.kind = ShapeKind.MESH
        description.mesh = route.url()
        description.material_library = None if library is None else route.url(library)
        description.file_format = served.path.suffix.lstrip(".").lower()
        description.scale = PublishedScale(self.shape.scale).published()
        return description

    def served_mesh(self) -> ServedMesh:
        """
        :return: The mesh file of a mesh shape.
        :raises MissingMeshFile: When the file does not exist.
        """
        path = Path(self.shape.filename)
        if not self.shape.filename or not path.is_file():
            raise MissingMeshFile(path)
        return ServedMesh(path.resolve())


@dataclass(frozen=True)
class BodyShapes:
    """
    The shapes a body is drawn with.
    """

    body: Body
    """
    The body.
    """

    def shapes(self) -> List[Shape]:
        """
        :return: The visual shapes, or the collision shapes of a body without visual
            ones; empty for a body without any.
        """
        if self.body.visual.shapes:
            return list(self.body.visual.shapes)
        return list(self.body.collision.shapes)


# %% catalog


@dataclass
class ObjectCatalogPublisher:
    """
    Describes the free objects after every model change and serves their meshes.
    """

    catalog: ObjectCatalog = field(default_factory=ObjectCatalog)
    """
    The newest catalog.
    """

    meshes: Dict[str, ServedMesh] = field(default_factory=dict)
    """
    The served mesh files by key.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards :attr:`catalog` and :attr:`meshes`.
    """

    def refresh(self, model: WorldModelView) -> ObjectCatalog:
        """
        Describe every free object of the model.

        :param model: The world's current model.
        :return: The new catalog.
        """
        objects: List[LiveObject] = []
        meshes: Dict[str, ServedMesh] = {}
        for body in model.objects:
            key = ObjectKey.of(body)
            descriptions = []
            for position, shape in enumerate(BodyShapes(body).shapes()):
                mesh_key = f"{key}{MESH_KEY_SEPARATOR}{position}"
                publication = ShapePublication(shape)
                descriptions.append(publication.description(mesh_key))
                if isinstance(shape, Mesh):
                    meshes[mesh_key] = publication.served_mesh()
            objects.append(
                LiveObject(key=key, identifier=str(body.name), shapes=descriptions)
            )
        catalog = ObjectCatalog(objects=objects)
        with self.lock:
            self.catalog = catalog
            self.meshes = meshes
        return catalog

    def current(self) -> ObjectCatalog:
        """
        :return: The newest catalog.
        """
        with self.lock:
            return self.catalog

    def mesh_file(self, key: str, side: Optional[str] = None) -> Path:
        """
        :param key: The key of a served mesh.
        :param side: A file beside the mesh, or None for the mesh itself.
        :return: The file to serve.
        :raises UnknownMesh: When no mesh is served under the key.
        """
        with self.lock:
            served = self.meshes.get(key)
        if served is None:
            raise UnknownMesh(key)
        if side is None:
            return served.path
        return served.side_file(side)
