"""
Copying mesh files and the files they reference into a scene bundle, so the bundle loads
in the browser without the machine it was written on.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from typing_extensions import Dict, List, Optional, Pattern, Set

from cramera_bridge.exceptions import PathOutsideBundle
from cramera_protocol.bundle.files import BundleFile
from semantic_digital_twin.adapters.package_resolver import PackageUriResolver
from semantic_digital_twin.exceptions import ParsingError

# %% formats


class MeshFormat(StrEnum):
    """
    The mesh formats the viewer loads, by the suffix their files carry.
    """

    STL = ".stl"
    """
    Stereolithography; geometry only.
    """

    OBJ = ".obj"
    """
    Wavefront OBJ; names its materials in a material library beside it.
    """

    DAE = ".dae"
    """
    COLLADA; names its textures inside the document.
    """

    GLB = ".glb"
    """
    Binary glTF; carries geometry, materials and textures in one file.
    """

    GLTF = ".gltf"
    """
    GlTF; names its buffers and textures inside the document.
    """

    @classmethod
    def of_path(cls, path: str) -> Optional[MeshFormat]:
        """
        :param path: A file name, path or URI.
        :return: The format its suffix names, or None for a file that is no mesh.
        """
        lowered = path.lower()
        for member in cls:
            if lowered.endswith(member.value):
                return member
        return None


class ReferenceScheme(StrEnum):
    """
    The URI schemes a URDF mesh reference may use.
    """

    PACKAGE = "package://"
    """
    A file inside a ROS package.
    """

    FILE = "file://"
    """
    A local file.
    """


DIGEST_LENGTH = 8
"""
How many hexadecimal digits of a source directory's digest name its bundle folder.
"""

# %% material libraries


@dataclass(frozen=True)
class MaterialLibrary:
    """
    The material library an OBJ mesh declares.
    """

    DECLARATION: Pattern[str] = re.compile(r"^\s*mtllib\s+(.+?)\s*$", re.MULTILINE)
    """
    The OBJ statement naming a material library.
    """

    TEXTURE_MAP: Pattern[str] = re.compile(r"^\s*map_\w+\s+(.+?)\s*$", re.MULTILINE)
    """
    The material-library statement naming a texture.
    """

    @classmethod
    def declared_by(cls, mesh_file: Path) -> List[str]:
        """
        :param mesh_file: An OBJ mesh file.
        :return: The material libraries it names, relative to its directory.
        """
        if MeshFormat.of_path(str(mesh_file)) is not MeshFormat.OBJ:
            return []
        text = mesh_file.read_bytes().decode("utf-8", "replace")
        return [match.strip() for match in cls.DECLARATION.findall(text)]

    @classmethod
    def textures_of(cls, library_file: Path) -> List[str]:
        """
        :param library_file: A material library.
        :return: The textures it names, relative to its own directory.
        """
        text = library_file.read_bytes().decode("utf-8", "replace")
        return [match.strip() for match in cls.TEXTURE_MAP.findall(text)]

    @classmethod
    def companion_of(cls, mesh_file: Path) -> Optional[Path]:
        """
        :param mesh_file: An OBJ mesh file.
        :return: The first material library it names that exists, or None.
        """
        for library in cls.declared_by(mesh_file):
            candidate = mesh_file.parent / library
            if candidate.is_file():
                return candidate
        return None


# %% copying


@dataclass
class BundledAssets:
    """
    The files copied into one bundle and the references that resolved to no file.
    """

    bundle_root: Path
    """
    The bundle directory; nothing is written outside it.
    """

    DOCUMENT_TEXTURE: Pattern[str] = field(
        default=re.compile(r"[\w./\-]+\.(?:png|jpg|jpeg|tga|tif|bin)", re.IGNORECASE),
        repr=False,
    )
    """
    The files a COLLADA or glTF document names.
    """

    copied: Dict[Path, Path] = field(default_factory=dict)
    """
    Each copied source file and where it was copied to first.
    """

    missing: List[str] = field(default_factory=list)
    """
    The references that resolved to no file, in the order they were met.
    """

    def require_inside(self, destination: Path) -> None:
        """
        :param destination: A path that is about to be written.
        :raises PathOutsideBundle: When it resolves outside :attr:`bundle_root`.
        """
        if not destination.resolve().is_relative_to(self.bundle_root.resolve()):
            raise PathOutsideBundle(str(destination))

    def folder_for(self, source: Path, mesh_directory: str) -> Path:
        """
        A bundle folder of its own for every source directory, so equally named files of
        different directories never overwrite each other, while files referencing each
        other within one directory stay side by side.

        :param source: A source file.
        :param mesh_directory: The folder below ``meshes`` the bundle's meshes go to.
        :return: The folder the file is copied into.
        """
        directory = source.resolve().parent
        digest = hashlib.sha1(str(directory).encode("utf-8")).hexdigest()
        return (
            self.bundle_root
            / BundleFile.MESHES
            / mesh_directory
            / f"{directory.name}-{digest[:DIGEST_LENGTH]}"
        )

    def copy_mesh(self, source: Optional[Path], destination: Path) -> bool:
        """
        Copy a mesh and every file it references.

        :param source: The mesh file, or None for a reference that resolved to nothing.
        :param destination: Where the mesh goes inside the bundle.
        :return: Whether the mesh is in the bundle afterwards.
        """
        if not self.copy(source, destination):
            return False
        self.copy_side_assets(source, destination)
        return True

    def copy(self, source: Optional[Path], destination: Path) -> bool:
        """
        Copy one file; a source copied before to the same place is not copied again.

        :param source: The file, or None for a reference that resolved to nothing.
        :param destination: Where it goes inside the bundle.
        :return: Whether the file is in the bundle afterwards.
        :raises PathOutsideBundle: When the destination leaves the bundle.
        """
        self.require_inside(destination)
        if source is None or not source.is_file():
            self.missing.append(str(source))
            return False
        if self.copied.get(source) == destination:
            return True
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        self.copied.setdefault(source, destination)
        return True

    def copy_side_assets(self, source_mesh: Path, bundled_mesh: Path) -> None:
        """
        Copy the files a mesh references to the same place relative to its copy, so a
        reference resolves in the bundle as it did on disk.

        :param source_mesh: The source mesh.
        :param bundled_mesh: Its copy inside the bundle.
        """
        for reference in self.side_references(source_mesh):
            source = Path(os.path.normpath(source_mesh.parent / reference))
            destination = Path(os.path.normpath(bundled_mesh.parent / reference))
            if source.is_file():
                self.copy(source, destination)
            else:
                self.missing.append(str(source))

    def side_references(self, source_mesh: Path) -> Set[str]:
        """
        :param source_mesh: A mesh file.
        :return: The files it references, relative to its directory: the material
            libraries of an OBJ with their textures, the textures and buffers of a
            COLLADA or glTF document.
        """
        mesh_format = MeshFormat.of_path(str(source_mesh))
        if mesh_format is MeshFormat.OBJ:
            references: Set[str] = set()
            for library in MaterialLibrary.declared_by(source_mesh):
                references.add(library)
                library_file = source_mesh.parent / library
                if library_file.is_file():
                    references.update(
                        str(PurePosixPath(library).parent / texture)
                        for texture in MaterialLibrary.textures_of(library_file)
                    )
            return references
        if mesh_format in (MeshFormat.DAE, MeshFormat.GLTF):
            text = source_mesh.read_bytes().decode("utf-8", "replace")
            return set(self.DOCUMENT_TEXTURE.findall(text))
        return set()

    def relative(self, path: Path) -> str:
        """
        :param path: A path inside the bundle.
        :return: The path relative to the bundle, with ``/`` separators.
        """
        return path.relative_to(self.bundle_root).as_posix()


# %% references of existing URDF files


@dataclass(frozen=True)
class MeshReference:
    """
    One mesh reference as a URDF writes it.
    """

    uri: str
    """
    The reference exactly as written.
    """

    def resolve(self, base_directory: Path) -> Optional[Path]:
        """
        :param base_directory: The directory a relative reference is read against.
        :return: The existing file the reference names, or None.
        """
        if self.uri.startswith(ReferenceScheme.PACKAGE):
            return self.resolved_package_file()
        if self.uri.startswith(ReferenceScheme.FILE):
            candidate = Path(unquote(self.uri[len(ReferenceScheme.FILE) :]))
        else:
            candidate = base_directory / unquote(self.uri)
        return candidate if candidate.is_file() else None

    def resolved_package_file(self) -> Optional[Path]:
        """
        The native resolver reports a package it cannot find only by raising, so this is
        the one place where that report becomes a missing file.

        :return: The file a ``package://`` reference names, or None when its package or
            file is not installed.
        """
        try:
            resolved = Path(PackageUriResolver().resolve(self.uri))
        except ParsingError:
            return None
        return resolved if resolved.is_file() else None
