"""
An existing URDF or xacro file written as an environment bundle: the file with every
mesh it references, a scene description naming it as the one model, no robot, and an
entry in the scenes directory's index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Set

from cramera_bridge.documents import DocumentFile
from cramera_bridge.exceptions import UrdfSourceNotFound
from cramera_bridge.export.scene_index import SceneIndexFile
from cramera_bridge.export.urdf import URDF_SUFFIX, XACRO_SUFFIX, UrdfBundle
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.bundle.index import SceneIndexEntry, SceneName
from cramera_protocol.bundle.scene import ModelDescription, SceneDescription
from cramera_protocol.plan_builder import LoadedEnvironment
from cramera_protocol.version import CURRENT_VERSION

FALLBACK_NAME = "environment"
"""
The bundle name of a file whose name keeps no letter or digit.
"""

MESH_DIRECTORY = "environment"
"""
The folder below ``meshes`` the copied meshes go to.
"""

NAME_LENGTH = 56
"""
How long the name taken from the file may be, leaving room for a number that makes it
unique within the 64 characters of a bundle name.
"""


@dataclass(frozen=True)
class UrdfEnvironmentBundle:
    """
    Writes an existing URDF or xacro file into a scenes directory as an environment.
    """

    scenes: Path
    """
    The scenes directory the bundle is written to.
    """

    def write(self, source: Path, taken: Set[str]) -> LoadedEnvironment:
        """
        :param source: The URDF or xacro file.
        :param taken: The bundle names already in use, in any scenes directory.
        :return: The bundle's index entry and, once each, the meshes that were not
            found.
        :raises UrdfSourceNotFound: When the source does not exist.
        """
        if not source.is_file():
            raise UrdfSourceNotFound(str(source))
        name = self.free_name(source, taken)
        bundle = self.scenes / name
        bundle.mkdir(parents=True)
        report = UrdfBundle(bundle, MESH_DIRECTORY).write(source, name)
        label = self.label_of(source)
        description = SceneDescription(
            name=name,
            format_version=CURRENT_VERSION,
            environment_name=label,
            models=[
                ModelDescription(
                    name=name,
                    urdf=report.urdf.name,
                    prefix="",
                    robot=False,
                    links=len(report.links),
                    movable_joints=report.movable_joints,
                )
            ],
        )
        DocumentFile(bundle / BundleFile.SCENE).write(description)
        entry = SceneIndexEntry(name=name, environment=label)
        SceneIndexFile(self.scenes).register(entry)
        return LoadedEnvironment(
            entry=entry, missing_meshes=sorted(set(report.missing))
        )

    @staticmethod
    def label_of(source: Path) -> str:
        """
        :param source: The URDF or xacro file.
        :return: The file's name without its suffixes, as the pickers show it.
        """
        name = source.name
        for suffix in (XACRO_SUFFIX, URDF_SUFFIX):
            name = name.removesuffix(suffix)
        return name

    def free_name(self, source: Path, taken: Set[str]) -> str:
        """
        :param source: The URDF or xacro file.
        :param taken: The bundle names already in use.
        :return: A valid bundle name after the file, numbered when it is in use or a
            directory of that name exists.
        """
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", self.label_of(source)).strip("_")
        stem = stem[:NAME_LENGTH] or FALLBACK_NAME
        name, number = stem, 1
        while (
            name in taken
            or (self.scenes / name).exists()
            or not SceneName.is_valid(name)
        ):
            number += 1
            name = f"{stem}_{number}"
        return name
