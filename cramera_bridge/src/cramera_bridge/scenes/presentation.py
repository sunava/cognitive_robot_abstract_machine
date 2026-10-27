"""
The authored appearance of a scene, carried from its bundle into bundles written of
worlds built from it: renderer settings, the initial view, lighting and which models and
objects keep their authored materials.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import FrozenSet, Optional

from cramera.confinement import ConfinedDirectory
from cramera_bridge.documents import DocumentFile
from cramera_bridge.exceptions import BundleNotFound
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.bundle.scene import Camera, Rendering, SceneDescription

# %% presentation


class PresentationSetting(StrEnum):
    """
    The settings a presentation's signature covers.
    """

    RENDERING = "rendering"
    """
    The renderer settings.
    """

    CAMERA = "camera"
    """
    The initial view.
    """

    ENVIRONMENT_MATERIALS = "environmentMaterials"
    """
    Whether environment models keep their materials.
    """

    OBJECT_MATERIALS = "objectMaterials"
    """
    Which objects keep their materials.
    """


@dataclass(frozen=True)
class ScenePresentation:
    """
    Appearance settings of an authored scene, without its manipulation controls.
    """

    source_directory: Path
    """
    The bundle the settings were read from; a lighting file is relative to it.
    """

    rendering: Optional[Rendering] = None
    """
    The renderer settings.
    """

    camera: Optional[Camera] = None
    """
    The initial view.
    """

    preserve_environment_materials: bool = False
    """
    Whether an environment model of the source keeps its authored materials.
    """

    preserved_object_keys: FrozenSet[str] = field(default_factory=frozenset)
    """
    The keys of the source's objects that keep their authored materials.
    """

    @classmethod
    def of_bundle(cls, directory: Path) -> ScenePresentation:
        """
        :param directory: A bundle directory.
        :return: The bundle's appearance settings.
        :raises BundleNotFound: When the directory holds no ``scene.json``.
        """
        scene_file = DocumentFile(directory / BundleFile.SCENE)
        if not scene_file.path.is_file():
            raise BundleNotFound(directory)
        scene = scene_file.read(SceneDescription)
        return cls(
            source_directory=directory,
            rendering=scene.rendering,
            camera=scene.camera,
            preserve_environment_materials=any(
                not model.robot and model.preserve_materials is True
                for model in scene.models
            ),
            preserved_object_keys=frozenset(
                description.key
                for description in scene.objects
                if description.preserve_materials is True
            ),
        )

    def apply(self, scene: SceneDescription, destination: Path) -> None:
        """
        Give a scene written into another bundle these settings, copying the lighting
        file the renderer settings name into that bundle.

        :param scene: The scene, changed in place.
        :param destination: The directory of the bundle the scene is written to.
        """
        if self.rendering is not None:
            scene.rendering = deepcopy(self.rendering)
            self.copy_lighting(destination)
        if self.camera is not None:
            scene.camera = deepcopy(self.camera)
        if self.preserve_environment_materials:
            for model in scene.models:
                if not model.robot:
                    model.preserve_materials = True
        for description in scene.objects:
            if description.key in self.preserved_object_keys:
                description.preserve_materials = True

    def copy_lighting(self, destination: Path) -> None:
        """
        :param destination: The directory of the bundle receiving the settings.
        :raises PathOutsideDirectory: When the lighting file named lies outside the
            source bundle or would be written outside the destination.
        """
        if self.rendering.environment is None:
            return
        source = ConfinedDirectory(self.source_directory).resolve(
            self.rendering.environment
        )
        target = ConfinedDirectory(destination).resolve(self.rendering.environment)
        if source.is_file() and source != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def signature(self) -> str:
        """
        :return: A digest of every setting that changes how a written scene looks.
        """
        settings = {
            PresentationSetting.RENDERING: (
                self.rendering.to_json() if self.rendering is not None else None
            ),
            PresentationSetting.CAMERA: (
                self.camera.to_json() if self.camera is not None else None
            ),
            PresentationSetting.ENVIRONMENT_MATERIALS: (
                self.preserve_environment_materials
            ),
            PresentationSetting.OBJECT_MATERIALS: sorted(self.preserved_object_keys),
        }
        return hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode("utf-8")
        ).hexdigest()
