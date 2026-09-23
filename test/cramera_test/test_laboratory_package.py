"""
Verify that an installed package contains the complete laboratory world.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from enum import StrEnum
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cramera

from cramera.laboratory_world import LaboratoryAsset, LaboratoryField


# %% package layout
class PackageFile(StrEnum):
    """
    Files needed to build and inspect the distributable package.
    """

    CONFIGURATION = "pyproject.toml"
    README = "README.md"
    REQUIREMENTS = "requirements.txt"
    SOURCE = "src"
    RESOURCES = "resources"
    TRAJECTORY = "trajectory"
    MESH = "mesh"
    FILENAME = "filename"
    MESH_ELEMENT = ".//mesh"


# %% distributable asset completeness
def test_wheel_contains_self_contained_laboratory(tmp_path: Path) -> None:
    """
    Every scene and URDF asset resolves inside the built package wheel.
    """
    package_directory = Path(cramera.__file__).parents[2]
    build_directory = tmp_path / "package"
    build_directory.mkdir()
    for filename in (
        PackageFile.CONFIGURATION,
        PackageFile.README,
        PackageFile.REQUIREMENTS,
    ):
        shutil.copy2(package_directory / filename, build_directory / filename)
    shutil.copytree(
        package_directory / PackageFile.SOURCE,
        build_directory / PackageFile.SOURCE,
        ignore=shutil.ignore_patterns("*.egg-info", "__pycache__"),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(build_directory),
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--wheel-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    [wheel_path] = tmp_path.glob("*.whl")
    bundle = (
        PurePosixPath(cramera.__name__)
        / PackageFile.RESOURCES
        / LaboratoryAsset.SCENE_NAME
    )
    with ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        assert str(bundle / LaboratoryAsset.SCENE) in names
        assert str(bundle / LaboratoryAsset.SEMANTICS) in names
        scene = json.loads(wheel.read(str(bundle / LaboratoryAsset.SCENE)))
        descriptions = {
            item[LaboratoryField.URDF]
            for item in scene[LaboratoryField.MODELS] + scene[LaboratoryField.OBJECTS]
        }
        references = descriptions | {
            item[PackageFile.MESH] for item in scene[LaboratoryField.OBJECTS]
        }
        references.add(scene[PackageFile.TRAJECTORY])
        for description in descriptions:
            assert str(bundle / description) in names
            document = ElementTree.fromstring(wheel.read(str(bundle / description)))
            references.update(
                mesh.attrib[PackageFile.FILENAME]
                for mesh in document.findall(PackageFile.MESH_ELEMENT)
            )
        for reference in references:
            path = PurePosixPath(reference)
            assert not path.is_absolute()
            assert ".." not in path.parts
            assert str(bundle / path) in names
