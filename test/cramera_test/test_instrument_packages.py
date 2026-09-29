"""
Portable instrument packages reject corrupt geometry and disconnected frames.
"""

from __future__ import annotations

import importlib
import json
import struct
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest
import trimesh
from trimesh.exchange.gltf import export_glb


# %% Blender-independent package fixture
@pytest.fixture
def package_modules(monkeypatch: pytest.MonkeyPatch):
    """
    Load authoring and validation without importing Blender.
    """
    directory = (
        Path(__file__).resolve().parents[2] / "cramera/tools/laboratory/instruments"
    )
    monkeypatch.syspath_prepend(str(directory))
    return tuple(
        importlib.import_module(name)
        for name in ("spec", "description", "validate_packages")
    )


@pytest.fixture
def portable_package(package_modules, tmp_path: Path):
    """
    Export an actual small URDF and glTF package using the public description API.
    """
    specification, descriptions, validation = package_modules
    model = specification.DeviceSpec("fixture", "Fixture", "Validation fixture")
    model.part("base").box("plinth", (0, 0, 0.02), (0.20, 0.18, 0.04))
    slider = model.part(
        "slider",
        parent="base",
        origin=(0, 0, 0.05),
        joint=specification.JointKind.PRISMATIC,
        axis=(0, 0, 1),
        lower=0,
        upper=0.10,
        open_value=0.08,
    )
    slider.box("slider", (0, 0, 0.01), (0.08, 0.06, 0.02))
    model.frame("grasp", "slider", (0, 0, 0.03), "grasp")
    directory = tmp_path / model.identifier
    descriptions.InstrumentDescription(model).write(directory)
    (directory / "assets").mkdir()
    for part in model.parts:
        shape = part.shapes[0]
        mesh = trimesh.creation.box(extents=shape.dimensions)
        mesh.apply_translation(shape.center)
        (directory / "assets" / f"{part.name}.glb").write_bytes(
            trimesh.Scene(mesh).export(file_type="glb")
        )
    return validation.PackageValidator(directory)


# %% Graph and metadata checks
class TestPortablePackageStructure:
    """
    The actual serialized graph must preserve the independently described frames.
    """

    def test_connected_serialization_validates(self, portable_package) -> None:
        """
        The public exporter emits a connected graph and matching interaction metadata.
        """
        portable_package.validate_structure()
        summaries = portable_package.validate_meshes()
        assert len(summaries) == 2
        assert sum(summary.triangles for summary in summaries) == 24

    def test_wrong_frame_parent_is_rejected(
        self, portable_package, package_modules
    ) -> None:
        """
        A grasp target cannot silently attach to the stationary base.
        """
        root = ElementTree.parse(portable_package.directory / "model.urdf")
        root.find("joint[@name='fixture_grasp_joint']/parent").set(
            "link", "fixture_base"
        )
        root.write(portable_package.directory / "model.urdf")
        with pytest.raises(package_modules[2].InvalidPackage):
            portable_package.validate_structure()

    def test_disconnected_cycle_is_rejected(
        self, portable_package, package_modules
    ) -> None:
        """
        A loop between a slider and its frame cannot masquerade as a valid tree.
        """
        root = ElementTree.parse(portable_package.directory / "model.urdf")
        root.find("joint[@name='fixture_slider_joint']/parent").set(
            "link", "fixture_grasp"
        )
        root.write(portable_package.directory / "model.urdf")
        with pytest.raises(package_modules[2].InvalidPackage):
            portable_package.validate_structure()

    def test_out_of_range_inspection_pose_is_rejected(
        self, portable_package, package_modules
    ) -> None:
        """
        A preview pose must respect the same exported limits as the joint.
        """
        path = portable_package.directory / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["joints"][0]["inspectionPosition"] = metadata["joints"][0]["upper"] + 1
        path.write_text(json.dumps(metadata))
        with pytest.raises(package_modules[2].InvalidPackage):
            portable_package.validate_structure()


# %% Real mesh decoding
class TestPortableMeshLoading:
    """
    Meshes must decode locally in the authored units and body coordinates.
    """

    def test_missing_mesh_is_rejected(self, portable_package, package_modules) -> None:
        """
        A missing mesh is an invalid package, even when the URDF is syntactically valid.
        """
        (portable_package.directory / "assets/base.glb").unlink()
        with pytest.raises(package_modules[2].InvalidPackage):
            portable_package.validate_meshes()

    def test_external_mesh_path_is_rejected(
        self, portable_package, package_modules
    ) -> None:
        """
        An instrument cannot require a file outside its own resource directory.
        """
        root = ElementTree.parse(portable_package.directory / "model.urdf")
        root.find("link/visual/geometry/mesh").set("filename", "../other.glb")
        root.write(portable_package.directory / "model.urdf")
        with pytest.raises(package_modules[2].InvalidPackage):
            portable_package.validate_meshes()

    def test_millimetre_scale_export_is_rejected(
        self, portable_package, package_modules
    ) -> None:
        """
        Geometry authored in metres must not be exported a thousand times larger.
        """
        path = portable_package.directory / "assets/base.glb"
        scene = trimesh.load_scene(path)
        scene.apply_scale(1000)
        path.write_bytes(scene.export(file_type="glb"))
        with pytest.raises(package_modules[2].InvalidPackage):
            portable_package.validate_meshes()

    def test_nonfinite_vertex_is_rejected(
        self, portable_package, package_modules
    ) -> None:
        """
        An otherwise loadable glTF cannot contain a nonfinite mesh coordinate.
        """
        path = portable_package.directory / "assets/base.glb"
        content = bytearray(path.read_bytes())
        json_length = struct.unpack_from("<I", content, 12)[0]
        document = json.loads(content[20 : 20 + json_length])
        position_index = document["meshes"][0]["primitives"][0]["attributes"][
            "POSITION"
        ]
        accessor = document["accessors"][position_index]
        view = document["bufferViews"][accessor["bufferView"]]
        position_offset = (
            28 + json_length + view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        )
        struct.pack_into("<f", content, position_offset, float("nan"))
        path.write_bytes(content)
        with pytest.raises(package_modules[2].InvalidPackage):
            portable_package.validate_meshes()

    def test_empty_mesh_is_rejected(self, portable_package, package_modules) -> None:
        """
        A parseable glTF scene without surfaces cannot represent a physical part.
        """
        path = portable_package.directory / "assets/base.glb"
        path.write_bytes(export_glb(trimesh.Scene()))
        with pytest.raises(package_modules[2].InvalidPackage):
            portable_package.validate_meshes()


# %% Generated resources when installed


def test_installed_instrument_resources_are_portable(package_modules) -> None:
    """
    Validate shipped packages when optional authored binary resources are installed.
    """
    repository = Path(__file__).resolve().parents[2]
    directory = repository / "cramera/src/cramera/resources/laboratory_devices"
    catalog = importlib.import_module("catalog").build_devices()
    if any(
        not (directory / model.identifier / "metadata.json").is_file()
        for model in catalog
    ):
        pytest.skip(
            "The optional generated instrument resource collection is not installed"
        )
    reports = []
    for model in catalog:
        validator = package_modules[2].PackageValidator(directory / model.identifier)
        validator.validate_definition(model)
        reports.append(validator.validate())
    assert {report.device for report in reports} == {
        model.identifier for model in catalog
    }


# %% Optional native Blender read-back


@pytest.mark.parametrize(
    "identifiers", (("vortex_mixer", "incubator"), ("barcode_dock",))
)
def test_saved_blender_hierarchy_matches_urdf(
    tmp_path: Path, identifiers: tuple[str, ...]
) -> None:
    """
    Locally installed Blender reopens saved models and checks embedded image resources.
    """
    executable = shutil.which("blender")
    repository = Path(__file__).resolve().parents[2]
    directory = repository / "cramera/src/cramera/resources/laboratory_devices"
    if executable is None or any(
        not (directory / name / "model.blend").is_file() for name in identifiers
    ):
        pytest.skip(
            "Blender and optional generated models are required for native project read-back"
        )
    report_path = tmp_path / "readback.json"
    command = [
        executable,
        "-b",
        "-t",
        "2",
        "--python-exit-code",
        "1",
        "--python",
        str(repository / "cramera/tools/laboratory/instruments/validate_blender.py"),
        "--",
        "--root",
        str(directory),
        "--report",
        str(report_path),
    ]
    for identifier in identifiers:
        command.extend(("--device", identifier))
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    reports = json.loads(report_path.read_text())
    assert {report["device"] for report in reports} == set(identifiers)
    for report in reports:
        metadata = json.loads(
            (directory / report["device"] / "metadata.json").read_text()
        )
        assert report["movable_bodies"] == len(metadata["joints"])
        assert report["targets"] == len(metadata["frames"])
        definition = json.loads(
            (directory / report["device"] / "definition.json").read_text()
        )
        if any(
            shape["kind"] == "logo"
            for part in definition["parts"]
            for shape in part["shapes"]
        ):
            assert report["packed_images"] > 0


# %% Authored definition parity


def test_stale_generated_definition_requires_rebuild(
    package_modules, tmp_path: Path
) -> None:
    """
    Changing an authored surface makes an older generated package detectable.
    """
    specification, descriptions, validation = package_modules
    model = specification.DeviceSpec("definition_fixture", "Fixture", "Parity fixture")
    model.part("base").box("surface", (0, 0, 0.01), (0.1, 0.1, 0.02))
    directory = tmp_path / model.identifier
    descriptions.InstrumentDescription(model).write(directory)
    validator = validation.PackageValidator(directory)
    validator.validate_definition(model)
    model.parts[0].shapes[0].center = (0, 0, 0.02)
    with pytest.raises(validation.InvalidPackage):
        validator.validate_definition(model)
