"""
Portable device resources preserve the authored laboratory geometry.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest
import trimesh

from .test_laboratory_bundle import laboratory_directory

# %% independent resource export


@pytest.fixture
def device_resources(laboratory_directory: Path) -> tuple[Path, Path]:
    """
    Export actual authored visuals alongside freshly generated descriptions.
    """
    repository = Path(__file__).resolve().parents[2]
    source = repository / "cramera/src/cramera/resources/precision_lab/assets"
    shutil.copytree(source, laboratory_directory / "assets", dirs_exist_ok=True)
    output = laboratory_directory / "devices"
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "cramera/tools/laboratory/device_resources.py"),
            "--source",
            str(laboratory_directory),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return output, laboratory_directory


def test_each_model_has_only_local_existing_asset_references(
    device_resources: tuple[Path, Path],
) -> None:
    """
    Each copied folder remains complete after the source laboratory disappears.
    """
    from cramera.laboratory_devices import LaboratoryDevice, LaboratoryDeviceResource

    output, source = device_resources
    shutil.rmtree(source / "assets")
    for device in LaboratoryDevice:
        resource = LaboratoryDeviceResource(device, output)
        description = ElementTree.parse(resource.urdf)
        links = {link.attrib["name"] for link in description.findall("link")}
        children = {
            joint.find("child").attrib["link"] for joint in description.findall("joint")
        }
        assert len(links - children) == 1
        for mesh in description.findall(".//mesh"):
            relative = Path(mesh.attrib["filename"])
            assert not relative.is_absolute()
            assert ".." not in relative.parts
            assert (resource.directory / relative).is_file()
        assert (resource.directory / "README.md").is_file()


def test_independent_models_preserve_collision_and_joint_contracts(
    device_resources: tuple[Path, Path],
) -> None:
    """
    Extraction keeps every contact primitive and articulated joint unchanged.
    """
    from cramera.laboratory_devices import LaboratoryDevice, LaboratoryDeviceResource

    output, source = device_resources
    original = ElementTree.parse(source / "environment.urdf")
    for device in LaboratoryDevice:
        resource = LaboratoryDeviceResource(device, output)
        description = ElementTree.parse(resource.urdf)
        for link in description.findall("link"):
            authored = original.find(f"link[@name='{link.attrib['name']}']")
            assert authored is not None
            for tag in ("collision", "inertial"):
                expected = [
                    ElementTree.tostring(item).strip() for item in authored.findall(tag)
                ]
                actual = [
                    ElementTree.tostring(item).strip() for item in link.findall(tag)
                ]
                assert actual == expected
        for joint in description.findall("joint"):
            authored = original.find(f"joint[@name='{joint.attrib['name']}']")
            assert (
                ElementTree.tostring(joint).strip()
                == ElementTree.tostring(authored).strip()
            )


@pytest.mark.parametrize(
    ("device", "mesh_name", "prefixes", "origin"),
    [
        (
            "scale",
            "scale.glb",
            ("Instrument base", "Balance", "Scale"),
            (0.65, 0.21, 0.968),
        ),
        (
            "tube_rack",
            "tube_rack.glb",
            ("Rack", "Socket identifier", "Tube rack"),
            (0.22, -0.04, 0.9),
        ),
    ],
)
def test_extracted_visuals_have_exact_device_local_geometry(
    device_resources: tuple[Path, Path],
    device: str,
    mesh_name: str,
    prefixes: tuple[str, ...],
    origin: tuple[float, float, float],
) -> None:
    """
    Static instruments contain their own meshes with the laboratory offset removed.
    """
    output, source = device_resources
    original = trimesh.load(
        source / "assets/environment.glb",
        force="scene",
        process=False,
        merge_primitives=True,
    )
    extracted = trimesh.load(
        output / device / "assets" / mesh_name,
        force="scene",
        process=False,
        merge_primitives=True,
    )
    expected = [
        name for name in original.graph.nodes_geometry if name.startswith(prefixes)
    ]
    assert set(extracted.graph.nodes_geometry) == set(expected)
    for name in expected:
        original_transform, original_geometry = original.graph[name]
        extracted_transform, extracted_geometry = extracted.graph[name]
        original_vertices = trimesh.transform_points(
            original.geometry[original_geometry].vertices, original_transform
        )
        extracted_vertices = trimesh.transform_points(
            extracted.geometry[extracted_geometry].vertices, extracted_transform
        )
        np.testing.assert_allclose(
            extracted_vertices, original_vertices - origin, atol=1e-7
        )
    assert (output / device / "assets" / mesh_name).stat().st_size < (
        source / "assets/environment.glb"
    ).stat().st_size / 2


def test_exported_folder_can_be_loaded_as_an_independent_cram_world(
    device_resources: tuple[Path, Path],
) -> None:
    """
    CRAM accepts each device without requiring room, robot, or other instruments.
    """
    from cramera.laboratory_devices import LaboratoryDevice, LaboratoryDeviceResource
    from semantic_digital_twin.adapters.urdf import URDFParser

    output, _ = device_resources
    for device in LaboratoryDevice:
        resource = LaboratoryDeviceResource(device, output)
        model = ElementTree.parse(resource.urdf)
        world = URDFParser.from_file(str(resource.urdf), prefix="").parse()
        assert {body.name.name for body in world.bodies} == {
            link.attrib["name"] for link in model.findall("link")
        }


def test_copy_resource_retains_complete_relative_model(
    device_resources: tuple[Path, Path], tmp_path: Path
) -> None:
    """
    The public resource API creates a reusable folder outside the package.
    """
    from cramera.laboratory_devices import LaboratoryDevice, LaboratoryDeviceResource

    output, _ = device_resources
    resource = LaboratoryDeviceResource(LaboratoryDevice.CENTRIFUGE, output)
    destination = resource.copy_to(tmp_path / "portable")
    assert destination == tmp_path / "portable" / resource.device.value
    for path in resource.directory.rglob("*"):
        if path.is_file():
            assert (
                destination / path.relative_to(resource.directory)
            ).read_bytes() == path.read_bytes()


def test_scale_support_has_visible_matching_contact_segments(
    device_resources: tuple[Path, Path],
) -> None:
    """
    The portable scale includes the collar normally supplied by the browser.
    """
    from cramera.laboratory_devices import LaboratoryDevice, LaboratoryDeviceResource

    output, _ = device_resources
    resource = LaboratoryDeviceResource(LaboratoryDevice.SCALE, output)
    description = ElementTree.parse(resource.urdf)
    segments = [
        item
        for item in description.findall(".//collision")
        if item.attrib["name"].startswith("weighing_holder_")
    ]
    assert segments
    for collision in segments:
        visual = description.find(f".//visual[@name='{collision.attrib['name']}']")
        assert visual is not None
        assert (
            visual.find("geometry/box").attrib == collision.find("geometry/box").attrib
        )
        assert visual.find("origin").attrib == collision.find("origin").attrib
