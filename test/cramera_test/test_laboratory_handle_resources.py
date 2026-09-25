"""
Portable handle frames stay aligned with the authored grip geometry.
"""

from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest
import trimesh

from cramera.laboratory_devices import LaboratoryDevice, LaboratoryDeviceResource
from cramera.laboratory_handles import LaboratoryHandle
from cramera.laboratory_world import LaboratoryEnvironment

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_device_resources import device_resources


# %% frame alignment with the existing Blender geometry
@pytest.mark.parametrize(
    "handle,parent,asset,mesh",
    [
        (LaboratoryHandle.DRAWER, "laboratory_drawer", "drawer", "Drawer pull"),
        (
            LaboratoryHandle.CABINET_LOWER,
            "laboratory_room",
            "environment",
            "Lower cabinet pull",
        ),
        (
            LaboratoryHandle.CABINET_MIDDLE,
            "laboratory_room",
            "environment",
            "Cabinet door pull.001",
        ),
        (
            LaboratoryHandle.CABINET_RIGHT,
            "laboratory_room",
            "environment",
            "Cabinet door pull",
        ),
        (
            LaboratoryHandle.CENTRIFUGE_LID,
            "laboratory_centrifuge_lid",
            "centrifuge_lid",
            "Centrifuge lid grip",
        ),
        (
            LaboratoryHandle.REFRIGERATOR_DOOR,
            "laboratory_refrigerator_door",
            "refrigerator_door",
            "Refrigerator handle",
        ),
    ],
)
def test_generated_handle_frame_is_at_the_visible_grip_center(
    laboratory_directory: Path,
    handle: LaboratoryHandle,
    parent: str,
    asset: str,
    mesh: str,
) -> None:
    """
    A fixed child frame identifies the actual handle without duplicating its mesh.
    """
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    link = description.find(f"link[@name='{handle}']")
    assert link is not None
    joint = next(
        item
        for item in description.findall("joint")
        if item.find("child").attrib["link"] == handle
    )
    assert joint.attrib["type"] == "fixed"
    assert joint.find("parent").attrib["link"] == parent
    model = trimesh.load(
        LaboratoryEnvironment.resolve_directory() / "assets" / f"{asset}.glb",
        force="scene",
        process=False,
        merge_primitives=True,
    )
    parent_T_mesh, geometry = model.graph[mesh]
    vertices = trimesh.transform_points(
        model.geometry[geometry].vertices, parent_T_mesh
    )
    expected = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    np.testing.assert_allclose(
        np.fromstring(joint.find("origin").attrib["xyz"], sep=" "), expected, atol=1e-7
    )
    np.testing.assert_array_equal(
        np.fromstring(joint.find("origin").attrib["rpy"], sep=" "), np.zeros(3)
    )
    assert list(link) == []


# %% reusable device descriptions
@pytest.mark.parametrize(
    "device,handle",
    [
        (LaboratoryDevice.CENTRIFUGE, LaboratoryHandle.CENTRIFUGE_LID),
        (LaboratoryDevice.REFRIGERATOR, LaboratoryHandle.REFRIGERATOR_DOOR),
    ],
)
def test_exported_device_retains_its_handle_frame(
    device_resources: tuple[Path, Path],
    device: LaboratoryDevice,
    handle: LaboratoryHandle,
) -> None:
    """
    Detaching an instrument from the room preserves its named handle transform.
    """
    output, source = device_resources
    standalone = ElementTree.parse(LaboratoryDeviceResource(device, output).urdf)
    original = ElementTree.parse(source / "environment.urdf")
    assert standalone.find(f"link[@name='{handle}']") is not None
    expected = next(
        item
        for item in original.findall("joint")
        if item.find("child").attrib["link"] == handle
    )
    actual = standalone.find(f"joint[@name='{expected.attrib['name']}']")
    assert (
        ElementTree.tostring(actual).strip() == ElementTree.tostring(expected).strip()
    )


def test_installed_lab_and_device_resources_include_named_handles() -> None:
    """
    Installed resources expose the same frames even without rerunning the builder.
    """
    description = ElementTree.parse(
        LaboratoryEnvironment.resolve_directory() / "environment.urdf"
    )
    assert set(LaboratoryHandle) <= {
        link.attrib["name"] for link in description.findall("link")
    }
    for device, handle in (
        (LaboratoryDevice.CENTRIFUGE, LaboratoryHandle.CENTRIFUGE_LID),
        (LaboratoryDevice.REFRIGERATOR, LaboratoryHandle.REFRIGERATOR_DOOR),
    ):
        assert (
            ElementTree.parse(LaboratoryDeviceResource(device).urdf).find(
                f"link[@name='{handle}']"
            )
            is not None
        )
