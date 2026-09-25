"""
Movable rack artwork remains separate and fits the prepared drawer opening.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import json
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest
import trimesh

from cramera.laboratory_world import CabinetPart, CabinetStructure

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_geometry import load_geometry


# %% material-preserving asset preparation
@pytest.fixture
def prepared_rack_assets(laboratory_directory: Path) -> Path:
    """
    Split current room artwork using the resource preparation command.
    """
    repository = Path(__file__).resolve().parents[2]
    shutil.copytree(
        repository / "cramera/src/cramera/resources/aicor_medical_lab/assets",
        laboratory_directory / "assets",
        dirs_exist_ok=True,
    )
    command = [
        sys.executable,
        str(repository / "cramera/tools/laboratory/device_resources.py"),
        "--source",
        str(laboratory_directory),
        "--output",
        str(laboratory_directory / "devices"),
        "--prepare-rack",
    ]
    first = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert first.returncode == 0, first.stdout + first.stderr
    files = [
        laboratory_directory / "assets" / name
        for name in ("environment.glb", "tube_rack.glb", "drawer.glb")
    ]
    before = [item.read_bytes() for item in files]
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert [item.read_bytes() for item in files] == before
    return laboratory_directory


def test_room_contains_no_stationary_duplicate_of_the_movable_rack(
    prepared_rack_assets: Path,
) -> None:
    """
    Only the separate rack body carries its sockets and branded artwork.
    """
    room = trimesh.load_scene(prepared_rack_assets / "assets/environment.glb")
    rack = trimesh.load_scene(
        prepared_rack_assets / "assets/tube_rack.glb", merge_primitives=True
    )
    assert not any(
        name.startswith(("Rack", "Socket identifier", "Tube rack"))
        for name in room.graph.nodes_geometry
    )
    assert "Rack AICOR logo" in rack.graph.nodes_geometry
    assert "Rack perforated plate" in rack.graph.nodes_geometry
    model = ElementTree.parse(prepared_rack_assets / "environment.urdf")
    visuals = model.findall("link[@name='laboratory_rack']/visual/geometry/mesh")
    assert [visual.attrib["filename"] for visual in visuals] == ["assets/tube_rack.glb"]


def test_carry_bar_and_mounts_match_their_contact_shapes(
    prepared_rack_assets: Path,
) -> None:
    """
    The grasp surfaces displayed in the browser occupy the authored collisions.
    """
    model = ElementTree.parse(prepared_rack_assets / "environment.urdf")
    rack = trimesh.load_scene(prepared_rack_assets / "assets/tube_rack.glb")
    for name, geometry in (
        ("rack_handle", "Rack carry handle"),
        ("rack_handle_mount_0", "Rack carry mounting"),
        ("rack_handle_mount_1", "Rack carry mounting.001"),
    ):
        collision = model.find(
            f"link[@name='laboratory_rack']/collision[@name='{name}']"
        )
        center = np.fromstring(collision.find("origin").attrib["xyz"], sep=" ")
        size = np.fromstring(collision.find("geometry/box").attrib["size"], sep=" ")
        transform, mesh_name = rack.graph[geometry]
        mesh = rack.geometry[mesh_name].copy()
        mesh.apply_transform(transform)
        np.testing.assert_allclose(
            mesh.bounds, [center - size / 2, center + size / 2], atol=1e-7
        )


def test_each_round_bottom_sample_has_a_visible_lower_guide(
    prepared_rack_assets: Path,
) -> None:
    """
    Each upper socket has a hollow guide around the rounded sample foot.
    """
    scene = json.loads((prepared_rack_assets / "scene.json").read_text())
    rack = trimesh.load_scene(prepared_rack_assets / "assets/tube_rack.glb")
    guides = [
        name
        for name in rack.graph.nodes_geometry
        if name.startswith("Rack lower guide")
    ]
    assert len(guides) == len(scene["laboratory"]["rackSlots"])


def test_lower_guides_preserve_the_open_bore_and_widened_insertion_mouth(
    prepared_rack_assets: Path,
) -> None:
    """
    Exported guide walls retain the shared hollow revolved profile at each slot.
    """
    dimensions = load_geometry().RackDimensions()
    expected = np.unique(
        np.round(
            [(point.radius, point.height) for point in dimensions.guide_profile()], 7
        ),
        axis=0,
    )
    rack = trimesh.load_scene(prepared_rack_assets / "assets/tube_rack.glb")
    for column, horizontal in enumerate(dimensions.columns):
        for row, depth in enumerate(dimensions.rows):
            transform, mesh_name = rack.graph[f"Rack lower guide {column}_{row}"]
            mesh = rack.geometry[mesh_name].copy()
            mesh.apply_transform(transform)
            radial = np.linalg.norm(mesh.vertices[:, :2] - (horizontal, depth), axis=1)
            profile = np.unique(
                np.round(np.column_stack((radial, mesh.vertices[:, 2])), 7), axis=0
            )
            np.testing.assert_array_equal(profile, expected)


def test_drawer_visuals_leave_the_actual_floor_clear_for_the_loaded_rack(
    prepared_rack_assets: Path,
) -> None:
    """
    Removed dividers and mat cannot visually pass through the seated rack.
    """
    drawer = trimesh.load_scene(prepared_rack_assets / "assets/drawer.glb")
    assert not any(
        name.startswith(("Drawer mat", "Tray divider"))
        for name in drawer.graph.nodes_geometry
    )
    assert "Drawer floor" in drawer.graph.nodes_geometry


def test_raised_crossmember_matches_the_semantic_contact_panel(
    prepared_rack_assets: Path,
) -> None:
    """
    Visual clearance agrees with the collision geometry above the drawer.
    """
    panel = next(
        item
        for item in CabinetStructure().panels()
        if item.part == CabinetPart.CROSS_MEMBER
    )
    room = trimesh.load_scene(prepared_rack_assets / "assets/environment.glb")
    transform, mesh_name = room.graph["Upper cross member"]
    mesh = room.geometry[mesh_name].copy()
    mesh.apply_transform(transform)
    np.testing.assert_allclose(
        mesh.bounds.mean(axis=0), panel.origin.to_np()[:3, 3], atol=1e-7
    )
