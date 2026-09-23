"""
Keep the deployment's collision asset list confined to the selected checkout.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree

import pytest


# %% deployment manifest
@pytest.fixture
def collision_description(fixture_scene: Path) -> Path:
    """
    Give the existing miniature scene a repeated reference to its mesh.
    """
    description = fixture_scene / "scenes" / "fixture" / "robot.urdf"
    document = ElementTree.parse(description)
    link = document.getroot().find("link")
    for kind in ("visual", "collision"):
        geometry = ElementTree.SubElement(
            ElementTree.SubElement(link, kind), "geometry"
        )
        ElementTree.SubElement(geometry, "mesh", filename="milk.stl")
    document.write(description)
    return description


def manifest(repository: Path, description: Path) -> subprocess.CompletedProcess:
    """
    Run the deployment file-list command without invoking SSH or rsync.
    """
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "laboratory-deployment"
        / "robot_assets.py"
    )
    return subprocess.run(
        [sys.executable, str(script), str(repository), str(description)],
        capture_output=True,
        text=True,
    )


def test_collision_manifest_contains_only_referenced_mesh(
    fixture_scene: Path, collision_description: Path
) -> None:
    """
    The robot export lists a shared mesh once and omits unrelated scene files.
    """
    result = manifest(fixture_scene, collision_description)
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        str((collision_description.parent / "milk.stl").relative_to(fixture_scene))
    ]


def test_missing_collision_mesh_stops_deployment(
    fixture_scene: Path, collision_description: Path
) -> None:
    """
    Missing geometry fails before an incomplete simulation can be deployed.
    """
    (collision_description.parent / "milk.stl").unlink()
    result = manifest(fixture_scene, collision_description)
    assert result.returncode == 1
    assert result.stdout == ""


def test_collision_manifest_rejects_mesh_outside_checkout(
    fixture_scene: Path, collision_description: Path
) -> None:
    """
    A mesh symlink cannot include unrelated host files in the deployment.
    """
    outside = fixture_scene.parent / "outside-mesh.stl"
    mesh = collision_description.parent / "milk.stl"
    mesh.rename(outside)
    mesh.symlink_to(outside)
    result = manifest(fixture_scene, collision_description)
    assert result.returncode == 1
    assert result.stdout == ""
