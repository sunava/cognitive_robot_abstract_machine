"""
Portable workshop articulation, loose fasteners and pilot-hole geometry.
"""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

# %% portable bundle fixture


@pytest.fixture
def workshop_directory(tmp_path: Path) -> Path:
    """
    Build descriptions without requiring Blender or graphics hardware.
    """
    builder = Path(__file__).resolve().parents[2] / "cramera/tools/workshop/bundle.py"
    result = subprocess.run(
        [sys.executable, str(builder), "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return tmp_path


# %% independent objects and articulated handles


def test_parts_are_independent_bodies_with_initial_poses(
    workshop_directory: Path,
) -> None:
    """
    Each screw, nail and open box can move without moving its neighbours.
    """
    scene = json.loads((workshop_directory / "scene.json").read_text())
    trajectory = json.loads((workshop_directory / scene["trajectory"]).read_text())
    expected = {
        item["key"] for item in scene["workshop"]["parts"] + scene["workshop"]["bins"]
    }
    assert {item["key"] for item in scene["objects"]} == expected
    assert len(scene["workshop"]["parts"]) == 10
    for item in scene["objects"]:
        model = ElementTree.parse(workshop_directory / item["urdf"])
        assert model.find("link").attrib["name"] == item["key"]
        assert len(model.findall("link")) == 1
        assert not model.findall("joint")
        assert len(model.findall(".//collision")) >= 2
        assert trajectory["objects"][0][item["key"]] == item["spawn"]


def test_drawer_handles_follow_bounded_sliding_joints(workshop_directory: Path) -> None:
    """
    A handle's fixed child frame follows its own independently sliding drawer.
    """
    scene = json.loads((workshop_directory / "scene.json").read_text())
    model = ElementTree.parse(workshop_directory / "environment.urdf")
    assert len(scene["workshop"]["drawers"]) == 3
    for drawer in scene["workshop"]["drawers"]:
        slide = model.find(f"joint[@name='{drawer['joint']}']")
        assert slide.attrib["type"] == "prismatic"
        assert slide.find("axis").attrib["xyz"] == "0 -1 0"
        assert float(slide.find("limit").attrib["upper"]) == drawer["open"]
        handle_joint = next(
            j
            for j in model.findall("joint")
            if j.find("child").attrib["link"] == drawer["handle"]
        )
        assert handle_joint.attrib["type"] == "fixed"
        assert (
            handle_joint.find("parent").attrib["link"]
            == slide.find("child").attrib["link"]
        )


def test_bin_cavity_and_assembly_socket_are_not_solid_boxes(
    workshop_directory: Path,
) -> None:
    """
    Separate walls leave open space for loose parts and the inserted screw.
    """
    box = ElementTree.parse(workshop_directory / "screw_box.urdf")
    assert len(box.findall(".//collision")) == 5
    model = ElementTree.parse(workshop_directory / "environment.urdf")
    assembly = model.find("link[@name='workshop_assembly']")
    collisions = assembly.findall("collision")
    assert len(collisions) >= 16
    for collision in collisions:
        x, y, _ = map(float, collision.find("origin").attrib["xyz"].split())
        assert x * x + y * y > 0
