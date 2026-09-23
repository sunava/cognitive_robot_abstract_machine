"""
Authored centrifuge articulation and open sample-holder geometry.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from .test_laboratory_bundle import laboratory_directory

# %% articulated instrument contract


def test_chamber_detection_matches_the_authored_collision_boundary(
    laboratory_directory: Path,
) -> None:
    """
    Runtime loose-glass detection uses the same open bowl as the collision model.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    instrument = scene["laboratory"]["centrifuge"]
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    wall = description.find(
        f"link[@name='{instrument['body']}']/collision[@name='chamber_wall_0']"
    )
    center = [float(value) for value in wall.find("origin").attrib["xyz"].split()]
    size = [float(value) for value in wall.find("geometry/box").attrib["size"].split()]
    assert instrument["chamberRadius"] == pytest.approx(center[0] - size[0] / 2)
    assert instrument["chamberHeight"] == pytest.approx(
        center[2] + size[2] / 2 - instrument["rotorOrigin"][2]
    )


def test_centrifuge_declares_separate_rotor_and_lid(laboratory_directory: Path) -> None:
    """
    The instrument exposes independent joints and material-bearing meshes.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    instrument = scene["laboratory"]["centrifuge"]
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    rotor = description.find(f"joint[@name='{instrument['rotorJoint']}']")
    lid = description.find(f"joint[@name='{instrument['lidJoint']}']")
    assert rotor.attrib["type"] == "continuous"
    assert rotor.find("axis").attrib["xyz"] == "0 0 1"
    assert lid.attrib["type"] == "revolute"
    assert float(lid.find("limit").attrib["lower"]) == 0.0
    assert float(lid.find("limit").attrib["upper"]) == instrument["openAngle"]
    for joint in (rotor, lid):
        link = description.find(f"link[@name='{joint.find('child').attrib['link']}']")
        assert link.find("visual/geometry/mesh") is not None
        assert link.find("inertial/mass") is not None
        assert joint.attrib["name"] in scene["models"][0]["movableJoints"]


def test_centrifuge_slots_have_opposite_balance_pairs(
    laboratory_directory: Path,
) -> None:
    """
    Each sample position has a diametrically opposite counterpart.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    instrument = scene["laboratory"]["centrifuge"]
    slots = {slot["id"]: slot for slot in instrument["slots"]}
    assert len(slots) == 4
    for slot in slots.values():
        opposite = slots[slot["opposite"]]
        assert opposite["opposite"] == slot["id"]
        assert slot["position"][:2] == pytest.approx(
            [-coordinate for coordinate in opposite["position"][:2]], abs=1e-12
        )
        assert slot["position"][2] == opposite["position"][2]


def test_sample_path_is_open_and_supported(laboratory_directory: Path) -> None:
    """
    A complete existing tube can descend into a hollow rotor sleeve.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    instrument = scene["laboratory"]["centrifuge"]
    semantics = json.loads((laboratory_directory / "semantics.json").read_text())
    radius = semantics["tube"]["outerRadius"]
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    rotor_joint = description.find(f"joint[@name='{instrument['rotorJoint']}']")
    rotor = description.find(
        f"link[@name='{rotor_joint.find('child').attrib['link']}']"
    )
    collisions = rotor.findall("collision")
    assert collisions
    for slot in instrument["slots"]:
        horizontal = slot["position"][:2]
        supported = False
        for collision in collisions:
            origin = collision.find("origin").attrib
            center = [float(value) for value in origin["xyz"].split()]
            shape = collision.find("geometry/box")
            if shape is None:
                continue
            size = [float(value) for value in shape.attrib["size"].split()]
            yaw = float(origin.get("rpy", "0 0 0").split()[2])
            delta = [horizontal[index] - center[index] for index in range(2)]
            local = (
                delta[0] * math.cos(yaw) + delta[1] * math.sin(yaw),
                -delta[0] * math.sin(yaw) + delta[1] * math.cos(yaw),
            )
            distance_squared = sum(
                max(abs(local[index]) - size[index] / 2, 0) ** 2 for index in range(2)
            )
            if center[2] + size[2] / 2 <= slot["position"][2] + 1e-9:
                supported |= distance_squared == 0
            else:
                assert distance_squared > radius**2
        assert supported


def test_lid_clears_the_full_tube_height(laboratory_directory: Path) -> None:
    """
    Closing the lid leaves headroom above the existing 150 mm tubes.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    instrument = scene["laboratory"]["centrifuge"]
    semantics = json.loads((laboratory_directory / "semantics.json").read_text())
    maximum_tube_top = max(
        instrument["rotorOrigin"][2] + slot["position"][2] + semantics["tube"]["height"]
        for slot in instrument["slots"]
    )
    assert instrument["lidOrigin"][2] > maximum_tube_top
