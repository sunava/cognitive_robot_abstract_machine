"""Geometry and state contracts of the manually operated laboratory bundle."""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

# %% generated scene fixture

BUILDER = Path(__file__).resolve().parents[2] / "cramera/tools/laboratory/bundle.py"
"""Standalone bundle builder exercised without Blender or network access."""


@pytest.fixture
def laboratory_directory(tmp_path: Path) -> Path:
    """Create the prototype description in an isolated output directory."""
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return tmp_path


class TestSeparateLaboratoryObjects:
    """Each tube and the stopper retain an independent pose and collision model."""

    def test_independent_objects_match_initial_slot_occupancy(
        self, laboratory_directory: Path
    ) -> None:
        scene = json.loads((laboratory_directory / "scene.json").read_text())
        trajectory = json.loads(
            (laboratory_directory / scene["trajectory"]).read_text()
        )
        objects = {item["key"]: item for item in scene["objects"]}
        workbench = scene["laboratory"]
        slots = {slot["id"]: slot["pose"] for slot in workbench["rackSlots"]}
        assert len(objects) == len(workbench["tubes"]) + 1
        assert len({tube["slot"] for tube in workbench["tubes"]}) == len(
            workbench["tubes"]
        )
        for tube in workbench["tubes"]:
            assert objects[tube["key"]]["spawn"] == slots[tube["slot"]]
            assert trajectory["objects"][0][tube["key"]] == slots[tube["slot"]]
            description = ElementTree.parse(
                laboratory_directory / objects[tube["key"]]["urdf"]
            )
            assert len(description.findall("link")) == 1
            assert len(description.findall(".//collision")) > 1
            assert not description.findall("joint")
        assert (
            objects[workbench["stopper"]["key"]]["spawn"]
            == workbench["stopper"]["home"]
        )


class TestDrawerArticulation:
    """The cabinet contains a bounded sliding joint rather than a baked pose."""

    def test_drawer_moves_along_its_declared_axis(
        self, laboratory_directory: Path
    ) -> None:
        scene = json.loads((laboratory_directory / "scene.json").read_text())
        drawer = scene["laboratory"]["drawer"]
        description = ElementTree.parse(laboratory_directory / "environment.urdf")
        joint = description.find(f"joint[@name='{drawer['joint']}']")
        assert joint.attrib["type"] == "prismatic"
        assert joint.find("axis").attrib["xyz"] == "0 -1 0"
        assert float(joint.find("limit").attrib["lower"]) == 0
        assert float(joint.find("limit").attrib["upper"]) == drawer["open"]
        assert joint.find("child").attrib["link"] == "laboratory_drawer"
        assert scene["models"][0]["preserveMaterials"] is True


class TestRackInsertionClearance:
    """Collision geometry leaves a path through each slot for a whole tube."""

    def test_tube_cross_sections_clear_the_top_plate(
        self, laboratory_directory: Path
    ) -> None:
        scene = json.loads((laboratory_directory / "scene.json").read_text())
        metadata = json.loads((laboratory_directory / "semantics.json").read_text())
        radius = metadata["tube"]["outerRadius"]
        rack_origin = metadata["rack"]["origin"]
        description = ElementTree.parse(laboratory_directory / "environment.urdf")
        collisions = description.findall("link[@name='laboratory_rack']/collision")
        top_collisions = [
            item for item in collisions if item.attrib["name"].startswith("slot_web")
        ]
        assert top_collisions
        for slot in scene["laboratory"]["rackSlots"]:
            horizontal = [
                slot["pose"][index] - rack_origin[index] for index in range(2)
            ]
            for collision in top_collisions:
                center = [
                    float(value)
                    for value in collision.find("origin").attrib["xyz"].split()
                ]
                size = [
                    float(value)
                    for value in collision.find("geometry/box").attrib["size"].split()
                ]
                distance_squared = sum(
                    max(abs(horizontal[index] - center[index]) - size[index] / 2, 0)
                    ** 2
                    for index in range(2)
                )
                assert distance_squared > radius**2


class TestPrototypeDisclosure:
    """An initial state must not be presented as a recorded robot execution."""

    def test_scene_describes_manual_kinematics(
        self, laboratory_directory: Path
    ) -> None:
        scene = json.loads((laboratory_directory / "scene.json").read_text())
        assert scene["validation"]["mode"] == "manual_kinematic"
        assert scene["validation"]["robotExecutionVerified"] is False
        assert scene["validation"]["contactPhysicsVerified"] is False
        assert scene["segments"] == []
        assert scene["actions"] == []
