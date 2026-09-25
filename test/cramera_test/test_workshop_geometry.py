"""
Moving workshop parts clear their supports and expose real tool interfaces.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .test_workshop_bundle import workshop_directory


# %% continuous drawer clearance
@dataclass(frozen=True)
class CollisionBox:
    """
    Axis-aligned collision envelope in the environment frame.
    """

    name: str
    """
    URDF collision name retained in assertion diagnostics.
    """

    minimum: np.ndarray
    """
    Lower world-coordinate corner.
    """

    maximum: np.ndarray
    """
    Upper world-coordinate corner.
    """

    @classmethod
    def from_element(
        cls, collision: ElementTree.Element, translation: np.ndarray
    ) -> CollisionBox:
        """
        Resolve the translation of one axis-aligned URDF box.
        """
        origin = collision.find("origin")
        centre = translation + np.fromstring(origin.attrib["xyz"], sep=" ")
        size = np.fromstring(collision.find("geometry/box").attrib["size"], sep=" ")
        return cls(collision.attrib["name"], centre - size / 2, centre + size / 2)

    def sweep(self, displacement: np.ndarray) -> CollisionBox:
        """
        Cover every intermediate configuration of a straight prismatic motion.
        """
        return CollisionBox(
            self.name,
            np.minimum(self.minimum, self.minimum + displacement),
            np.maximum(self.maximum, self.maximum + displacement),
        )

    def intersects(self, other: CollisionBox) -> bool:
        """
        Distinguish actual volume penetration from touching boundaries.
        """
        return bool(
            np.all(
                np.minimum(self.maximum, other.maximum)
                - np.maximum(self.minimum, other.minimum)
                > 1e-8
            )
        )


def test_drawers_clear_fixed_structure_through_their_entire_travel(
    workshop_directory: Path,
) -> None:
    """
    Drawer floors, walls, fronts and handles never sweep through fixed supports.
    """
    scene = json.loads((workshop_directory / "scene.json").read_text())
    model = ElementTree.parse(workshop_directory / "environment.urdf")
    static = [
        CollisionBox.from_element(collision, np.zeros(3))
        for collision in model.find("link[@name='workshop_environment']").findall(
            "collision"
        )
        if collision.find("geometry/box") is not None
    ]
    overlaps = []
    for drawer in scene["workshop"]["drawers"]:
        joint = model.find(f"joint[@name='{drawer['joint']}']")
        axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
        travel = float(joint.find("limit").attrib["upper"])
        origin = np.array(drawer["origin"])
        moving = [
            CollisionBox.from_element(collision, origin)
            for collision in model.find(f"link[@name='{drawer['link']}']").findall(
                "collision"
            )
        ]
        handle_joint = model.find(f"joint[@name='{drawer['handle']}_joint']")
        handle_origin = origin + np.fromstring(
            handle_joint.find("origin").attrib["xyz"], sep=" "
        )
        moving.extend(
            CollisionBox.from_element(collision, handle_origin)
            for collision in model.find(f"link[@name='{drawer['handle']}']").findall(
                "collision"
            )
        )
        overlaps.extend(
            (drawer["id"], part.name, support.name)
            for part in moving
            for support in static
            if part.sweep(axis * travel).intersects(support)
        )
    assert overlaps == []


# %% independent workpiece and matching screwdriver


def test_workpiece_and_driver_are_independent_of_the_room(
    workshop_directory: Path,
) -> None:
    """
    The workpiece socket and screwdriver tip follow their own movable bodies.
    """
    scene = json.loads((workshop_directory / "scene.json").read_text())
    room = ElementTree.parse(workshop_directory / "environment.urdf")
    objects = {item["key"]: item for item in scene["objects"]}
    for kind, frame_field in (("assembly", "socketFrame"), ("driver", "bitFrame")):
        metadata = scene["workshop"][kind]
        item = objects[metadata["key"]]
        assert room.find(f"link[@name='{metadata['key']}']") is None
        model = ElementTree.parse(workshop_directory / item["urdf"])
        root = model.find(f"link[@name='{metadata['key']}']")
        assert root.find("inertial/mass") is not None
        frame = model.find(f"link[@name='{metadata[frame_field]}']")
        assert frame is not None
        parent_joint = next(
            joint
            for joint in model.findall("joint")
            if joint.find("child").attrib["link"] == metadata[frame_field]
        )
        assert parent_joint.find("parent").attrib["link"] == metadata["key"]
        assert item["spawn"] == metadata["home"]


def test_screw_head_recess_accepts_the_declared_flat_blade(
    workshop_directory: Path,
) -> None:
    """
    Collision geometry leaves the blade recess open while retaining both cheeks.
    """
    scene = json.loads((workshop_directory / "scene.json").read_text())
    driver = scene["workshop"]["driver"]
    screw = scene["workshop"]["screw"]
    blade = np.array(driver["bladeSize"])
    slot = np.array(screw["slotSize"])
    assert np.all(blade[:2] < slot[:2])
    model = ElementTree.parse(workshop_directory / "screw_1.urdf")
    depth = screw["headTop"] - slot[2] / 2
    recess = CollisionBox(
        "blade envelope",
        np.array([-blade[0] / 2, -blade[1] / 2, depth - slot[2] / 3]),
        np.array([blade[0] / 2, blade[1] / 2, depth + slot[2] / 3]),
    )
    cheeks = [
        CollisionBox.from_element(collision, np.zeros(3))
        for collision in model.findall("link/collision")
        if collision.find("geometry/box") is not None
    ]
    assert len(cheeks) >= 4
    assert not any(recess.intersects(cheek) for cheek in cheeks)
    assert any(cheek.minimum[1] >= slot[1] / 2 - 1e-8 for cheek in cheeks)
    assert any(cheek.maximum[1] <= -slot[1] / 2 + 1e-8 for cheek in cheeks)
    for collision in model.findall("link/collision"):
        cylinder = collision.find("geometry/cylinder")
        if cylinder is not None:
            centre = np.fromstring(collision.find("origin").attrib["xyz"], sep=" ")
            assert (
                centre[2] + float(cylinder.attrib["length"]) / 2
                <= screw["headTop"] - slot[2] + 1e-8
            )


def test_seated_screw_stays_above_the_bench(workshop_directory: Path) -> None:
    """
    The complete driven shaft fits inside the block without entering its support.
    """
    scene = json.loads((workshop_directory / "scene.json").read_text())
    assembly = scene["workshop"]["assembly"]
    assert assembly["localMouth"][2] - assembly["maximumDepth"] > 0
    assert 0 < assembly["localSocket"][2] < assembly["localMouth"][2]
