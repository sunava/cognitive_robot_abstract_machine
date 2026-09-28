"""
Moving workshop parts clear their supports and expose real tool interfaces.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

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

    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> CollisionBox:
        """
        Transform an orthogonal box into its enclosing world-space bounds.
        """
        center = translation + rotation @ ((self.minimum + self.maximum) / 2)
        extent = np.abs(rotation) @ ((self.maximum - self.minimum) / 2)
        return CollisionBox(self.name, center - extent, center + extent)


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


def test_horizontal_assembly_is_supported_and_clear_of_the_resting_driver(
    workshop_directory: Path,
) -> None:
    """
    The aisle-facing pilot rests in a jig without intersecting the nearby blade.
    """
    scene = json.loads((workshop_directory / "scene.json").read_text())
    assembly = scene["workshop"]["assembly"]
    driver = scene["workshop"]["driver"]
    room = ElementTree.parse(workshop_directory / "environment.urdf")
    worktop = CollisionBox.from_element(
        room.find("link/collision[@name='worktop']"), np.zeros(3)
    )
    origin = np.asarray(assembly["home"][:3])
    rotation = Rotation.from_quat(assembly["home"][3:]).as_matrix()
    np.testing.assert_allclose(rotation @ [0, 0, 1], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(
        assembly["socketPose"][:3],
        origin + rotation @ assembly["localSocket"][:3],
        atol=1e-12,
    )
    np.testing.assert_allclose(assembly["socketPose"][3:], assembly["home"][3:])
    model = ElementTree.parse(workshop_directory / "workshop_assembly.urdf")
    parts = [
        CollisionBox.from_element(collision, np.zeros(3)).transform(rotation, origin)
        for collision in model.findall("link/collision")
    ]
    for part in parts:
        assert np.all(part.minimum[:2] >= worktop.minimum[:2])
        assert np.all(part.maximum[:2] <= worktop.maximum[:2])
        assert part.minimum[2] >= worktop.maximum[2]
    supports = [
        CollisionBox.from_element(collision, np.zeros(3))
        for collision in room.findall("link/collision")
        if collision.attrib["name"].startswith("assembly_jig")
    ]
    assert supports
    floor = min(part.minimum[2] for part in parts)
    assert any(abs(part.maximum[2] - floor) < 1e-8 for part in supports)
    driver_model = ElementTree.parse(workshop_directory / "driver.urdf")
    blade = driver_model.find("link/collision[@name='flat_blade']")
    local = CollisionBox.from_element(blade, np.zeros(3))
    resting_blade = local.transform(
        Rotation.from_quat(driver["home"][3:]).as_matrix(),
        np.asarray(driver["home"][:3]),
    )
    assert not any(resting_blade.intersects(part) for part in parts)


# %% shelf pickup and bench destination


def test_screw_box_starts_on_the_right_shelf_with_a_clear_front_exit(
    workshop_directory: Path,
) -> None:
    """
    The loaded box rests above a shelf and clears every support when pulled forward.
    """
    scene = json.loads((workshop_directory / "scene.json").read_text())
    item = scene["workshop"]["bins"][0]
    room = ElementTree.parse(workshop_directory / "environment.urdf")
    supports = [
        CollisionBox.from_element(collision, np.zeros(3))
        for collision in room.find("link[@name='workshop_environment']").findall(
            "collision"
        )
    ]
    worktop = next(part for part in supports if part.name == "worktop")
    origin = np.array(item["home"][:3])
    assert origin[0] > worktop.maximum[0]
    shelf = next(
        part
        for part in supports
        if part.name.startswith("shelf_board")
        and np.all(origin[:2] > part.minimum[:2])
        and np.all(origin[:2] < part.maximum[:2])
        and 0 < origin[2] - part.maximum[2] < 0.003
    )
    model = ElementTree.parse(workshop_directory / (item["key"] + ".urdf"))
    pieces = [
        CollisionBox.from_element(collision, origin)
        for collision in model.find(f"link[@name='{item['key']}']").findall("collision")
    ]
    handle_joint = model.find(f"joint[@name='{item['handle']}_joint']")
    handle_origin = origin + np.fromstring(
        handle_joint.find("origin").attrib["xyz"], sep=" "
    )
    pieces.extend(
        CollisionBox.from_element(collision, handle_origin)
        for collision in model.find(f"link[@name='{item['handle']}']").findall(
            "collision"
        )
    )
    exit_displacement = np.array([0, shelf.minimum[1] - origin[1] - 0.35, 0])
    assert [
        (part.name, support.name)
        for part in pieces
        for support in supports
        if part.sweep(exit_displacement).intersects(support)
    ] == []
    bench_pose = np.array(item["benchPose"][:3])
    np.testing.assert_allclose(bench_pose[2], worktop.maximum[2], rtol=0, atol=1e-12)
    assert np.all(bench_pose[:2] > worktop.minimum[:2])
    assert np.all(bench_pose[:2] < worktop.maximum[:2])
    bench_floor = CollisionBox.from_element(
        model.find("link/collision[@name='floor']"), bench_pose
    )
    assert np.all(bench_floor.minimum[:2] >= worktop.minimum[:2])
    assert np.all(bench_floor.maximum[:2] <= worktop.maximum[:2])


def test_screw_contents_start_inside_the_shelf_box(workshop_directory: Path) -> None:
    """
    Loose screws retain their own poses above the moved container floor.
    """
    scene = json.loads((workshop_directory / "scene.json").read_text())
    item = scene["workshop"]["bins"][0]
    origin = np.array(item["home"][:3])
    model = ElementTree.parse(workshop_directory / (item["key"] + ".urdf"))
    floor = CollisionBox.from_element(
        model.find("link/collision[@name='floor']"), origin
    )
    for part in scene["workshop"]["parts"]:
        if part["kind"] != "screw":
            continue
        screw = ElementTree.parse(workshop_directory / (part["key"] + ".urdf"))
        head_radius = float(
            screw.find("link/collision[@name='head_base']/geometry/cylinder").attrib[
                "radius"
            ]
        )
        position = np.array(part["home"][:3])
        assert position[2] - head_radius > floor.maximum[2]
        assert np.all(position[:2] > floor.minimum[:2])
        assert np.all(position[:2] < floor.maximum[:2])


def test_stored_timber_has_collision_geometry_above_the_pickup_shelf(
    workshop_directory: Path,
) -> None:
    """
    Fixed stock occupies the top shelf, leaving the fastener pickup shelf empty.
    """
    room = ElementTree.parse(workshop_directory / "environment.urdf")
    supports = [
        CollisionBox.from_element(collision, np.zeros(3))
        for collision in room.find("link[@name='workshop_environment']").findall(
            "collision"
        )
    ]
    shelf_top = max(
        part.maximum[2] for part in supports if part.name.startswith("shelf_board")
    )
    stock = [part for part in supports if part.name.startswith("shelf_stock")]
    assert len(stock) == 5
    assert all(part.minimum[2] > shelf_top for part in stock)
    assert not any(
        part.intersects(neighbour)
        for index, part in enumerate(stock)
        for neighbour in stock[index + 1 :]
    )


def test_driver_cradle_leaves_the_handle_grasp_clear(workshop_directory: Path) -> None:
    """
    Two elevated supports expose the middle handle to an axial finger approach.
    """
    scene = json.loads((workshop_directory / "scene.json").read_text())
    driver = scene["workshop"]["driver"]
    room = ElementTree.parse(workshop_directory / "environment.urdf")
    components = [
        CollisionBox.from_element(collision, np.zeros(3))
        for collision in room.find("link[@name='workshop_environment']").findall(
            "collision"
        )
    ]
    worktop = next(part for part in components if part.name == "worktop")
    origin = np.array(driver["home"][:3])
    assert origin[2] - worktop.maximum[2] > 0.10
    cradle = [part for part in components if part.name.startswith("driver_cradle")]
    assert len(cradle) == 5
    for part in cradle:
        assert np.all(part.minimum[:2] >= worktop.minimum[:2])
        assert np.all(part.maximum[:2] <= worktop.maximum[:2])
    grasp = origin + np.array([driver["grasp"][2], 0, 0])
    finger_space = CollisionBox(
        "handle grasp clearance",
        grasp - np.array([0.022, 0.045, 0.045]),
        grasp + np.array([0.022, 0.045, 0.045]),
    )
    assert not any(part.intersects(finger_space) for part in cradle)


def test_driver_guides_allow_small_lateral_lift_error(workshop_directory: Path) -> None:
    """
    The shaft can lift with three millimetres of lateral controller error.
    """
    scene = json.loads((workshop_directory / "scene.json").read_text())
    origin = np.asarray(scene["workshop"]["driver"]["home"][:3])
    environment = ElementTree.parse(workshop_directory / "environment.urdf")
    guides = [
        CollisionBox.from_element(collision, np.zeros(3))
        for collision in environment.find("link[@name='workshop_environment']").findall(
            "collision"
        )
        if collision.attrib["name"].startswith("driver_cradle")
        and float(collision.find("origin").attrib["xyz"].split()[2]) == origin[2]
    ]
    assert len(guides) == 2
    driver = ElementTree.parse(workshop_directory / "driver.urdf")
    shaft = driver.find("link/collision[@name='shaft']/geometry/cylinder")
    radius = float(shaft.attrib["radius"])
    allowed_lateral_error = 0.003
    for guide in guides:
        assert (
            min(abs(guide.minimum[1] - origin[1]), abs(guide.maximum[1] - origin[1]))
            > radius + allowed_lateral_error
        )
