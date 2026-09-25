"""
The loaded sample rack fits in its drawer and exposes a physical carry handle.
"""

from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np

from cramera.laboratory_world import CabinetPart, CabinetStructure

from .test_laboratory_bundle import laboratory_directory


# %% movable rack and drawer clearance
def test_lower_guides_keep_round_bottom_samples_seated(
    laboratory_directory: Path,
) -> None:
    """
    Every upper aperture has a lower guide surrounding the straight vial wall.
    """
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    rack = description.find("link[@name='laboratory_rack']")
    for column in range(3):
        for row in range(2):
            guides = [
                collision
                for collision in rack.findall("collision")
                if collision.attrib["name"].startswith(f"lower_socket_{column}_{row}_")
            ]
            assert len(guides) >= 8
            for guide in guides:
                center = np.fromstring(guide.find("origin").attrib["xyz"], sep=" ")
                size = np.fromstring(guide.find("geometry/box").attrib["size"], sep=" ")
                assert center[2] + size[2] / 2 > 0.006 + 0.009


def test_rack_has_floating_articulation_and_a_grasp_frame(
    laboratory_directory: Path,
) -> None:
    """
    The rack can carry its samples independently of the laboratory root.
    """
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    rack = description.find("link[@name='laboratory_rack']")
    joint = description.find("joint[@name='laboratory_rack_joint']")
    assert joint.attrib["type"] == "floating"
    assert (
        rack.find("visual/geometry/mesh").attrib["filename"] == "assets/tube_rack.glb"
    )
    assert float(rack.find("inertial/mass").attrib["value"]) > 0
    handle = rack.find("collision[@name='rack_handle']")
    frame = description.find("joint[@name='laboratory_rack_handle_joint']")
    np.testing.assert_array_equal(
        np.fromstring(frame.find("origin").attrib["xyz"], sep=" "),
        np.fromstring(handle.find("origin").attrib["xyz"], sep=" "),
    )
    assert frame.find("parent").attrib["link"] == rack.attrib["name"]


def test_closed_drawer_has_headroom_for_loaded_rack(laboratory_directory: Path) -> None:
    """
    The full vial height clears the cabinet above the closed drawer.
    """
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    drawer = description.find("joint[@name='laboratory_drawer_joint']")
    drawer_height = np.fromstring(drawer.find("origin").attrib["xyz"], sep=" ")[2]
    floor = description.find(
        "link[@name='laboratory_drawer']/collision[@name='drawer_bottom']"
    )
    floor_height = np.fromstring(floor.find("origin").attrib["xyz"], sep=" ")[2]
    floor_thickness = np.fromstring(floor.find("geometry/box").attrib["size"], sep=" ")[
        2
    ]
    plate = description.find(
        "link[@name='laboratory_rack']/collision[@name='rack_base']"
    )
    plate_thickness = np.fromstring(plate.find("geometry/box").attrib["size"], sep=" ")[
        2
    ]
    member = next(
        panel
        for panel in CabinetStructure().panels()
        if panel.part == CabinetPart.CROSS_MEMBER
    )
    ceiling = member.origin.to_np()[2, 3] - member.size.z / 2
    tube_height = 0.15
    loaded_height = (
        drawer_height
        + floor_height
        + floor_thickness / 2
        + plate_thickness
        + tube_height
    )
    assert ceiling - loaded_height >= 0.019
