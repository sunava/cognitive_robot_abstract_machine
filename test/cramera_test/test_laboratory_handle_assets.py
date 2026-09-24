"""
The exported instrument handle matches its graspable collision geometry.
"""

from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest
import trimesh

from .test_laboratory_bundle import laboratory_directory


# %% authored visual and collision agreement
def test_lid_visual_reaches_the_authored_handle_surface(
    laboratory_directory: Path,
) -> None:
    """
    The exported lid includes the elevated bar that the fingertips grasp.
    """
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    handle = description.find(
        "link[@name='laboratory_centrifuge_lid']/collision[@name='lid_handle']"
    )
    center = np.fromstring(handle.find("origin").attrib["xyz"], sep=" ")
    size = np.fromstring(handle.find("geometry/box").attrib["size"], sep=" ")
    resource = (
        Path(__file__).resolve().parents[2]
        / "cramera/src/cramera/resources/precision_lab/assets/centrifuge_lid.glb"
    )
    lid = trimesh.load_scene(resource).to_mesh()
    assert lid.bounds[1, 2] == pytest.approx(center[2] + size[2] / 2, abs=0.0001)
