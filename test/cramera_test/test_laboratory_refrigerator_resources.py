"""
The sample refrigerator is reusable without the assembled laboratory.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from cramera.laboratory_devices import LaboratoryDevice, LaboratoryDeviceResource
from semantic_digital_twin.adapters.urdf import URDFParser


# %% standalone refrigerator
def test_refrigerator_copy_retains_its_articulated_door(tmp_path: Path) -> None:
    """
    A copied refrigerator loads with the same door joint as the full scene.
    """
    resource = LaboratoryDeviceResource(LaboratoryDevice.REFRIGERATOR)
    folder = resource.copy_to(tmp_path)
    model = LaboratoryDeviceResource(LaboratoryDevice.REFRIGERATOR, tmp_path)
    world = URDFParser.from_file(str(model.urdf), prefix="").parse()
    scene_path = resource.directory.parents[1] / "aicor_medical_lab/scene.json"
    instrument = json.loads(scene_path.read_text())["laboratory"]["refrigerator"]
    description = ElementTree.parse(model.urdf)
    joint = description.find(f"joint[@name='{instrument['doorJoint']}']")
    assert joint.attrib["type"] == "revolute"
    assert float(joint.find("limit").attrib["upper"]) == instrument["doorOpen"]
    assert {body.name.name for body in world.bodies} == {
        instrument["body"],
        instrument["doorBody"],
    }
    for mesh in description.findall(".//mesh"):
        reference = Path(mesh.attrib["filename"])
        assert not reference.is_absolute()
        assert ".." not in reference.parts
        assert (folder / reference).is_file()
