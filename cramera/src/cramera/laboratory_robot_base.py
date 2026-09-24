"""
Expose physical planar base motion through ordinary renderer joints.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cramera.laboratory_physics_robot import RobotBaseJoint, RobotPhysicsParameters


# %% recording articulation
@dataclass(frozen=True)
class RobotBaseDescription:
    """
    Extend a copied robot description with world-aligned planar coordinates.
    """

    path: Path
    """
    Copied URDF whose original root becomes the planar chain's child.
    """

    rotation: tuple[float, float, float, float]
    """
    Original model orientation in XYZW quaternion order.
    """

    def write(self) -> None:
        """
        Add the three physical base coordinates once, keeping meshes unchanged.
        """
        tree = ElementTree.parse(self.path)
        root = tree.getroot()
        if root.find(f"joint[@name='{RobotBaseJoint.HORIZONTAL}']") is not None:
            return
        children = {
            joint.find("child").attrib["link"] for joint in root.findall("joint")
        }
        [original] = [
            link.attrib["name"]
            for link in root.findall("link")
            if link.attrib["name"] not in children
        ]
        rotation = Rotation.from_quat(self.rotation)
        parameters = RobotPhysicsParameters()
        previous = "pr2/mobile_origin"
        ElementTree.SubElement(root, "link", name=previous)
        for index, name in enumerate(RobotBaseJoint):
            child = (
                original
                if name == RobotBaseJoint.HEADING
                else name.replace("_joint", "_link")
            )
            if child != original:
                ElementTree.SubElement(root, "link", name=child)
            heading = name == RobotBaseJoint.HEADING
            joint = ElementTree.SubElement(
                root, "joint", name=name, type="revolute" if heading else "prismatic"
            )
            ElementTree.SubElement(joint, "parent", link=previous)
            ElementTree.SubElement(joint, "child", link=child)
            ElementTree.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            ElementTree.SubElement(
                joint,
                "axis",
                xyz=" ".join(map(str, rotation.inv().apply(np.eye(3)[index]))),
            )
            limit = math.pi if heading else 2.0
            ElementTree.SubElement(
                joint,
                "limit",
                lower=str(-limit),
                upper=str(limit),
                effort=str(
                    parameters.base_torque if heading else parameters.base_force
                ),
                velocity="0.2",
            )
            previous = child
        ElementTree.indent(tree)
        tree.write(self.path, encoding="utf-8", xml_declaration=True)
