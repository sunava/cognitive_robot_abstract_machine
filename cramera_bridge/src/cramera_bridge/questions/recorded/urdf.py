"""
The links and joints of a robot model, read from its URDF file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import Dict, List, Optional

# %% vocabulary


class UrdfTag(StrEnum):
    """
    The URDF elements and attributes read.
    """

    LINK = "link"
    """
    A link element.
    """

    JOINT = "joint"
    """
    A joint element.
    """

    PARENT = "parent"
    """
    A joint's parent element.
    """

    CHILD = "child"
    """
    A joint's child element.
    """

    NAME = "name"
    """
    The name attribute.
    """

    TYPE = "type"
    """
    The type attribute.
    """

    LINK_REFERENCE = "link"
    """
    The attribute naming the link of a parent or child element.
    """


class UrdfJointType(StrEnum):
    """
    The joint types of URDF.
    """

    REVOLUTE = "revolute"
    """
    Turns within limits.
    """

    CONTINUOUS = "continuous"
    """
    Turns without limits.
    """

    PRISMATIC = "prismatic"
    """
    Slides within limits.
    """

    FIXED = "fixed"
    """
    Does not move.
    """

    FLOATING = "floating"
    """
    Moves freely.
    """

    PLANAR = "planar"
    """
    Moves in a plane.
    """


# %% structure


@dataclass(frozen=True)
class UrdfJoint:
    """
    One joint of a robot model.
    """

    name: str
    """
    The joint's name.
    """

    joint_type: UrdfJointType
    """
    How the joint moves.
    """

    parent: str
    """
    The link the joint hangs from.
    """

    child: str
    """
    The link the joint carries.
    """

    def is_movable(self) -> bool:
        """
        :return: Whether the joint moves at all.
        """
        return self.joint_type is not UrdfJointType.FIXED


@dataclass(frozen=True)
class UrdfStructure:
    """
    The kinematic tree of a robot model.
    """

    links: List[str] = field(default_factory=list)
    """
    The link names in file order.
    """

    joints: List[UrdfJoint] = field(default_factory=list)
    """
    The joints in file order.
    """

    @classmethod
    def of_file(cls, path: Path) -> UrdfStructure:
        """
        :param path: A URDF file of a bundle.
        :return: Its links and joints: the direct children of ``<robot>``; elements
            of the same name inside other blocks, such as ``<ros2_control>``, describe
            controllers, not the robot's structure.
        """
        root = ElementTree.parse(path).getroot()
        return cls(
            links=[link.attrib[UrdfTag.NAME] for link in root.findall(UrdfTag.LINK)],
            joints=[
                UrdfJoint(
                    name=joint.attrib[UrdfTag.NAME],
                    joint_type=UrdfJointType(joint.attrib[UrdfTag.TYPE]),
                    parent=joint.find(UrdfTag.PARENT).attrib[UrdfTag.LINK_REFERENCE],
                    child=joint.find(UrdfTag.CHILD).attrib[UrdfTag.LINK_REFERENCE],
                )
                for joint in root.findall(UrdfTag.JOINT)
            ],
        )

    def joint_carrying(self, link: str) -> Optional[UrdfJoint]:
        """
        :param link: A link name.
        :return: The joint whose child the link is, or None for the root link.
        """
        return self.joints_by_child().get(link)

    def joints_by_child(self) -> Dict[str, UrdfJoint]:
        """
        :return: Each joint by the link it carries.
        """
        return {joint.child: joint for joint in self.joints}
