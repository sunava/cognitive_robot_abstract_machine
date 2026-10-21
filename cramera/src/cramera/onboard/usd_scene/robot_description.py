"""
The URDF of a converted scene: one link per imported link with its GLB as visual, and
one joint per non-root link.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy

from cramera.mesh_format import MeshFormat
from cramera.onboard.usd_scene.bundle_layout import BundleFile
from cramera.onboard.usd_scene.scene_description import (
    ImportedJoint,
    ImportedLink,
    JointKind,
    Pose,
)

# %% URDF vocabulary


class UrdfTag(StrEnum):
    """
    The URDF elements the writer emits.
    """

    ROBOT = "robot"
    """
    The document root.
    """

    LINK = "link"
    """
    A rigid body.
    """

    VISUAL = "visual"
    """
    A link's visual geometry.
    """

    GEOMETRY = "geometry"
    """
    The shape of a visual.
    """

    MESH = "mesh"
    """
    A mesh file shape.
    """

    JOINT = "joint"
    """
    A joint between two links.
    """

    PARENT = "parent"
    """
    A joint's parent link.
    """

    CHILD = "child"
    """
    A joint's child link.
    """

    ORIGIN = "origin"
    """
    A joint's frame in its parent link.
    """

    AXIS = "axis"
    """
    A joint's motion axis.
    """

    LIMIT = "limit"
    """
    A joint's limits.
    """


class UrdfAttribute(StrEnum):
    """
    The URDF attributes the writer emits.
    """

    NAME = "name"
    """
    The name of a link, joint or robot.
    """

    TYPE = "type"
    """
    A joint's kind.
    """

    FILENAME = "filename"
    """
    A mesh file, relative to the URDF.
    """

    LINK = "link"
    """
    The link a parent or child element names.
    """

    TRANSLATION = "xyz"
    """
    A translation or axis.
    """

    ROTATION = "rpy"
    """
    Fixed-axis roll, pitch and yaw.
    """

    LOWER = "lower"
    """
    The lowest joint value.
    """

    UPPER = "upper"
    """
    The highest joint value.
    """

    EFFORT = "effort"
    """
    The largest joint force or torque.
    """

    VELOCITY = "velocity"
    """
    The largest joint speed.
    """


# %% writer


@dataclass
class RobotDescription:
    """
    A URDF document for links and joints of a converted scene.
    """

    name: str
    """
    The robot name in the URDF.
    """

    links: list[ImportedLink]
    """
    The links, parents before children.
    """

    joints: list[ImportedJoint]
    """
    One joint per non-root link.
    """

    mesh_directory: str = BundleFile.MESHES.value
    """
    The directory, relative to the URDF, holding one GLB per link.
    """

    effort_limit: float = 1000.0
    """
    The effort written for every movable joint; USD stages author none.
    """

    velocity_limit: float = 1.0
    """
    The velocity written for every movable joint; USD stages author none.
    """

    unbounded_travel: float = 1000.0
    """
    The travel in metres written for a sliding joint without authored limits, which
    URDF cannot express.
    """

    def mesh_filename(self, link: ImportedLink) -> str:
        """
        The GLB file of a link, relative to the URDF.

        :param link: The link.
        """
        return f"{self.mesh_directory}/{link.name}{MeshFormat.GLB.value}"

    def to_element(self) -> ElementTree.Element:
        """
        The URDF document as an XML element.
        """
        robot = ElementTree.Element(UrdfTag.ROBOT, {UrdfAttribute.NAME: self.name})
        for link in self.links:
            element = ElementTree.SubElement(
                robot, UrdfTag.LINK, {UrdfAttribute.NAME: link.name}
            )
            if not link.meshes:
                continue
            geometry = ElementTree.SubElement(
                ElementTree.SubElement(element, UrdfTag.VISUAL), UrdfTag.GEOMETRY
            )
            ElementTree.SubElement(
                geometry,
                UrdfTag.MESH,
                {UrdfAttribute.FILENAME: self.mesh_filename(link)},
            )
        for joint in self.joints:
            self.add_joint(robot, joint)
        ElementTree.indent(robot)
        return robot

    def add_joint(self, robot: ElementTree.Element, joint: ImportedJoint) -> None:
        """
        Add one joint element.

        :param robot: The document root.
        :param joint: The joint.
        """
        element = ElementTree.SubElement(
            robot,
            UrdfTag.JOINT,
            {UrdfAttribute.NAME: joint.name, UrdfAttribute.TYPE: joint.kind.value},
        )
        ElementTree.SubElement(
            element, UrdfTag.PARENT, {UrdfAttribute.LINK: joint.parent}
        )
        ElementTree.SubElement(
            element, UrdfTag.CHILD, {UrdfAttribute.LINK: joint.child}
        )
        ElementTree.SubElement(
            element, UrdfTag.ORIGIN, self.origin_attributes(joint.origin)
        )
        if not joint.kind.movable:
            return
        ElementTree.SubElement(
            element, UrdfTag.AXIS, {UrdfAttribute.TRANSLATION: self.numbers(joint.axis)}
        )
        limit = {
            UrdfAttribute.EFFORT: self.numbers([self.effort_limit]),
            UrdfAttribute.VELOCITY: self.numbers([self.velocity_limit]),
        }
        if joint.kind is not JointKind.CONTINUOUS:
            limit[UrdfAttribute.LOWER] = self.numbers(
                [-self.unbounded_travel if joint.lower is None else joint.lower]
            )
            limit[UrdfAttribute.UPPER] = self.numbers(
                [self.unbounded_travel if joint.upper is None else joint.upper]
            )
        ElementTree.SubElement(element, UrdfTag.LIMIT, limit)

    def origin_attributes(self, pose: Pose) -> dict[str, str]:
        """
        The ``xyz`` and ``rpy`` attributes of a pose.

        :param pose: The pose.
        """
        return {
            UrdfAttribute.TRANSLATION: self.numbers(pose.translation),
            UrdfAttribute.ROTATION: self.numbers(pose.roll_pitch_yaw),
        }

    @staticmethod
    def numbers(values: numpy.ndarray | list[float]) -> str:
        """
        Numbers separated by spaces, without needless digits and without negative
        zeros.

        :param values: The numbers.
        """
        return " ".join(f"{float(value) + 0.0:.9g}" for value in values)

    def write(self, path: Path) -> None:
        """
        Write the URDF file.

        :param path: The destination file.
        """
        ElementTree.ElementTree(self.to_element()).write(
            path, encoding="utf-8", xml_declaration=True
        )
