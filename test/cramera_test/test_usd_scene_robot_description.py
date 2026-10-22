"""
The URDF written for the links and joints of a converted scene.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree

import numpy
import pytest

from cramera.onboard.usd_scene.robot_description import (
    RobotDescription,
    UrdfAttribute,
    UrdfTag,
)
from cramera.onboard.usd_scene.scene_description import (
    ImportedJoint,
    ImportedLink,
    ImportedMesh,
    JointKind,
    Pose,
)

# %% fixtures


def link(name: str, with_mesh: bool) -> ImportedLink:
    """
    A link, with or without one triangle.

    :param name: The link name.
    :param with_mesh: Whether the link has visual geometry.
    """
    meshes = (
        [
            ImportedMesh(
                f"/World/{name}/Mesh",
                numpy.eye(3, dtype=numpy.float32),
                numpy.array([[0, 1, 2]], dtype=numpy.uint32),
                "paint",
            )
        ]
        if with_mesh
        else []
    )
    return ImportedLink(name, f"/World/{name}", Pose(), meshes=meshes)


@pytest.fixture()
def description() -> RobotDescription:
    """
    A root, a body with a hinge, a wheel without limits and a slide.
    """
    origin = Pose.from_translation_quaternion(
        numpy.array([1.0, 2.0, 3.0]),
        numpy.array([0.0, 0.0, numpy.sin(numpy.pi / 4), numpy.cos(numpy.pi / 4)]),
    )
    return RobotDescription(
        "lab",
        [
            link("World", False),
            link("Door", True),
            link("Wheel", True),
            link("Drawer", True),
        ],
        [
            ImportedJoint(
                "Hinge",
                "/World/Joints/Hinge",
                JointKind.REVOLUTE,
                "World",
                "Door",
                origin,
                numpy.array([0.0, 0.0, 1.0]),
                0.0,
                1.5,
            ),
            ImportedJoint(
                "Spin",
                "/World/Joints/Spin",
                JointKind.CONTINUOUS,
                "World",
                "Wheel",
                Pose(),
                numpy.array([0.0, 1.0, 0.0]),
            ),
            ImportedJoint(
                "Slide",
                "/World/Joints/Slide",
                JointKind.PRISMATIC,
                "World",
                "Drawer",
                Pose(),
                numpy.array([1.0, 0.0, 0.0]),
                None,
                0.4,
            ),
        ],
    )


def joint_element(description: RobotDescription, name: str) -> ElementTree.Element:
    """
    The joint element with a name.

    :param description: The URDF writer.
    :param name: The joint name.
    """
    return next(
        element
        for element in description.to_element().iter(UrdfTag.JOINT)
        if element.get(UrdfAttribute.NAME) == name
    )


# %% links


def test_links_with_geometry_reference_their_glb(description):
    robot = description.to_element()
    meshes = {
        element.get(UrdfAttribute.NAME): [
            mesh.get(UrdfAttribute.FILENAME) for mesh in element.iter(UrdfTag.MESH)
        ]
        for element in robot.iter(UrdfTag.LINK)
    }
    assert meshes == {
        "World": [],
        "Door": [description.mesh_filename(description.links[1])],
        "Wheel": [description.mesh_filename(description.links[2])],
        "Drawer": [description.mesh_filename(description.links[3])],
    }


# %% joints


def test_revolute_joint_writes_origin_axis_and_limits(description):
    element = joint_element(description, "Hinge")
    assert element.get(UrdfAttribute.TYPE) == JointKind.REVOLUTE
    origin = element.find(UrdfTag.ORIGIN)
    assert [
        float(value) for value in origin.get(UrdfAttribute.TRANSLATION).split()
    ] == [1.0, 2.0, 3.0]
    assert [
        float(value) for value in origin.get(UrdfAttribute.ROTATION).split()
    ] == pytest.approx([0.0, 0.0, numpy.pi / 2])
    assert element.find(UrdfTag.AXIS).get(UrdfAttribute.TRANSLATION) == "0 0 1"
    limit = element.find(UrdfTag.LIMIT)
    assert (
        float(limit.get(UrdfAttribute.LOWER)),
        float(limit.get(UrdfAttribute.UPPER)),
    ) == (0.0, 1.5)


def test_continuous_joint_has_no_position_limits(description):
    limit = joint_element(description, "Spin").find(UrdfTag.LIMIT)
    assert limit.get(UrdfAttribute.LOWER) is None
    assert limit.get(UrdfAttribute.UPPER) is None


def test_missing_slide_limit_becomes_the_unbounded_travel(description):
    limit = joint_element(description, "Slide").find(UrdfTag.LIMIT)
    assert float(limit.get(UrdfAttribute.LOWER)) == -description.unbounded_travel
    assert float(limit.get(UrdfAttribute.UPPER)) == 0.4
