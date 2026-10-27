"""
Native poses, scales and colours as the protocol writes them.
"""

from __future__ import annotations

import numpy

from cramera_bridge.geometry import (
    COLOR_CHANNEL_MAXIMUM,
    NumericPose,
    PublishedColor,
    PublishedScale,
)
from cramera_protocol.geometry import Pose, Vector3
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color, Scale

# %% poses


def test_an_entity_pose_is_read_relative_to_the_root(kitchen):
    pose = NumericPose.of_entity(kitchen.world, kitchen.milk)
    assert pose.published() == Pose(1.0, 0.1, 0.9, 0.0, 0.0, 0.0, 1.0)


def test_a_published_pose_is_a_copy(kitchen):
    pose = NumericPose.of_entity(kitchen.world, kitchen.milk)
    kitchen.milk_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        reference_frame=kitchen.root
    )
    assert pose.published_position() == Vector3(1.0, 0.1, 0.9)


def test_a_rotation_becomes_a_quaternion():
    quarter_turn = numpy.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0, 0, 0, 1],
        ]
    )
    orientation = NumericPose.of_matrix(quarter_turn).published_orientation(3)
    assert orientation.qz == orientation.qw == round(numpy.sqrt(0.5), 3)
    assert "q(0.00, 0.00, 0.71, 0.71)" in NumericPose.of_matrix(quarter_turn).label


# %% scales and colours


def test_a_scale_keeps_its_axes():
    assert PublishedScale(Scale(0.1, 0.2, 0.3)).published() == Vector3(0.1, 0.2, 0.3)


def test_a_colour_becomes_css_and_keeps_its_opacity():
    color = PublishedColor(Color(1.0, 0.0, 0.5, 0.25))
    blue = round(0.5 * COLOR_CHANNEL_MAXIMUM)
    assert color.hexadecimal() == "#ff00%02x" % blue
    assert color.opacity() == 0.25


def test_white_stays_white():
    assert PublishedColor(Color()).hexadecimal() == "#ffffff"
