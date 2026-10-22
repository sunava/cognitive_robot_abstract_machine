"""
The neutral scene description: joint values recovered from body poses, and the names
handed out to links and joints.
"""

from __future__ import annotations

import numpy
import pytest
from scipy.spatial.transform import Rotation

from cramera.onboard.usd_scene.scene_description import (
    ImportedJoint,
    JointKind,
    Pose,
    PoseTrack,
    UniqueNames,
)

# %% fixtures


def hinge(lower: float | None, upper: float | None) -> ImportedJoint:
    """
    A hinge about Z between two bodies whose joint frames are their own frames.

    :param lower: The lowest angle in radians, or None.
    :param upper: The highest angle in radians, or None.
    """
    return ImportedJoint(
        name="Hinge",
        path="/World/Hinge",
        kind=JointKind.REVOLUTE if lower is not None else JointKind.CONTINUOUS,
        parent="Frame",
        child="Leaf",
        origin=Pose(),
        axis=numpy.array([0.0, 0.0, 1.0]),
        lower=lower,
        upper=upper,
    )


def turned(angle: float) -> Pose:
    """
    A body turned about Z.

    :param angle: The angle in radians.
    """
    return Pose.from_rotation(Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix())


# %% joint values


def test_angle_past_half_a_turn_is_taken_within_the_limits():
    joint = hinge(0.0, 4.0)
    assert joint.value_between(Pose(), turned(3.5)) == pytest.approx(3.5)


def test_angle_of_a_joint_without_limits_lies_within_half_a_turn():
    assert hinge(None, None).value_between(Pose(), turned(3.5)) == pytest.approx(
        3.5 - 2.0 * numpy.pi
    )


def test_one_value_and_a_track_of_values_agree():
    joint = hinge(0.0, 4.0)
    angles = numpy.array([0.5, 2.0, 3.5])
    relative = PoseTrack.of_parts(
        Rotation.from_rotvec(numpy.outer(angles, joint.axis)).as_matrix(),
        numpy.zeros((len(angles), 3)),
    )
    assert joint.values_of(relative) == pytest.approx(
        [joint.value_between(Pose(), turned(angle)) for angle in angles]
    )


def test_motion_at_a_value_turns_the_child_about_the_axis():
    joint = hinge(0.0, 4.0)
    assert joint.motion(3.5).is_close(turned(3.5))


# %% poses


def test_rigid_part_removes_scale_and_mirroring():
    matrix = turned(0.3).matrix @ numpy.diag([2.0, 2.0, -2.0, 1.0])
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    rigid = Pose.rigid_part(matrix)
    assert numpy.linalg.det(rigid.rotation) == pytest.approx(1.0)
    assert rigid.rotation @ rigid.rotation.T == pytest.approx(numpy.eye(3))
    assert rigid.translation == pytest.approx([1.0, 2.0, 3.0])


def test_inverse_undoes_a_pose():
    pose = turned(0.7) @ Pose.from_translation(numpy.array([1.0, -2.0, 0.5]))
    assert (pose @ pose.inverse()).is_close(Pose())


# %% names


def test_repeated_names_are_numbered_from_two():
    names = UniqueNames()
    assert [names.claim(name) for name in ["Link", "Link", "Hand", "Link"]] == [
        "Link",
        "Link_2",
        "Hand",
        "Link_3",
    ]
