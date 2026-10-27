"""
Reading the links and joints of a bundle's URDF.
"""

from __future__ import annotations

from pathlib import Path

from cramera_bridge.questions.recorded.urdf import UrdfJointType, UrdfStructure

DATASET = Path(__file__).parent / "dataset" / "urdf"
"""
The URDF files of these tests.
"""


def test_joints_inside_control_blocks_are_not_robot_joints():
    structure = UrdfStructure.of_file(DATASET / "arm_with_ros2_control.urdf")
    assert [joint.name for joint in structure.joints] == ["finger_joint"]
    assert structure.joints[0].joint_type is UrdfJointType.REVOLUTE
    assert structure.links == ["base", "finger"]
