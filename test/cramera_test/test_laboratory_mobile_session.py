"""
The generated viewer receives the mobile robot's measured base joints.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ElementTree

from cramera.laboratory_physics_program import LaboratoryProgramPhysics
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics, RobotBaseJoint
from cramera.laboratory_robot_session import LaboratoryRobotSession

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_mobile_robot import mobile_robot


# %% measured mobile scene
def test_mobile_session_augments_only_copied_robot_description(
    mobile_robot: LaboratoryRobotPhysics, tmp_path: Path
) -> None:
    """
    The viewer model moves with physical base frames while source assets stay intact.
    """
    physics = mobile_robot
    session = LaboratoryRobotSession(tmp_path)
    session.recording_directory = physics.robot_directory
    session.physics = LaboratoryProgramPhysics(physics, physics.robot_directory)
    source = physics.robot_directory / "pr2.urdf"
    original = source.read_bytes()
    session._write_scene(physics.bundle_directory)
    scene = json.loads((session.output_directory / "scene.json").read_text())
    [robot] = [model for model in scene["models"] if model.get("robot")]
    assert set(RobotBaseJoint) <= set(robot["movableJoints"])
    copied = ElementTree.parse(session.output_directory / robot["urdf"])
    for joint in RobotBaseJoint:
        assert copied.find(f"joint[@name='{joint}']") is not None
    trajectory = json.loads((session.output_directory / "trajectory.json").read_text())
    for joint in RobotBaseJoint:
        assert trajectory["frames"][0][joint] == physics.robot_state()[joint]
    assert source.read_bytes() == original
