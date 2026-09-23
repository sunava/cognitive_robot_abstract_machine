"""
Actuated robot contacts share the laboratory's authoritative dynamic state.
"""

from pathlib import Path

import numpy as np
import pytest

from cramera import paths
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics

from .test_laboratory_bundle import laboratory_directory


# %% local robot model
@pytest.fixture
def robot_physics(laboratory_directory: Path) -> LaboratoryRobotPhysics:
    """
    Load the available PR2 collision assets and recorded initial posture.
    """
    recording = paths.resolve_scene_directory("precision_lab_pr2")
    source = (
        paths.repository_root()
        / "cramera/scenes/pr2_breakfast_c/pr2_with_ft2_cableguide.urdf"
    )
    if recording is None or not source.is_file():
        pytest.skip("The optional local PR2 collision and recording bundles are absent")
    return LaboratoryRobotPhysics(
        bundle_directory=laboratory_directory,
        robot_directory=recording,
        robot_description=source,
    )


class TestForceLimitedRobot:
    """
    Actuators advance robot joints without assigning dynamic poses.
    """

    def test_joint_target_does_not_teleport(
        self, robot_physics: LaboratoryRobotPhysics
    ) -> None:
        before = robot_physics.data.qpos.copy()
        key = "pr2/l_gripper_l_finger_joint"
        robot_physics.set_joint_targets({key: 0.3})
        np.testing.assert_array_equal(robot_physics.data.qpos, before)
        robot_physics.step(600)
        assert robot_physics.robot_state()[key] > 0.2

    def test_initial_posture_is_stable(
        self, robot_physics: LaboratoryRobotPhysics
    ) -> None:
        before = robot_physics.robot_state()
        robot_physics.step(1500)
        after = robot_physics.robot_state()
        for key in robot_physics.joint_targets:
            assert abs(after[key] - before[key]) < 0.002
        assert np.isfinite(robot_physics.data.qpos).all()

    def test_mimic_fingers_follow_physical_coupling(
        self, robot_physics: LaboratoryRobotPhysics
    ) -> None:
        key = "pr2/l_gripper_l_finger_joint"
        robot_physics.set_joint_targets({key: 0.3})
        robot_physics.step(1000)
        state = robot_physics.robot_state()
        for sibling in (
            "pr2/l_gripper_r_finger_joint",
            "pr2/l_gripper_l_finger_tip_joint",
            "pr2/l_gripper_r_finger_tip_joint",
        ):
            assert state[sibling] == pytest.approx(state[key], abs=0.0001)

    def test_command_and_contact_force_are_bounded(
        self, robot_physics: LaboratoryRobotPhysics
    ) -> None:
        key = "pr2/l_gripper_l_finger_joint"
        robot_physics.set_joint_targets({key: 0.3})
        robot_physics.step(10)
        index = robot_physics.model.actuator(key).id
        assert robot_physics.model.actuator_forcelimited[index]
        assert abs(robot_physics.data.actuator_force[index]) <= (
            robot_physics.robot_parameters.finger_torque
        )

    def test_reset_restores_robot_and_free_objects(
        self, robot_physics: LaboratoryRobotPhysics
    ) -> None:
        before = robot_physics.data.qpos.copy()
        robot_physics.set_joint_targets({"pr2/l_gripper_l_finger_joint": 0.3})
        robot_physics.step(100)
        robot_physics.reset()
        np.testing.assert_array_equal(robot_physics.data.qpos, before)
        assert robot_physics.snapshot()["frames"] == robot_physics.robot_state()

    def test_invalid_command_does_not_partially_update_targets(
        self, robot_physics: LaboratoryRobotPhysics
    ) -> None:
        before = robot_physics.joint_targets.copy()
        with pytest.raises(ValueError):
            robot_physics.set_joint_targets(
                {"pr2/l_gripper_l_finger_joint": 0.3, "absent": 0.2}
            )
        assert robot_physics.joint_targets == before
