"""
Reference interpolation and actual-contact completion of a PR2 program.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cramera.laboratory_physics_program import (
    JointReference,
    ProgramState,
    LaboratoryProgramPhysics,
)
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_bundle import laboratory_directory


# %% native recording reference
@pytest.fixture
def recording_directory(tmp_path: Path) -> Path:
    """
    Write a small recorded joint reference with an independent object track.
    """
    frames = [
        {"pr2/l_gripper_l_finger_joint": angle, "pr2/l_elbow_flex_joint": index / 10}
        for index, angle in enumerate((0.4, 0.2, 0.12, 0.12, 0.12, 0.4))
    ]
    (tmp_path / "trajectory.json").write_text(
        json.dumps(
            {
                "framesPerSecond": 10,
                "frames": frames,
                "objects": [
                    {"tube_clear": [0.16, -0.065, 0.906, 0, 0, 0, 1]},
                    {"tube_clear": [0.28, -0.065, 0.906, 0, 0, 0, 1]},
                ],
            }
        )
    )
    (tmp_path / "scene.json").write_text(
        json.dumps({"segments": [{"picks": "tube_clear", "attach": 3, "detach": 4}]})
    )
    return tmp_path


class TestJointReference:
    """
    Reference sampling affects joints while object poses remain observations.
    """

    def test_joint_targets_interpolate_between_recorded_frames(
        self, recording_directory: Path
    ) -> None:
        reference = JointReference.load(recording_directory)
        targets = reference.sample(0.15)
        assert targets["pr2/l_elbow_flex_joint"] == pytest.approx(0.15)
        assert targets["pr2/l_gripper_l_finger_joint"] == pytest.approx(0.16)
        assert "tube_clear" not in targets

    def test_pre_lift_gate_uses_start_of_closed_gripper_plateau(
        self, recording_directory: Path
    ) -> None:
        reference = JointReference.load(recording_directory)
        assert reference.grasp_time == 0.2
        assert reference.release_time == 0.4
        assert reference.duration == 0.5
        np.testing.assert_array_equal(reference.destination, [0.28, -0.065, 0.906])

    def test_recording_samples_clamp_at_both_ends(
        self, recording_directory: Path
    ) -> None:
        reference = JointReference.load(recording_directory)
        assert reference.sample(-1) == reference.frames[0]
        assert reference.sample(100) == reference.frames[-1]

    def test_nonfinite_recorded_joint_cannot_reach_actuator(
        self, recording_directory: Path
    ) -> None:
        path = recording_directory / "trajectory.json"
        recording = json.loads(path.read_text())
        recording["frames"][2]["pr2/l_elbow_flex_joint"] = float("nan")
        path.write_text(json.dumps(recording))
        with pytest.raises(ValueError):
            JointReference.load(recording_directory)


# %% real physical transfer
class TestPhysicalTransfer:
    """
    A recorded plan succeeds only when the physical tube is actually placed.
    """

    def test_contact_grasp_lifts_and_releases_into_the_destination(
        self, robot_physics: LaboratoryRobotPhysics
    ) -> None:
        world = LaboratoryProgramPhysics(robot_physics, robot_physics.robot_directory)
        world.run_program()
        for _ in range(100):
            world.step(500)
            if world.program.state != ProgramState.RUNNING:
                break
        result = world.snapshot()["robot"]
        assert result["state"] == ProgramState.SUCCEEDED, result
        assert result["graspVerified"]
        assert result["maximumLift"] >= world.program.parameters.minimum_lift
        assert result["released"]
        assert (
            result["destinationError"] <= world.program.parameters.destination_tolerance
        )
        assert robot_physics.target is None
        assert not np.any(robot_physics.data.xfrc_applied)

    def test_fresh_run_restores_initial_scene_after_idle_manual_movement(
        self, robot_physics: LaboratoryRobotPhysics
    ) -> None:
        world = LaboratoryProgramPhysics(robot_physics, robot_physics.robot_directory)
        initial = robot_physics.data.qpos.copy()
        initial_pose = robot_physics.snapshot()["objects"][
            world.program.reference.object_key
        ]
        world.set_target(
            world.program.reference.object_key,
            [initial_pose[0], initial_pose[1], initial_pose[2] + 0.18],
        )
        world.step(1000)
        moved_pose = robot_physics.snapshot()["objects"][
            world.program.reference.object_key
        ]
        assert moved_pose[2] > initial_pose[2] + 0.1
        assert world.program.state == ProgramState.IDLE
        world.run_program()
        assert world.program.state == ProgramState.RUNNING
        assert robot_physics.target is None
        assert world.program.reference_time == 0
        np.testing.assert_array_equal(robot_physics.data.qpos, initial)

    def test_pause_and_release_stop_reference_without_freezing_physics(
        self, robot_physics: LaboratoryRobotPhysics
    ) -> None:
        world = LaboratoryProgramPhysics(robot_physics, robot_physics.robot_directory)
        world.run_program()
        world.step(50)
        world.pause_program()
        reference_time = world.program.reference_time
        physics_time = robot_physics.data.time
        world.step(50)
        assert world.program.reference_time == reference_time
        assert robot_physics.data.time > physics_time
        world.run_program()
        world.step(10)
        assert world.program.reference_time > reference_time
        world.release()
        assert world.program.state == ProgramState.PAUSED
        assert robot_physics.target is None

    def test_reset_clears_program_and_restores_physical_state(
        self, robot_physics: LaboratoryRobotPhysics
    ) -> None:
        world = LaboratoryProgramPhysics(robot_physics, robot_physics.robot_directory)
        initial = robot_physics.data.qpos.copy()
        world.run_program()
        world.step(200)
        world.reset()
        assert world.program.state == ProgramState.IDLE
        assert world.program.reference_time == 0
        assert not world.program.grasp_verified
        np.testing.assert_array_equal(robot_physics.data.qpos, initial)

    def test_recorded_intent_cannot_confirm_a_grasp_without_contacts(
        self, robot_physics: LaboratoryRobotPhysics
    ) -> None:
        world = LaboratoryProgramPhysics(robot_physics, robot_physics.robot_directory)
        initial_frame = world.program.reference.frames[0]
        world.program.reference = replace(
            world.program.reference,
            frames=(initial_frame,) * 20,
            frame_rate=10,
            grasp_time=0.1,
            release_time=0.5,
        )
        world.run_program()
        world.step(3500)
        assert world.program.state == ProgramState.FAILED
        assert not world.program.grasp_verified
        assert world.program.reference_time == world.program.reference.grasp_time
