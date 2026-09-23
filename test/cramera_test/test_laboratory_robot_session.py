"""
The physical PR2 runs in an isolated laboratory with measured joint feedback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from cramera import laboratory_robot_session
from cramera.laboratory_robot_session import LaboratoryRobotSession, RobotPhysicsRoute
from cramera.laboratory_physics_session import InvalidPhysicsRequest, PhysicsRoute

from .test_laboratory_physics_session import ControlledContactSimulation, physics_bundle
from .test_server import get_json, post, server


# %% controlled robot and authored bundle
@dataclass
class ControlledRobotSimulation(ControlledContactSimulation):
    """
    Track program ownership separately from measured robot joint positions.
    """

    recording_directory: Path | None = None
    """
    Source of the native motion reference.
    """

    program_state: str = "idle"
    """
    Current program execution state.
    """

    def run_program(self) -> None:
        """
        Start the motor target program and remove a manual hold.
        """
        self.release()
        self.program_state = "running"

    def pause_program(self) -> None:
        """
        Hold the current program stage.
        """
        self.program_state = "paused"

    def reset(self) -> None:
        """
        Reset both physical objects and the reference program.
        """
        super().reset()
        self.program_state = "idle"

    def snapshot(self) -> dict:
        """
        Return measured joints, not the recorded reference values.
        """
        return {
            **super().snapshot(),
            "frames": {"pr2/l_elbow_flex_joint": -0.7},
            "robot": {"state": self.program_state, "progress": 0},
        }


@pytest.fixture()
def robot_bundle(physics_bundle) -> Path:
    """
    Provide the robot visuals independently of the authored laboratory.
    """
    directory = physics_bundle.parent / LaboratoryRobotSession.RECORDING_SCENE
    directory.mkdir()
    model = {
        "name": "pr2",
        "urdf": "pr2.urdf",
        "prefix": "pr2",
        "robot": True,
        "pose": [0.4, -0.72, 0, 0, 0, 0.7071, 0.7071],
    }
    (directory / "scene.json").write_text(
        json.dumps({"models": [model], "robot": {"name": "pr2"}})
    )
    (directory / "trajectory.json").write_text(
        json.dumps({"frames": [{"pr2/l_elbow_flex_joint": -1.5}]})
    )
    (directory / "pr2.urdf").write_bytes(b"robot description")
    (directory / "meshes").mkdir()
    (directory / "meshes" / "finger.stl").write_bytes(b"finger shape")
    return directory


@pytest.fixture()
def robot_session(physics_bundle, robot_bundle, monkeypatch):
    """
    Own and reliably terminate a physical robot worker.
    """
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", ControlledRobotSimulation
    )
    session = LaboratoryRobotSession(physics_bundle.parents[1])
    yield session
    session.stop()


# %% copied appearance and authoritative state
def test_robot_scene_combines_lab_materials_and_actual_joint_state(
    robot_session, physics_bundle, robot_bundle
):
    """
    A physical robot scene reuses the laboratory and starts at measured joints.
    """
    source = (physics_bundle / "scene.json").read_bytes()
    recorded = (robot_bundle / "scene.json").read_bytes()
    result = robot_session.start()
    assert result["state"] == "running"
    assert result["viewerUrl"] == LaboratoryRobotSession.VIEWER_URL
    scene = json.loads((robot_session.output_directory / "scene.json").read_text())
    original = json.loads(source)
    assert scene["physics"]["robot"] is True
    assert scene["models"][0] == original["models"][0]
    assert scene["models"][1]["urdf"] == "robot/pr2.urdf"
    assert scene["objects"] == original["objects"]
    assert scene["rendering"] == original["rendering"]
    assert (
        robot_session.output_directory / "robot/meshes/finger.stl"
    ).read_bytes() == b"finger shape"
    trajectory = json.loads(
        (robot_session.output_directory / "trajectory.json").read_text()
    )
    assert trajectory["frames"] == [result["frames"]]
    assert (physics_bundle / "scene.json").read_bytes() == source
    assert (robot_bundle / "scene.json").read_bytes() == recorded


def test_run_releases_manual_hold_and_blocks_conflicting_drag(robot_session):
    """
    Only one controller can manipulate the tube during robot execution.
    """
    robot_session.start()
    robot_session.set_target("tube_clear", [0.2, 0, 1.1])
    assert robot_session.run_program()["robot"]["state"] == "running"
    assert robot_session.status()["target"] is None
    with pytest.raises(InvalidPhysicsRequest):
        robot_session.set_target("tube_clear", [0.3, 0, 1.1])
    assert robot_session.pause_program()["robot"]["state"] == "paused"
    robot_session.set_target("tube_clear", [0.3, 0, 1.1])
    assert robot_session.reset()["robot"]["state"] == "idle"


# %% isolated HTTP routes
def test_robot_api_keeps_manual_physics_separate(robot_session, server):
    """
    Robot startup and program commands do not start the manual physics session.
    """
    assert post(server + RobotPhysicsRoute.START)[0] == 202
    assert get_json(server + PhysicsRoute.STATE)["state"] == "idle"
    assert post(server + RobotPhysicsRoute.RUN)[1]["robot"]["state"] == "running"
    assert post(server + RobotPhysicsRoute.PAUSE)[1]["robot"]["state"] == "paused"
    assert post(server + RobotPhysicsRoute.RESET)[1]["robot"]["state"] == "idle"
    assert post(server + RobotPhysicsRoute.STOP)[1]["state"] == "idle"


def test_robot_api_rejects_arbitrary_program_parameters(robot_session, server):
    """
    The fixed demo cannot be replaced by browser-supplied executable input.
    """
    code, result = post(server + RobotPhysicsRoute.RUN, {"command": "arbitrary"})
    assert code == 400
    assert result["ok"] is False


def test_robot_api_requires_started_session(robot_session, server):
    """
    Program execution cannot implicitly allocate an uninitialized simulator.
    """
    code, result = post(server + RobotPhysicsRoute.RUN)
    assert code == 409
    assert result["ok"] is False
