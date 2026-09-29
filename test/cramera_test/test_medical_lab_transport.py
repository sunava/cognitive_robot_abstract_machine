"""
Expanded instrument requests retain authentication and exclusive plan ownership.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from cramera import laboratory_robot_session
from cramera.laboratory_robot_session import RobotPhysicsRoute
from cramera.medical_lab_command import MedicalLabAction, MedicalLabCommand

from .test_laboratory_physics_session import physics_bundle
from .test_laboratory_robot_session import ControlledRobotSimulation, robot_bundle
from .test_server import post, server


# %% controlled mechanism feedback
@dataclass
class MechanismOwnershipSimulation(ControlledRobotSimulation):
    """
    Publish mechanism ownership while retaining ordinary robot simulation calls.
    """

    device_command: MedicalLabCommand | None = None
    """
    Most recent validated command forwarded by the session.
    """

    def run_devices(self, command: MedicalLabCommand) -> None:
        """
        Retain the typed request separately from the robot program.
        """
        self.device_command = command

    def snapshot(self) -> dict:
        """
        Expose a measured hinge and independent mechanism lifecycle.
        """
        state = "running" if self.device_command is not None else "idle"
        return {
            **super().snapshot(),
            "instruments": {
                "joints": {"fixture__hinge": 0.2},
                "demo": {"state": state},
            },
        }


@pytest.fixture()
def mechanism_server(physics_bundle, robot_bundle, monkeypatch, server):
    """
    Start the real HTTP session with deterministic measured mechanisms.
    """
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", MechanismOwnershipSimulation
    )
    assert post(server + RobotPhysicsRoute.START)[0] == 202
    return server


def test_device_route_forwards_valid_selection(mechanism_server) -> None:
    """
    The public contract selects an authored demo rather than arbitrary code.
    """
    code, result = post(
        mechanism_server + RobotPhysicsRoute.DEVICES,
        {"action": MedicalLabAction.RUN, "device": "all"},
    )
    assert code == 200
    assert result["instruments"]["demo"]["state"] == "running"


def test_device_route_rejects_arbitrary_joint_targets(mechanism_server) -> None:
    """
    Remote input cannot replace authored safe motion parameters.
    """
    code, _ = post(
        mechanism_server + RobotPhysicsRoute.DEVICES,
        {
            "action": MedicalLabAction.RUN,
            "device": "all",
            "joints": {"fixture__hinge": 10},
        },
    )
    assert code == 400


def test_running_mechanisms_block_robot_start(mechanism_server) -> None:
    """
    Only one native simulated execution environment owns the scene at a time.
    """
    post(
        mechanism_server + RobotPhysicsRoute.DEVICES,
        {"action": MedicalLabAction.RUN, "device": "all"},
    )
    code, _ = post(mechanism_server + RobotPhysicsRoute.RUN)
    assert code == 400


def test_initial_trajectory_preserves_measured_instrument_joints(
    mechanism_server, physics_bundle
) -> None:
    """
    The viewer starts from actual mechanisms before its first polling update.
    """
    directory = (
        physics_bundle.parents[1]
        / "scenes"
        / laboratory_robot_session.LaboratoryRobotSession.SCENE_NAME
    )
    trajectory = json.loads((directory / "trajectory.json").read_text())
    assert trajectory["frames"][0]["fixture__hinge"] == 0.2
