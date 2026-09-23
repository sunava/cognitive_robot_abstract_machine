"""
A fixed mixing demo shares the robot world's ownership and HTTP boundaries.
"""

from dataclasses import dataclass

import pytest

from cramera import laboratory_robot_session
from cramera.laboratory_physics_session import InvalidPhysicsRequest, PhysicsRoute
from cramera.laboratory_robot_session import LaboratoryRobotSession, RobotPhysicsRoute

from .test_laboratory_robot_session import ControlledRobotSimulation, robot_bundle
from .test_laboratory_physics_session import physics_bundle
from .test_server import get_json, post, server


# %% fixed recipe command
@dataclass
class MixingCommandSimulation(ControlledRobotSimulation):
    """
    Count mixing commands independently of ordinary transfer commands.
    """

    mixing_runs: int = 0
    """
    Number of fixed mixing commands received.
    """

    def run_mixing(self) -> None:
        """
        Start the fixed mixing controller and relinquish manual forces.
        """
        self.mixing_runs += 1
        self.run_program()


@pytest.fixture
def mixing_session(physics_bundle, robot_bundle, monkeypatch):
    """
    Own a robot worker with an observable mixing command.
    """
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", MixingCommandSimulation
    )
    session = LaboratoryRobotSession(physics_bundle.parents[1])
    yield session
    session.stop()


def test_mixing_owns_manipulation_until_paused(mixing_session):
    """
    Starting the recipe clears a manual hold and excludes conflicting drags.
    """
    mixing_session.start()
    mixing_session.set_target("tube_clear", [0.2, 0, 1.1])
    outcome = mixing_session.run_mixing()
    assert outcome["robot"]["state"] == "running"
    assert mixing_session.physics.mixing_runs == 1
    assert outcome["target"] is None
    with pytest.raises(InvalidPhysicsRequest):
        mixing_session.set_target("tube_clear", [0.3, 0, 1.1])
    mixing_session.pause_program()
    mixing_session.set_target("tube_clear", [0.3, 0, 1.1])


def test_mixing_http_dispatch_isolated_from_manual_world(mixing_session, server):
    """
    The fixed mixing route starts the robot controller only.
    """
    assert post(server + RobotPhysicsRoute.START)[0] == 202
    code, outcome = post(server + RobotPhysicsRoute.MIX)
    assert code == 200
    assert outcome["robot"]["state"] == "running"
    assert get_json(server + PhysicsRoute.STATE)["state"] == "idle"


def test_mixing_http_requires_started_world(mixing_session, server):
    """
    A recipe cannot run before its contact session exists.
    """
    code, outcome = post(server + RobotPhysicsRoute.MIX)
    assert code == 409
    assert outcome["ok"] is False


def test_mixing_http_rejects_recipe_overrides(mixing_session, server):
    """
    The browser command cannot supply executable or arbitrary recipe data.
    """
    code, outcome = post(server + RobotPhysicsRoute.MIX, {"command": "custom"})
    assert code == 400
    assert outcome["ok"] is False
