"""
Serialized station controls and password-protected remote robot commands.
"""

from dataclasses import dataclass, field
from http import HTTPStatus
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cramera import laboratory_physics_session, laboratory_robot_session
from cramera.laboratory_pipetting import PipettingCommand, PipettingField
from cramera.laboratory_physics_session import PhysicsRoute
from cramera.laboratory_robot_session import RobotPhysicsRoute

from .test_laboratory_physics_session import physics_bundle
from .test_laboratory_robot_session import ControlledRobotSimulation, robot_bundle
from .test_laboratory_share import SharedLaboratory, shared_laboratory, sharing_prefix
from .test_server import get_json, post, server


# %% command observation
@dataclass
class RecordedDosingCommands(ControlledRobotSimulation):
    """
    Observe validated dose forwarding without requiring a physical model.
    """

    commands: list[PipettingCommand] = field(default_factory=list)
    """
    Accepted serialized pump operations.
    """

    pipetting_started: bool = False
    """
    Whether the robot station demonstration was requested.
    """

    def pipetting_command(self, action, key=None, source=None, volume_ml=None):
        """
        Retain a station operation for isolation and argument assertions.
        """
        self.commands.append(PipettingCommand(action, key, source, volume_ml))

    def run_pipetting(self):
        """
        Record the recipe selection independently of the transfer program.
        """
        self.pipetting_started = True
        self.run_program()

    def snapshot(self):
        """
        Expose accepted operations beside the robot's ownership state.
        """
        return {
            **super().snapshot(),
            PipettingField.PIPETTING: len(self.commands),
            "pipettingStarted": self.pipetting_started,
        }


@pytest.fixture
def pipetting_api(physics_bundle, robot_bundle, monkeypatch):
    """
    Preserve real session and HTTP validation around independent pumps.
    """
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", RecordedDosingCommands
    )
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", RecordedDosingCommands
    )


@pytest.fixture(params=[PhysicsRoute, RobotPhysicsRoute])
def pipetting_routes(request):
    """
    Exercise both manual and robot station endpoints.
    """
    return request.param


# %% API ownership and forwarding
def test_pump_command_isolated_to_selected_session(
    pipetting_api, pipetting_routes, server
):
    """
    Station operations reach only the selected physical session.
    """
    for routes in (PhysicsRoute, RobotPhysicsRoute):
        assert post(server + routes.START)[0] == HTTPStatus.ACCEPTED
    code, result = post(
        server + pipetting_routes.PIPETTING,
        {"action": "dispense", "key": "tube_clear", "source": "amber", "volumeMl": 1},
    )
    assert code == HTTPStatus.OK
    assert result[PipettingField.PIPETTING] == 1
    other = RobotPhysicsRoute if pipetting_routes is PhysicsRoute else PhysicsRoute
    assert get_json(server + other.STATE)[PipettingField.PIPETTING] == 0


def test_robot_ownership_preserves_stop(pipetting_api, server):
    """
    Manual dispensing is blocked during robot execution, but stop remains usable.
    """
    post(server + RobotPhysicsRoute.START)
    post(server + RobotPhysicsRoute.RUN)
    assert (
        post(
            server + RobotPhysicsRoute.PIPETTING,
            {
                "action": "dispense",
                "key": "tube_clear",
                "source": "amber",
                "volumeMl": 1,
            },
        )[0]
        == HTTPStatus.BAD_REQUEST
    )
    assert (
        post(server + RobotPhysicsRoute.PIPETTING, {"action": "stop"})[0]
        == HTTPStatus.OK
    )
    assert get_json(server + RobotPhysicsRoute.STATE)[PipettingField.PIPETTING] == 1


def test_station_robot_recipe_has_distinct_route(pipetting_api, server):
    """
    The dedicated robot command starts the station recipe.
    """
    post(server + RobotPhysicsRoute.START)
    code, result = post(server + RobotPhysicsRoute.PIPETTE)
    assert code == HTTPStatus.OK
    assert result["pipettingStarted"] is True


def test_station_requires_active_session(pipetting_api, pipetting_routes, server):
    """
    A station operation cannot implicitly start the contact worker.
    """
    assert (
        post(server + pipetting_routes.PIPETTING, {"action": "stop"})[0]
        == HTTPStatus.CONFLICT
    )


@pytest.mark.parametrize("sharing_prefix", ["", "/laboratory"])
@pytest.mark.parametrize("operation", ["pipetting", "pipette"])
def test_shared_station_requires_password(
    shared_laboratory: SharedLaboratory, operation
):
    """
    Both dosing and robot demonstration remain behind the existing password gate.
    """
    route = shared_laboratory.options.public_path(
        f"/api/laboratory/physics/robot/{operation}"
    )
    body = json.dumps({"action": "stop"} if operation == "pipetting" else {}).encode()
    headers = {"Origin": shared_laboratory.origin, "Content-Type": "application/json"}
    assert (
        shared_laboratory.request(
            route, method="POST", body=body, headers=headers
        ).status
        == HTTPStatus.UNAUTHORIZED
    )
    assert shared_laboratory.backend.requests == []
    shared_laboratory.login()
    assert (
        shared_laboratory.request(
            route, method="POST", body=body, headers=headers
        ).status
        == HTTPStatus.OK
    )
    assert shared_laboratory.backend.requests[-1].body == body


def test_station_http_contract_does_not_require_optional_physics_packages():
    """
    The ordinary CRAMERA server remains importable without scientific dependencies.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "dataset/laboratory_pipetting_import.py"),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
