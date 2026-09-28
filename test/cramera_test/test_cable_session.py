"""
Lifecycle and HTTP boundaries of the independent cable laboratory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus
import json
import urllib.error
import urllib.request
import time
from unittest.mock import Mock

import pytest

from cramera import cable_session
from cramera.cable_session import CableAction, CableRoute, CableSession
from cramera.laboratory_physics_session import PhysicsSessionInactive

from .test_server import get_json, post, server


# %% observable simulation lifecycle
@dataclass
class ControlledCable:
    """
    Record lifecycle changes independently of the physics worker.
    """

    running: bool = False
    """
    Whether the simulation currently advances.
    """

    stopped: bool = False
    """
    Whether its resources have been released.
    """

    commands: list = field(default_factory=list)
    """
    Accepted gripper commands.
    """

    def start(self) -> None:
        """
        Begin advancing the simulation.
        """
        self.running = True

    def stop(self) -> None:
        """
        Release this simulation's resources.
        """
        self.running = False
        self.stopped = True

    def command(self, action: str, position: list[float] | None = None) -> None:
        """
        Record the requested motion without changing any cable pose.
        """
        self.commands.append((action, position))

    def snapshot(self) -> dict:
        """
        Expose detached command data through the same snapshot boundary.
        """
        return {"commands": self.commands, "physical": True}


@pytest.fixture
def cable_factory(monkeypatch) -> Mock:
    """
    Replace optional physics loading with a new observable engine per start.
    """
    factory = Mock(side_effect=lambda config: ControlledCable())
    monkeypatch.setattr(cable_session, "create_physics", factory)
    return factory


def test_stop_releases_worker_and_returns_idle(cable_factory) -> None:
    """
    Stopping the owner terminates the engine before reporting idle.
    """
    session = CableSession()
    assert session.start({})["state"] == "running"
    physics = session.physics
    assert session.stop()["state"] == "idle"
    assert physics.stopped is True
    session.stop()


def test_restart_and_reset_release_previous_worker(cable_factory) -> None:
    """
    Applying configuration and resetting leave exactly one live engine.
    """
    session = CableSession()
    session.start({"mass": 0.07})
    first = session.physics
    session.start({"mass": 0.08})
    second = session.physics
    assert first.stopped is True
    session.reset()
    assert second.stopped is True
    assert cable_factory.call_args.args == ({"mass": 0.08},)
    session.stop()


def test_failed_start_releases_partial_engine(cable_factory) -> None:
    """
    An engine that fails during startup cannot retain a live worker.
    """
    physics = ControlledCable()
    physics.start = Mock(side_effect=RuntimeError("Cannot compile cable"))
    cable_factory.side_effect = None
    cable_factory.return_value = physics
    session = CableSession()
    answer = session.start({})
    assert answer["ok"] is False
    assert answer["state"] == "failed"
    assert answer["error"] == "Cannot compile cable"
    assert physics.stopped is True


def test_snapshot_does_not_expose_engine_mutable_state(cable_factory) -> None:
    """
    Mutating a response cannot alter the next simulation snapshot.
    """
    session = CableSession()
    session.start({})
    result = session.command({"action": CableAction.PINCH})
    result["commands"].clear()
    assert session.status()["commands"] == [(CableAction.PINCH, None)]
    session.stop()


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "weld"},
        {"action": CableAction.PINCH, "position": [0, 0, 1]},
        {"action": CableAction.MOVE, "position": [True, 0, 1]},
        {"action": CableAction.MOVE, "position": [0, float("nan"), 1]},
        {"action": CableAction.MOVE, "position": [0, 0]},
        {"action": CableAction.MOVE, "position": [0, 0, 100]},
        {"action": CableAction.MOVE},
        {"action": CableAction.PINCH, "segment": 1},
    ],
)
def test_rejects_invalid_commands_before_engine_mutation(
    cable_factory, payload
) -> None:
    """
    Only bounded gripper motion can cross the session boundary.
    """
    session = CableSession()
    session.start({})
    with pytest.raises(ValueError):
        session.command(payload)
    assert session.physics.commands == []
    session.stop()


def test_command_requires_running_simulation() -> None:
    """
    A gripper command never implicitly creates a simulation.
    """
    with pytest.raises(PhysicsSessionInactive):
        CableSession().command({"action": CableAction.PINCH})


def test_invalid_configuration_type_keeps_existing_worker(cable_factory) -> None:
    """
    An invalid replacement is a request error and leaves the valid experiment running.
    """
    session = CableSession()
    session.start({})
    original = session.physics
    cable_factory.side_effect = TypeError("mass must be numeric")
    with pytest.raises(TypeError):
        session.start({"mass": None})
    assert original.running is True
    session.stop()


def test_startup_value_error_is_reported_as_engine_failure(cable_factory) -> None:
    """
    A validated engine failing to start reports failure and releases resources.
    """
    physics = ControlledCable()
    physics.start = Mock(side_effect=ValueError("Cannot start worker"))
    cable_factory.side_effect = None
    cable_factory.return_value = physics
    session = CableSession()
    assert session.start({})["state"] == "failed"
    assert physics.stopped is True


# %% public HTTP contract
def test_http_start_command_reset_stop(cable_factory, server) -> None:
    """
    The browser can operate a cable without a robot scene bundle.
    """
    assert get_json(server + CableRoute.STATE)["state"] == "idle"
    code, started = post(server + CableRoute.START, {"mass": 0.07})
    assert code == HTTPStatus.ACCEPTED
    assert started["viewerUrl"] == "/cable-lab.html"
    code, answer = post(server + CableRoute.COMMAND, {"action": CableAction.PINCH})
    assert code == HTTPStatus.OK
    assert answer["commands"] == [[CableAction.PINCH, None]]
    assert post(server + CableRoute.RESET)[0] == HTTPStatus.OK
    assert post(server + CableRoute.STOP)[1]["state"] == "idle"


def test_http_inactive_command_is_conflict(cable_factory, server) -> None:
    """
    Commands for an absent worker return an actionable HTTP error.
    """
    assert (
        post(server + CableRoute.COMMAND, {"action": CableAction.PINCH})[0]
        == HTTPStatus.CONFLICT
    )


@pytest.mark.parametrize("route", [CableRoute.RESET, CableRoute.STOP])
def test_http_parameterless_operations_reject_payload(
    cable_factory, server, route
) -> None:
    """
    Reset and stop cannot accidentally accept unsupported cable settings.
    """
    assert post(server + route, {"mass": 0.1})[0] == HTTPStatus.BAD_REQUEST


def test_http_cross_origin_cannot_start_simulation(cable_factory, server) -> None:
    """
    A foreign page cannot create a background physics worker.
    """
    request = urllib.request.Request(
        server + CableRoute.START,
        method="POST",
        data=json.dumps({}).encode(),
        headers={"Origin": "https://example.com"},
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request)
    assert caught.value.code == HTTPStatus.FORBIDDEN
    cable_factory.assert_not_called()


@pytest.mark.parametrize(
    "configuration",
    [
        {"strategy": "connect"},
        {"use_composite": False},
        {"mass_per_segment": 0.01},
        {"segment_count": 101},
        {"segment_count": 10, "segment_length": 0.3},
        {"color": [1, 0, 0]},
        {"color": [1, 0, 0, float("nan")]},
    ],
)
def test_config_boundary_rejects_removed_or_unbounded_settings(configuration) -> None:
    """
    Unsupported cable settings fail before any physics worker starts.
    """
    with pytest.raises(ValueError):
        cable_session.create_physics(configuration)


def test_http_real_engine_emits_advancing_native_geometry(server) -> None:
    """
    The complete API returns serializable measured geometry from a live worker.
    """
    code, initial = post(server + CableRoute.START, {"anchor_to_parent": False})
    assert code == HTTPStatus.ACCEPTED
    assert initial["physical"] is True
    assert len(initial["segments"]) == initial["config"]["segment_count"]
    deadline = time.monotonic() + 3
    latest = initial
    while (
        latest["simulation_time"] <= initial["simulation_time"]
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
        latest = get_json(server + CableRoute.STATE)
    assert latest["simulation_time"] > initial["simulation_time"]
    assert (
        post(server + CableRoute.COMMAND, {"action": CableAction.PINCH})[0]
        == HTTPStatus.OK
    )
    assert post(server + CableRoute.RESET)[1]["physical"] is True
    assert post(server + CableRoute.STOP)[1]["state"] == "idle"
