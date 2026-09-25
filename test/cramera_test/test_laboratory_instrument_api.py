"""Standalone robot handle requests keep their HTTP and authentication boundaries."""

from dataclasses import dataclass, field
from enum import StrEnum
from http import HTTPStatus
import json
from pathlib import Path

import pytest
from typing_extensions import Any

from cramera import laboratory_robot_session
from cramera.laboratory_instrument_command import (
    InstrumentAction,
    InstrumentCommand,
    InstrumentField,
    InstrumentName,
)
from cramera.laboratory_robot_session import RobotPhysicsRoute
from cramera.laboratory_workflow_command import WorkflowField

from .test_laboratory_physics_session import physics_bundle
from .test_laboratory_robot_session import ControlledRobotSimulation, robot_bundle
from .test_laboratory_share import SharedLaboratory, shared_laboratory, sharing_prefix
from .test_server import post, server


# %% observable robot intentions
class ObservationField(StrEnum):
    """Evidence forwarded through the actual server and session."""

    REQUESTS = "instrumentRequests"
    """Ordered robot handle requests received by the simulation."""


@dataclass
class RecordedInstrumentRequests(ControlledRobotSimulation):
    """Record robot requests independently of the manual device motors."""

    commands: list[InstrumentCommand] = field(default_factory=list)
    """Handle actions accepted by the robot session."""

    def run_instrument(self, command: InstrumentCommand) -> None:
        """Accept the requested native operation without invoking a device motor."""
        self.commands.append(command)
        self.run_program()

    def snapshot(self) -> dict[str, Any]:
        """Expose the exact intentions received through HTTP."""
        snapshot = super().snapshot()
        snapshot[ObservationField.REQUESTS] = [
            {
                InstrumentField.INSTRUMENT: command.instrument,
                InstrumentField.ACTION: command.action,
            }
            for command in self.commands
        ]
        return snapshot


@pytest.fixture
def instrument_api(
    physics_bundle: Path, robot_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the real HTTP and session boundaries around recorded intentions."""
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", RecordedInstrumentRequests
    )


# %% request dispatch and validation
@pytest.mark.parametrize("instrument", list(InstrumentName))
@pytest.mark.parametrize("action", list(InstrumentAction))
def test_robot_instrument_route_dispatches_selected_operation(
    instrument_api: None,
    server: str,
    instrument: InstrumentName,
    action: InstrumentAction,
) -> None:
    """Every device/button combination reaches robot control with exact parameters."""
    post(server + RobotPhysicsRoute.START)
    payload = {InstrumentField.INSTRUMENT: instrument, InstrumentField.ACTION: action}
    status, result = post(server + RobotPhysicsRoute.INSTRUMENT, payload)
    assert status == HTTPStatus.OK
    assert result[ObservationField.REQUESTS] == [payload]


def test_instrument_request_requires_started_simulation(
    instrument_api: None, server: str
) -> None:
    """A robot operation cannot implicitly start the contact world."""
    payload = {
        InstrumentField.INSTRUMENT: InstrumentName.CENTRIFUGE,
        InstrumentField.ACTION: InstrumentAction.OPEN,
    }
    assert (
        post(server + RobotPhysicsRoute.INSTRUMENT, payload)[0] == HTTPStatus.CONFLICT
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {InstrumentField.INSTRUMENT: InstrumentName.CENTRIFUGE},
        {
            InstrumentField.INSTRUMENT: "unknown",
            InstrumentField.ACTION: InstrumentAction.OPEN,
        },
        {
            InstrumentField.INSTRUMENT: InstrumentName.CENTRIFUGE,
            InstrumentField.ACTION: "unknown",
        },
        {
            InstrumentField.INSTRUMENT: InstrumentName.CENTRIFUGE,
            InstrumentField.ACTION: InstrumentAction.OPEN,
            WorkflowField.PREPARE: True,
        },
    ],
)
def test_invalid_instrument_request_is_bad_request(
    instrument_api: None, server: str, payload: dict[str, Any]
) -> None:
    """Malformed handle requests are rejected before any action is dispatched."""
    post(server + RobotPhysicsRoute.START)
    assert (
        post(server + RobotPhysicsRoute.INSTRUMENT, payload)[0]
        == HTTPStatus.BAD_REQUEST
    )


# %% protected public access
@pytest.mark.parametrize("sharing_prefix", ["", "/laboratory"])
def test_robot_instrument_requires_password_and_same_origin(
    shared_laboratory: SharedLaboratory,
) -> None:
    """The standalone controls inherit the laboratory password and origin boundary."""
    route = shared_laboratory.options.public_path(RobotPhysicsRoute.INSTRUMENT)
    payload = json.dumps(
        {
            InstrumentField.INSTRUMENT: InstrumentName.REFRIGERATOR,
            InstrumentField.ACTION: InstrumentAction.OPEN_CLOSE,
        }
    ).encode()
    headers = {"Origin": shared_laboratory.origin, "Content-Type": "application/json"}
    assert (
        shared_laboratory.request(
            route, method="POST", body=payload, headers=headers
        ).status
        == HTTPStatus.UNAUTHORIZED
    )
    assert shared_laboratory.backend.requests == []
    shared_laboratory.login()
    assert (
        shared_laboratory.request(
            route,
            method="POST",
            body=payload,
            headers={**headers, "Origin": "https://untrusted.invalid"},
        ).status
        == HTTPStatus.FORBIDDEN
    )
    assert shared_laboratory.backend.requests == []
    assert (
        shared_laboratory.request(
            route, method="POST", body=payload, headers=headers
        ).status
        == HTTPStatus.OK
    )
    assert shared_laboratory.backend.requests[-1].path == RobotPhysicsRoute.INSTRUMENT
    assert shared_laboratory.backend.requests[-1].body == payload
