"""
Validate centrifuge command schemas without importing the physics engine.
"""

import math

import pytest

from cramera.laboratory_centrifuge_command import (
    CentrifugeAction,
    CentrifugeCommand,
    CentrifugeSlot,
    InvalidCentrifugeCommand,
)


# %% accepted commands
@pytest.mark.parametrize(
    "action", [CentrifugeAction.OPEN, CentrifugeAction.CLOSE, CentrifugeAction.STOP]
)
def test_parameterless_commands(action):
    """
    Lid and stop commands contain only the requested operation.
    """
    assert CentrifugeCommand.from_payload({"action": action}) == CentrifugeCommand(
        action
    )


def test_secure_identifies_slot_and_vessel():
    """
    Fixture engagement identifies both physical participants.
    """
    assert CentrifugeCommand.from_payload(
        {"action": "secure", "slot": "C2", "key": "tube_clear"}
    ) == CentrifugeCommand(CentrifugeAction.SECURE, CentrifugeSlot.C2, "tube_clear")


def test_release_identifies_slot():
    """
    Fixture release needs no caller-supplied object pose.
    """
    assert CentrifugeCommand.from_payload(
        {"action": "release", "slot": "C2"}
    ) == CentrifugeCommand(CentrifugeAction.RELEASE, CentrifugeSlot.C2)


@pytest.mark.parametrize(
    "parameters", [{}, {"rpm": 120}, {"duration": 20}, {"rpm": 120, "duration": 20}]
)
def test_start_accepts_optional_cycle_parameters(parameters):
    """
    Missing cycle settings remain delegated to the physical instrument defaults.
    """
    command = CentrifugeCommand.from_payload({"action": "start", **parameters})
    assert command == CentrifugeCommand(
        CentrifugeAction.START,
        rpm=parameters.get("rpm"),
        duration=parameters.get("duration"),
    )


# %% rejected commands
@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"action": "invalid"},
        {"action": "open", "rpm": 100},
        {"action": "secure", "slot": "C1"},
        {"action": "secure", "slot": "C5", "key": "tube_clear"},
        {"action": "secure", "slot": "C1", "key": ""},
        {"action": "secure", "slot": "C1", "key": 7},
        {"action": "release"},
        {"action": "release", "slot": "C1", "key": "tube_clear"},
        {"action": "start", "rpm": True},
        {"action": "start", "rpm": 0},
        {"action": "start", "rpm": -1},
        {"action": "start", "rpm": math.inf},
        {"action": "start", "rpm": math.nan},
        {"action": "start", "duration": False},
        {"action": "start", "duration": 0},
        {"action": "start", "duration": 301},
        {"action": "start", "duration": math.inf},
        {"action": "start", "duration": "20"},
    ],
)
def test_invalid_command_rejected(payload):
    """
    Malformed controls cannot be forwarded to the physical instrument.
    """
    with pytest.raises(InvalidCentrifugeCommand):
        CentrifugeCommand.from_payload(payload)
