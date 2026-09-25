"""
Workshop commands preserve a single synchronized physical world.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest

from cramera.workshop_session import (
    InvalidWorkshopRequest,
    WorkshopAction,
    WorkshopCommand,
    WorkshopSession,
    WorkshopSessionInactive,
)


# %% observable deterministic dynamics
@dataclass
class ContactWorld:
    """
    Expose worker progression and mutations without requiring a physics engine.
    """

    time: float = 0.0
    """
    Elapsed simulated seconds.
    """

    held: str | None = None
    """
    Selected manipulation target.
    """

    stepped: threading.Event = field(default_factory=threading.Event)
    """
    Signal that the worker has advanced the model.
    """

    released: int = 0
    """
    Number of force-release operations.
    """

    poses: dict = field(default_factory=lambda: {"screw_box": [0, 0, 1, 0, 0, 0, 1]})
    """
    Detached pose payload source.
    """

    def step(self, duration: float) -> None:
        """
        Record simulated progression and wake its observer.
        """
        self.time += duration
        self.stepped.set()

    def snapshot(self) -> dict:
        """
        Return engine-owned state to exercise snapshot detachment.
        """
        return {"time": self.time, "held": self.held, "objects": self.poses}

    def grab(self, key: str) -> None:
        """
        Select one target.
        """
        self.held = key

    def move(self, pose: list[float]) -> None:
        """
        Store the selected target pose.
        """
        self.poses[self.held] = pose

    def release(self) -> None:
        """
        Remove the manipulation force.
        """
        self.held = None
        self.released += 1

    def reset(self) -> None:
        """
        Restore the initial physical state.
        """
        self.time = 0.0
        self.held = None

    def set_drawer(self, joint: str, value: float) -> None:
        """
        Accept the drawer interface without mutating unrelated parts.
        """

    def drive(self, direction: int) -> None:
        """
        Accept the bounded thread controller interface.
        """


# %% command validation and worker ownership
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"action": "unknown"},
        {"action": "start", "object": "screw_box"},
        {"action": "move", "pose": [0, 0, float("nan"), 0, 0, 0, 1]},
        {"action": "move", "pose": [False, 0, 1, 0, 0, 0, 1]},
        {"action": "move", "pose": [0, 0, 1, 0, 0, 0, 0]},
        {"action": "drive", "direction": True},
        {"action": "drive", "direction": 2},
        {"action": "drawer", "joint": "drawer", "position": float("inf")},
    ],
)
def test_invalid_commands_are_rejected_before_physical_mutation(payload) -> None:
    """
    Malformed commands cannot enter the serialized dynamics worker.
    """
    with pytest.raises(InvalidWorkshopRequest):
        WorkshopCommand.from_json(payload)


def test_pause_releases_force_and_freezes_simulation_until_resumed() -> None:
    """
    Pausing preserves a state without letting a held body keep moving.
    """
    world = ContactWorld()
    session = WorkshopSession(factory=lambda: world)
    try:
        session.command(WorkshopCommand(WorkshopAction.START))
        assert world.stepped.wait(1)
        worker = session.worker
        session.command(WorkshopCommand(WorkshopAction.GRAB, object="screw_box"))
        paused = session.command(WorkshopCommand(WorkshopAction.PAUSE))
        assert paused["running"] is False
        assert paused["held"] is None
        assert world.released == 1
        assert session.state()["time"] == paused["time"]
        session.command(WorkshopCommand(WorkshopAction.START))
        assert session.worker is worker
    finally:
        session.close()
    assert not worker.is_alive()


def test_snapshot_does_not_expose_engine_owned_arrays() -> None:
    """
    A returned JSON payload cannot silently move a simulated object.
    """
    world = ContactWorld()
    session = WorkshopSession(factory=lambda: world)
    try:
        state = session.command(WorkshopCommand(WorkshopAction.RESET))
        state["objects"]["screw_box"][0] = 100
        assert world.poses["screw_box"][0] == 0
        assert state["running"] is False
    finally:
        session.close()


def test_manipulation_requires_a_running_session() -> None:
    """
    No request can apply a force to a paused or uninitialized world.
    """
    session = WorkshopSession(factory=ContactWorld)
    with pytest.raises(WorkshopSessionInactive):
        session.command(WorkshopCommand(WorkshopAction.GRAB, object="screw_box"))
    assert session.physics is None


def test_inactivity_pauses_and_releases_the_world() -> None:
    """
    Closing a client cannot leave a held-object controller running forever.
    """
    world = ContactWorld()
    session = WorkshopSession(factory=lambda: world, inactivity_timeout=0.0)
    try:
        session.command(WorkshopCommand(WorkshopAction.START))
        worker = session.worker
        session.expire()
        assert session.state()["running"] is False
        assert world.released == 1
    finally:
        session.close()
    assert not worker.is_alive()


def test_target_command_names_the_currently_held_body() -> None:
    """
    A browser move carries an explicit object identity alongside its pose.
    """
    world = ContactWorld()
    session = WorkshopSession(factory=lambda: world)
    pose = [0.1, 0, 1, 0, 0, 0, 1]
    try:
        session.command(WorkshopCommand(WorkshopAction.START))
        session.command(WorkshopCommand(WorkshopAction.GRAB, object="screw_box"))
        command = WorkshopCommand.from_json(
            {"action": "move", "object": "screw_box", "pose": pose}
        )
        session.command(command)
        assert world.poses["screw_box"] == pose
    finally:
        session.close()


def test_stale_client_cannot_move_a_different_held_object() -> None:
    """
    A target from a previous selection cannot control another client's body.
    """
    world = ContactWorld()
    session = WorkshopSession(factory=lambda: world)
    try:
        session.command(WorkshopCommand(WorkshopAction.START))
        session.command(WorkshopCommand(WorkshopAction.GRAB, object="screw_box"))
        with pytest.raises(InvalidWorkshopRequest):
            session.command(
                WorkshopCommand(
                    WorkshopAction.MOVE, object="nail_box", pose=[0.1, 0, 1, 0, 0, 0, 1]
                )
            )
        assert world.poses["screw_box"] == [0, 0, 1, 0, 0, 0, 1]
    finally:
        session.close()


def test_closed_listener_cannot_restart_its_physics_worker() -> None:
    """
    An already admitted HTTP request cannot resurrect a shut-down session.
    """
    session = WorkshopSession(factory=ContactWorld)
    session.command(WorkshopCommand(WorkshopAction.START))
    worker = session.worker
    session.close()
    with pytest.raises(WorkshopSessionInactive):
        session.command(WorkshopCommand(WorkshopAction.START))
    assert not worker.is_alive()
    assert session.worker is None
    assert session.state()["available"] is False
