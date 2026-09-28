"""Own the native cable laboratory and validate its browser commands."""

from __future__ import annotations

import copy
import importlib
import math
import threading
from dataclasses import dataclass, field, fields
from enum import StrEnum

from typing_extensions import Any, ClassVar, Protocol

from cramera.laboratory_physics_session import PhysicsSessionInactive


# %% browser contract
class CableRoute(StrEnum):
    """Local routes for one server-owned cable simulation."""

    STATE = "/api/cable/state"
    """Read the simulated cable and contact measurements."""
    START = "/api/cable/start"
    """Build and start a cable with the supplied configuration."""
    COMMAND = "/api/cable/command"
    """Move or open the physical gripper."""
    RESET = "/api/cable/reset"
    """Rebuild the current configuration at its initial pose."""
    STOP = "/api/cable/stop"
    """Terminate the owned simulation worker."""


class CableAction(StrEnum):
    """Commands supported by the actuated parallel gripper."""

    PINCH = "pinch"
    """Close the pads with bounded actuator force."""
    RELEASE = "release"
    """Open the pads."""
    LIFT = "lift"
    """Raise the gripper target."""
    LOWER = "lower"
    """Lower the gripper target."""
    MOVE = "move"
    """Set a gripper target in world coordinates."""


class CableField(StrEnum):
    """Named fields of cable requests and session responses."""

    ACTION = "action"
    """The requested gripper operation."""
    POSITION = "position"
    """The requested world-frame gripper target."""
    OK = "ok"
    """Whether the last startup completed successfully."""
    STATE = "state"
    """The simulation lifecycle state."""
    ERROR = "error"
    """The failure explanation."""
    VIEWER_URL = "viewerUrl"
    """The independent cable viewer."""
    COLOR = "color"
    """The optional RGBA cable appearance."""


class CableState(StrEnum):
    """Observable lifecycle states of the cable laboratory."""

    IDLE = "idle"
    """No simulation worker is advancing."""
    RUNNING = "running"
    """The background physics worker is advancing."""
    FAILED = "failed"
    """The simulation could not start or stopped unexpectedly."""


class CableSimulation(Protocol):
    """Physics operations serialized by the cable session."""

    @property
    def running(self) -> bool:
        """Whether the MuJoCo worker is advancing."""
        ...

    def start(self) -> None:
        """Start the background MuJoCo worker."""
        ...

    def stop(self) -> None:
        """Stop the worker and disconnect world synchronization."""
        ...

    def command(self, action: str, position: list[float] | None = None) -> None:
        """Set an actuator target without moving cable bodies directly."""
        ...

    def snapshot(self) -> dict[str, Any]:
        """Read consistent geometry and contact measurements."""
        ...


# %% optional physics loading
def create_physics(payload: dict[str, Any]) -> CableSimulation:
    """Validate cable configuration before loading a native simulation."""
    cable = importlib.import_module("cramera.cable")
    allowed = {attribute.name for attribute in fields(cable.CableConfig)}
    if set(payload) - allowed:
        raise ValueError(
            "Unknown cable parameters: " + ", ".join(sorted(set(payload) - allowed))
        )
    values = copy.deepcopy(payload)
    if CableField.COLOR in values:
        color = values[CableField.COLOR]
        if (
            not isinstance(color, list)
            or len(color) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
                for value in color
            )
        ):
            raise ValueError("Color must contain four RGBA values between zero and one")
        geometry = importlib.import_module(
            "semantic_digital_twin.world_description.geometry"
        )
        values[CableField.COLOR] = geometry.Color(*color)
    config = cable.CableConfig(**values)
    if config.segment_count > 100 or config.segment_count * config.segment_length > 2:
        raise ValueError(
            "The cable lab supports at most 100 elements and two metres of cable"
        )
    if config.radius > 0.05 or config.mass > 10:
        raise ValueError(
            "The cable lab supports radius up to 0.05 m and total mass up to 10 kg"
        )
    return importlib.import_module("cramera.cable_physics").CablePhysics(config=config)


# %% simulation ownership
@dataclass
class CableSession:
    """Serialize lifecycle changes and expose detached physics snapshots."""

    VIEWER_URL: ClassVar[str] = "/cable-lab.html"
    """The packaged native cable viewer."""
    TARGET_MINIMUM: ClassVar[tuple[float, float, float]] = (-1.0, -1.0, 0.02)
    """Lower bounds of the gripper workspace in metres."""
    TARGET_MAXIMUM: ClassVar[tuple[float, float, float]] = (1.0, 1.0, 1.5)
    """Upper bounds of the gripper workspace in metres."""
    physics: CableSimulation | None = field(default=None, init=False)
    """The currently owned engine, including its stopped final snapshot."""
    configuration: dict[str, Any] = field(default_factory=dict, init=False)
    """The last successfully constructed configuration."""
    error: str | None = field(default=None, init=False)
    """A startup or unexpected worker failure."""
    expected_running: bool = field(default=False, init=False)
    """Whether the session expects an active physics worker."""
    lock: Any = field(default_factory=threading.RLock, init=False, repr=False)
    """Serialize HTTP operations while the engine owns its stepping lock."""

    def start(self, configuration: dict[str, Any]) -> dict[str, Any]:
        """Build a fresh cable and release the previous worker before starting it."""
        if not isinstance(configuration, dict):
            raise ValueError("Expected a cable configuration object")
        with self.lock:
            try:
                candidate = create_physics(configuration)
            except (TypeError, ValueError):
                raise
            except Exception as error:
                self._stop()
                self.error = str(error)
                return self._status()
            self._stop()
            self.physics = candidate
            self.configuration = copy.deepcopy(configuration)
            self.error = None
            try:
                candidate.start()
                self.expected_running = candidate.running
            except Exception as error:
                self._stop()
                self.error = str(error)
            return self._status()

    def reset(self) -> dict[str, Any]:
        """Rebuild the last configuration and resume physics from the spawn pose."""
        with self.lock:
            if self.physics is None:
                raise PhysicsSessionInactive("Start the cable lab first")
            return self.start(self.configuration)

    def stop(self) -> dict[str, Any]:
        """Release the worker before returning a final detached snapshot."""
        with self.lock:
            self._stop()
            self.error = None
            return self._status()

    def _stop(self) -> None:
        """Disconnect the current worker while holding the lifecycle lock."""
        self.expected_running = False
        if self.physics is not None:
            self.physics.stop()

    def status(self) -> dict[str, Any]:
        """Read geometry and lifecycle consistently across concurrent requests."""
        with self.lock:
            return self._status()

    def _status(self) -> dict[str, Any]:
        """Add lifecycle information to a detached engine snapshot."""
        running = self.physics is not None and self.physics.running
        if self.expected_running and not running:
            self.error = "The cable physics worker stopped unexpectedly"
        result = (
            copy.deepcopy(self.physics.snapshot()) if self.physics is not None else {}
        )
        result.update(
            {
                CableField.OK: self.error is None,
                CableField.STATE: (
                    CableState.FAILED
                    if self.error
                    else CableState.RUNNING if running else CableState.IDLE
                ),
                CableField.VIEWER_URL: self.VIEWER_URL,
            }
        )
        if self.error:
            result[CableField.ERROR] = self.error
        return result

    def command(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and apply a gripper target in the running simulation."""
        with self.lock:
            if self.physics is None or not self.physics.running:
                raise PhysicsSessionInactive("Start the cable lab first")
            if not isinstance(payload, dict) or CableField.ACTION not in payload:
                raise ValueError("Expected a gripper action")
            action = CableAction(payload[CableField.ACTION])
            position = None
            if action == CableAction.MOVE:
                if set(payload) != {CableField.ACTION, CableField.POSITION}:
                    raise ValueError("Move requires an action and position")
                position = payload[CableField.POSITION]
                self._validate_position(position)
                position = position.copy()
            elif set(payload) != {CableField.ACTION}:
                raise ValueError(
                    "This gripper action accepts no position or other parameters"
                )
            self.physics.command(action, position=position)
            return self._status()

    @classmethod
    def _validate_position(cls, position: list[float]) -> None:
        """Reject nonfinite coordinates and targets beyond the lab workspace."""
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("Expected three finite position coordinates")
        for value, minimum, maximum in zip(
            position, cls.TARGET_MINIMUM, cls.TARGET_MAXIMUM
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not minimum <= value <= maximum
            ):
                raise ValueError("Gripper position is outside the cable lab workspace")
