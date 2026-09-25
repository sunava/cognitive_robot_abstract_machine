"""Serialize workshop controls and advance one shared contact simulation."""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from _thread import RLock

from typing_extensions import Any, Callable, Protocol


# %% bounded command vocabulary
class WorkshopRoute(StrEnum):
    """HTTP operations for the independent workshop world."""

    STATE = "/api/workshop/state"
    """Read the current physical state."""
    COMMAND = "/api/workshop/command"
    """Apply a validated control request."""


class WorkshopAction(StrEnum):
    """Supported simulation and manipulation operations."""

    START = "start"
    """Resume physical integration."""
    PAUSE = "pause"
    """Release manipulation and pause integration."""
    RESET = "reset"
    """Restore authored initial poses."""
    GRAB = "grab"
    """Apply a bounded manipulation controller to one body."""
    MOVE = "move"
    """Change the active controller target."""
    RELEASE = "release"
    """Remove the active manipulation force."""
    DRAWER = "drawer"
    """Change a force-limited drawer target."""
    DRIVE = "drive"
    """Select forward, reverse or stopped screwdriver rotation."""


class WorkshopField(StrEnum):
    """Fields exchanged with the browser controls."""

    ACTION = "action"
    """Requested operation."""
    OBJECT = "object"
    """Scene object key."""
    POSE = "pose"
    """World target position and XYZW quaternion."""
    JOINT = "joint"
    """Articulated drawer identifier."""
    POSITION = "position"
    """Drawer travel in metres."""
    DIRECTION = "direction"
    """Signed driver command."""
    RUNNING = "running"
    """Whether dynamics integration is active."""
    AVAILABLE = "available"
    """Whether the session can accept controls."""
    OBJECTS = "objects"
    """Simulated independent object poses."""
    JOINTS = "joints"
    """Measured articulated positions."""
    HELD = "held"
    """Current force-controller object."""
    CONTACTS = "contacts"
    """Measured contact geometry and forces."""
    THREAD = "thread"
    """Current ideal-thread engagement."""
    ENGAGED = "engaged"
    """Whether the driver and screw are coupled."""
    TIME = "time"
    """Elapsed simulated seconds."""
    ERROR = "error"
    """Recoverable simulation failure."""


class InvalidWorkshopRequest(ValueError):
    """A control request violates the public workshop schema."""


class WorkshopSessionInactive(RuntimeError):
    """Manipulation was requested while integration is paused."""


@dataclass(frozen=True)
class WorkshopCommand:
    """One parsed operation with only the fields its action consumes."""

    action: WorkshopAction
    """Requested physical operation."""
    object: str | None = None
    """Independent body to manipulate."""
    pose: list[float] | None = None
    """Finite world coordinates followed by a nonzero XYZW quaternion."""
    joint: str | None = None
    """Drawer joint receiving a new target."""
    position: float | None = None
    """Requested drawer travel in metres."""
    direction: int = 0
    """Driver rotation sign."""

    @staticmethod
    def finite(value: Any) -> bool:
        """Reject booleans and non-finite JSON numeric values."""
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )

    @classmethod
    def from_json(cls, payload: Any) -> WorkshopCommand:
        """Validate the complete request before any world mutation occurs."""
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get(WorkshopField.ACTION), str)
            or payload[WorkshopField.ACTION] not in WorkshopAction
        ):
            raise InvalidWorkshopRequest("Unknown workshop action")
        action = WorkshopAction(payload[WorkshopField.ACTION])
        fields = {
            WorkshopAction.GRAB: {WorkshopField.OBJECT},
            WorkshopAction.MOVE: {WorkshopField.OBJECT, WorkshopField.POSE},
            WorkshopAction.DRAWER: {WorkshopField.JOINT, WorkshopField.POSITION},
            WorkshopAction.DRIVE: {WorkshopField.DIRECTION},
        }.get(action, set())
        if set(payload) != fields | {WorkshopField.ACTION}:
            raise InvalidWorkshopRequest("Unexpected workshop command fields")
        if action in (WorkshopAction.GRAB, WorkshopAction.MOVE) and (
            not isinstance(payload[WorkshopField.OBJECT], str)
            or not payload[WorkshopField.OBJECT]
        ):
            raise InvalidWorkshopRequest("Expected an object key")
        if action == WorkshopAction.MOVE:
            pose = payload[WorkshopField.POSE]
            if (
                not isinstance(pose, list)
                or len(pose) != 7
                or not all(cls.finite(value) for value in pose)
                or math.hypot(*pose[3:]) < 1e-6
            ):
                raise InvalidWorkshopRequest(
                    "Expected a finite pose with a nonzero quaternion"
                )
        if action == WorkshopAction.DRAWER and (
            not isinstance(payload[WorkshopField.JOINT], str)
            or not cls.finite(payload[WorkshopField.POSITION])
        ):
            raise InvalidWorkshopRequest("Expected a drawer joint and finite position")
        if action == WorkshopAction.DRIVE and (
            type(payload[WorkshopField.DIRECTION]) is not int
            or payload[WorkshopField.DIRECTION] not in (-1, 0, 1)
        ):
            raise InvalidWorkshopRequest("Driver direction must be -1, 0 or 1")
        return cls(**{**payload, WorkshopField.ACTION: action})


# %% engine contract and synchronized lifetime
class WorkshopDynamics(Protocol):
    """Contact dynamics interface with detached observable state."""

    held: str | None
    """Body currently receiving manipulation forces."""

    def step(self, duration: float) -> None:
        """Advance physical integration."""

    def snapshot(self) -> dict[str, Any]:
        """Read current model state."""

    def grab(self, key: str) -> None:
        """Select one body for force-based movement."""

    def move(self, pose: list[float]) -> None:
        """Change the manipulation controller target."""

    def release(self) -> None:
        """Remove manipulation forces."""

    def set_drawer(self, joint: str, value: float) -> None:
        """Change a drawer actuator target."""

    def drive(self, direction: int) -> None:
        """Change the screwdriver rotation command."""

    def reset(self) -> None:
        """Restore the authored initial physical state."""


@dataclass
class WorkshopSession:
    """Own one lazily constructed physical world and its stepping worker."""

    factory: Callable[[], WorkshopDynamics]
    """Construct the configured workshop engine on first use."""
    step_duration: float = 0.01
    """Simulated seconds advanced per worker cycle."""
    inactivity_timeout: float = 90.0
    """Pause interval after all browser clients stop polling."""
    physics: WorkshopDynamics | None = field(default=None, init=False)
    """Engine instance shared by every authenticated client."""
    worker: threading.Thread | None = field(default=None, init=False)
    """Single integration thread."""
    running: bool = field(default=False, init=False)
    """Whether the worker may advance time."""
    closed: bool = field(default=False, init=False)
    """Terminal shutdown state preventing a late request from restarting a worker."""
    lock: RLock = field(default_factory=threading.RLock, init=False, repr=False)
    """Serialize controls, snapshots and solver integration."""
    stopped: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    """Interrupt the worker on server shutdown."""
    last_activity: float = field(default_factory=time.monotonic, init=False)
    """Latest client interaction on the monotonic clock."""
    error: str | None = field(default=None, init=False)
    """Reported integration failure, cleared by reset."""

    def state(self) -> dict[str, Any]:
        """Read a detached snapshot and keep an observing client active."""
        with self.lock:
            self.last_activity = time.monotonic()
            return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        """Copy physical state while the session lock is held."""
        result = (
            copy.deepcopy(self.physics.snapshot())
            if self.physics is not None
            else {
                WorkshopField.OBJECTS: {},
                WorkshopField.JOINTS: {},
                WorkshopField.HELD: None,
                WorkshopField.CONTACTS: [],
                WorkshopField.THREAD: {WorkshopField.ENGAGED: False},
                WorkshopField.TIME: 0.0,
            }
        )
        result.update(
            {
                WorkshopField.RUNNING: self.running,
                WorkshopField.AVAILABLE: self.error is None and not self.closed,
            }
        )
        if self.error:
            result[WorkshopField.ERROR] = self.error
        return result

    def command(self, command: WorkshopCommand) -> dict[str, Any]:
        """Apply one operation atomically with respect to the contact solver."""
        with self.lock:
            if self.closed:
                raise WorkshopSessionInactive("The workshop session is closed")
            self.last_activity = time.monotonic()
            if command.action in (WorkshopAction.START, WorkshopAction.RESET):
                if self.physics is None:
                    self.physics = self.factory()
                if command.action == WorkshopAction.RESET:
                    self.physics.reset()
                    self.error = None
                else:
                    if self.error:
                        raise WorkshopSessionInactive(
                            "Reset the workshop after a simulation failure"
                        )
                    self.running = True
                    if self.worker is None:
                        self.stopped.clear()
                        self.worker = threading.Thread(
                            target=self._run, name="workshop-physics", daemon=True
                        )
                        self.worker.start()
            elif command.action == WorkshopAction.PAUSE:
                self._pause()
            else:
                if not self.running or self.physics is None:
                    raise WorkshopSessionInactive("Start workshop physics first")
                if command.action == WorkshopAction.GRAB:
                    self.physics.grab(command.object)
                elif command.action == WorkshopAction.MOVE:
                    if command.object != self.physics.held:
                        raise InvalidWorkshopRequest(
                            "The requested object is no longer held"
                        )
                    self.physics.move(command.pose)
                elif command.action == WorkshopAction.RELEASE:
                    self.physics.release()
                elif command.action == WorkshopAction.DRAWER:
                    self.physics.set_drawer(command.joint, command.position)
                elif command.action == WorkshopAction.DRIVE:
                    self.physics.drive(command.direction)
            return self._snapshot()

    def _pause(self) -> None:
        """Remove external manipulation before suspending physical time."""
        if self.physics is not None and self.running:
            self.physics.drive(0)
            self.physics.release()
        self.running = False

    def expire(self) -> None:
        """Release and pause a workshop without observing clients."""
        with self.lock:
            if (
                self.running
                and time.monotonic() - self.last_activity >= self.inactivity_timeout
            ):
                self._pause()

    def _run(self) -> None:
        """Advance bounded physical intervals and stop controls on solver failure."""
        while not self.stopped.is_set():
            started = time.monotonic()
            with self.lock:
                self.expire()
                if self.running:
                    try:
                        self.physics.step(self.step_duration)
                    except Exception as error:
                        self._pause()
                        self.error = str(error)
            self.stopped.wait(
                max(0.001, self.step_duration - (time.monotonic() - started))
            )

    def close(self) -> None:
        """Join integration before the owning server releases its resources."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.stopped.set()
            self._pause()
            worker = self.worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=5)
        with self.lock:
            self.worker = None
