"""
Performing a plan in a scene that is already running.

A scene brought up by the Plan Builder stands idle between plans, so a plan asked for
over the bridge cannot wait for a motion tick to carry it: it brings its own thread, the
way teleoperation does. Only one plan runs at a time — the execution environment a plan
is performed in is process-wide state — and what became of it is kept, because a plan
that fails says so nowhere else the builder can read.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from typing_extensions import TYPE_CHECKING, Any, Dict, Optional

from cramera.live.requested_plan import RequestedPlan
from cramera.logging_setup import get_logger

if TYPE_CHECKING:
    from coraplex.visualization import WorldVisualization

logger = get_logger(__name__)


class RunState(StrEnum):
    """
    Where a scene is between plans.
    """

    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class RunField(StrEnum):
    """
    Key a run reports one part of its outcome under.
    """

    STATE = "state"
    ERROR = "error"


class PlanAlreadyRunning(Exception):
    """
    Raised when a plan is asked for while the scene is still performing one.
    """


class PlanRunnerUnavailable(Exception):
    """
    Raised when a plan is asked for from a scene that does not serve plans.
    """


@dataclass
class PlanRun:
    """
    What became of the plan a scene was last asked to perform.
    """

    state: RunState = RunState.IDLE
    """
    Whether a plan is running, and how the last one ended.
    """

    error: Optional[str] = None
    """
    What went wrong, while the last run is a failed one.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    """
    Guards the outcome, which the run thread writes and HTTP threads read.
    """

    def begin(self) -> None:
        """
        Take the scene for a new run.

        :raises PlanAlreadyRunning: If the scene is still performing a plan.
        """
        with self._lock:
            if self.state is RunState.RUNNING:
                raise PlanAlreadyRunning(
                    "the scene is still performing a plan — wait for it to finish"
                )
            self.state = RunState.RUNNING
            self.error = None

    def finish(self) -> None:
        """
        Record that the plan was performed.
        """
        with self._lock:
            self.state = RunState.FINISHED
            self.error = None

    def fail(self, error: BaseException) -> None:
        """
        Record that the plan could not be performed.

        :param error: What stopped it.
        """
        with self._lock:
            self.state = RunState.FAILED
            self.error = "%s: %s" % (type(error).__name__, error)

    def payload(self) -> Dict[str, Any]:
        """
        The outcome as the builder polls it.
        """
        with self._lock:
            return {
                RunField.STATE.value: self.state.value,
                RunField.ERROR.value: self.error,
            }


@dataclass
class PlanRunner:
    """
    The running scene's ability to perform plans asked for over the bridge.
    """

    context: Context
    """
    The scene's context, whose world every requested plan is resolved against.
    """

    visualization: Optional["WorldVisualization"] = None
    """
    The viewer to publish each plan to, or None for a scene nobody is watching.
    """

    run: PlanRun = field(default_factory=PlanRun)
    """
    What became of the plan this scene was last asked to perform.
    """

    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    """
    The thread performing the current plan, while one is being performed.
    """

    def submit(self, requested: RequestedPlan) -> None:
        """
        Start performing a plan, and return without waiting for it.

        :param requested: The plan to perform.
        :raises PlanAlreadyRunning: If the scene is still performing one.
        """
        self.run.begin()
        self._thread = threading.Thread(
            target=self._perform, args=(requested,), name="plan-runner", daemon=True
        )
        self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> None:
        """
        Wait for the plan being performed to end.

        :param timeout: How long to wait at most, or None to wait as long as it takes.
        """
        if self._thread is not None:
            self._thread.join(timeout)

    def _perform(self, requested: RequestedPlan) -> None:
        """
        Build the plan against the live world and perform it, on this runner's thread.

        :param requested: The plan to perform.
        """
        try:
            plan = requested.plan(self.context)
            if self.visualization is not None:
                self.visualization.attach_plan(plan)
            with simulated_robot:
                plan.perform()
        except Exception as error:  # a bad plan must not take the scene down with it
            logger.exception("the requested plan could not be performed")
            self.run.fail(error)
            return
        self.run.finish()
