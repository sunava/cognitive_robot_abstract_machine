"""Execute modular instrument mechanism demonstrations as native Coraplex plans."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum

from typing_extensions import Any, Protocol

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType, TaskStatus
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.plans.executables import Executable, GiskardExecutable
from coraplex.plans.factories import sequential
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ActionNode, ExecutionBoundaryNode, PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from cramera.laboratory_plan_world import LaboratoryPlanWorld
from cramera.medical_lab_command import (
    InvalidMedicalLabCommand,
    MedicalLabAction,
    MedicalLabCommand,
    MedicalLabSelection,
)
from cramera.medical_lab_description import MedicalLabDescription
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World


# %% independent physical requests
class MedicalLabState(StrEnum):
    """Mechanism demonstration states exposed to the browser."""

    IDLE = "idle"
    """No native mechanism demonstration currently owns the devices."""
    RUNNING = "running"
    """A native sequence is waiting for or executing a physical operation."""
    PAUSED = "paused"
    """The same pending operation is retained until resume."""
    COMPLETE = "complete"
    """Every native action received successful physical acknowledgment."""
    FAILED = "failed"
    """Native execution or a measured physical operation failed."""


class MedicalLabPlanField(StrEnum):
    """Measured demonstration progress published by the native execution owner."""

    STATE = "state"
    """Operator-visible lifecycle."""
    DEVICE = "device"
    """Selected resource or all-resource inspection."""
    ACTION = "action"
    """English label of the currently pending physical operation."""
    COMPLETED = "completed"
    """Number of physically acknowledged native actions."""
    TOTAL = "total"
    """Number of independent actions in the requested recipe."""
    ERROR = "error"
    """Physical or native failure explanation."""
    FRAMEWORK = "framework"
    """Name of the plan framework owning the sequence."""
    CORAPLEX = "coraplex"
    """Native framework identifier."""
    NATIVE_STATUS = "nativeStatus"
    """Lifecycle name reported by the actual native root node."""


@dataclass(frozen=True)
class MedicalLabJointTarget:
    """One bounded reference to an authored scalar mechanism joint."""

    name: str
    """Exact joint name shared by the physical and semantic worlds."""
    position: float
    """Requested physical coordinate in metres or radians."""
    velocity: float
    """Maximum reference speed in metres or radians per second."""


@dataclass(frozen=True)
class MedicalLabMotionRequest:
    """An immutable group of joint targets acknowledged only after measurement."""

    device: str
    """Installed resource selected for this independent action."""
    targets: tuple[MedicalLabJointTarget, ...]
    """Mechanism coordinates to reach together using bounded motor references."""
    label: str
    """English description of the current inspection or return step."""


@dataclass
class MedicalLabMotionFailed(PlanFailure):
    """A physical mechanism failed its measured acceptance condition."""

    message: str
    """Physical rejection reason retained in the native plan."""

    def error_message(self) -> str:
        """Describe the condition that prevented physical completion."""
        return self.message


class MedicalLabExecutor(Protocol):
    """A serialized physical operation runner bound to one native context."""

    def execute(self, request: MedicalLabMotionRequest) -> None:
        """Return only after measured completion, or raise a native failure."""


@dataclass(eq=False)
class MedicalLabContext(Context):
    """Native world and robot context with a separately owned instrument executor."""

    executor: MedicalLabExecutor = field(kw_only=True)
    """Physical owner that receives deferred mechanism operations."""


@dataclass(eq=False, repr=False)
class MedicalLabMotionNode(ExecutionBoundaryNode):
    """Defer physical instrument movement until the native action executes."""

    request: MedicalLabMotionRequest = field(kw_only=True)
    """Immutable targets kept between preceding and following native actions."""

    def notify(self) -> None:
        """Expand the native tree without changing any physical mechanism."""

    def parse(self) -> MedicalLabMotionExecutable:
        """Build an executable bound to the plan's own physical executor."""
        return MedicalLabMotionExecutable(node=self, context=self.plan.context)


@dataclass
class MedicalLabMotionExecutable(Executable):
    """Execute one mechanism action with native lifecycle and world-state records."""

    node: MedicalLabMotionNode = field(kw_only=True)
    """Deferred node whose scope records actual execution and failure."""

    def execute(self) -> None:
        """Dispatch simulated mechanism motion only after native resume."""
        with self.node.execution_scope():
            while self.node.is_paused and not self.node.is_interrupted:
                time.sleep(0.01)
            if self.node.is_interrupted:
                self.node.status = TaskStatus.INTERRUPTED
                return
            if GiskardExecutable.execution_type == ExecutionType.NO_EXECUTION:
                return
            if (
                GiskardExecutable.execution_type != ExecutionType.SIMULATED
                or not isinstance(self.context, MedicalLabContext)
            ):
                raise MedicalLabMotionFailed(
                    "Instrument motions require their simulated medical-laboratory context"
                )
            action = next(
                (node for node in self.node.path if isinstance(node, ActionNode)), None
            )
            if action is not None:
                action.create_execution_data_pre_perform()
                action.execution_data.execution_start_world_state = (
                    self.context.world.state._data.copy()
                )
            try:
                self.context.executor.execute(self.node.request)
            finally:
                if action is not None:
                    action.update_execution_data_post_perform()
                    action.execution_data.execution_end_world_state = (
                        self.context.world.state._data.copy()
                    )


# %% reusable actions and authored recipes
@dataclass
class MoveInstrumentJointsAction(ActionDescription):
    """Move one instrument's mechanisms to a measured authored target group."""

    device: str
    """Installed instrument identifier."""
    targets: tuple[MedicalLabJointTarget, ...]
    """Desired mechanism coordinates and reference-speed limits."""
    label: str
    """English physical operation description."""

    @property
    def request(self) -> MedicalLabMotionRequest:
        """Capture action parameters without dispatching any physical operation."""
        return MedicalLabMotionRequest(self.device, self.targets, self.label)

    @property
    def _action_plan(self) -> PlanNode:
        """Expose one deferred operation to the ordinary sequential plan builder."""
        return sequential([MedicalLabMotionNode(request=self.request)])


@dataclass(frozen=True)
class MedicalLabPlan:
    """Inspect authored mechanisms and restore rest positions with modular actions."""

    description: MedicalLabDescription
    """Validated resource and joint metadata for this expanded world."""
    device: str = MedicalLabSelection.ALL
    """One instrument or the collection of all actuated instruments."""
    action: MedicalLabAction = MedicalLabAction.RUN
    """Complete inspection cycle, inspection endpoint, or rest endpoint."""

    def actions(self) -> list[MoveInstrumentJointsAction]:
        """Select authored joint targets without claiming sample processing results."""
        MedicalLabCommand(self.action, self.device)
        if self.action not in (
            MedicalLabAction.RUN,
            MedicalLabAction.OPEN,
            MedicalLabAction.CLOSE,
        ):
            raise InvalidMedicalLabCommand(
                "Only run, open and close produce mechanism plans"
            )
        selected = [
            instrument
            for instrument in self.description.instruments
            if self.device == MedicalLabSelection.ALL or instrument.id == self.device
        ]
        moving = [instrument for instrument in selected if instrument.joints]
        if not moving:
            raise InvalidMedicalLabCommand(
                "Select an installed instrument with moving joints"
            )
        actions = []
        for instrument in moving:
            if self.action != MedicalLabAction.CLOSE:
                actions.append(
                    MoveInstrumentJointsAction(
                        instrument.id,
                        tuple(
                            MedicalLabJointTarget(
                                joint.name, joint.inspection, joint.velocity
                            )
                            for joint in instrument.joints
                        ),
                        f"Inspect {instrument.label}",
                    )
                )
            if self.action != MedicalLabAction.OPEN:
                actions.append(
                    MoveInstrumentJointsAction(
                        instrument.id,
                        tuple(
                            MedicalLabJointTarget(
                                joint.name, joint.rest, joint.velocity
                            )
                            for joint in instrument.joints
                        ),
                        f"Return {instrument.label} to rest",
                    )
                )
        return actions

    def build(self, context: MedicalLabContext) -> Plan:
        """Construct an ordinary native sequential plan without moving devices."""
        return sequential(self.actions(), context=context).plan


# %% measured execution handshake
@dataclass
class MedicalLabExecution:
    """Serialize native plans and physically measured mechanism completions."""

    plan: Plan | None = field(default=None, init=False)
    """Native action tree and its independent semantic context."""
    worker: threading.Thread | None = field(default=None, init=False, repr=False)
    """Native plan worker; it never accesses MuJoCo state."""
    condition: threading.Condition = field(
        default_factory=threading.Condition, init=False, repr=False
    )
    """Coordination between the native worker and the physics owner."""
    active_request: MedicalLabMotionRequest | None = field(default=None, init=False)
    """Current physical request awaiting measured acknowledgment."""
    acknowledged: bool = field(default=False, init=False)
    """Whether the physical owner supplied the current action's result."""
    measurement: LaboratoryPlanWorld | None = field(default=None, init=False)
    """Detached physical observation applied only by the semantic-world owner."""
    completed: int = field(default=0, init=False)
    """Number of actions whose physical completion was accepted."""
    total: int = field(default=0, init=False)
    """Number of independent actions in this recipe."""
    state: MedicalLabState = field(default=MedicalLabState.IDLE, init=False)
    """Operator-visible lifecycle of the mechanism demonstration."""
    device: str | None = field(default=None, init=False)
    """Selected resource or all for the current recipe."""
    error: str | None = field(default=None, init=False)
    """Failure message from the physical operation or native execution boundary."""
    cancelled: bool = field(default=False, init=False)
    """Whether reset revoked this plan's physical execution ownership."""
    finished: bool = field(default=False, init=False)
    """Whether the native worker has terminated."""

    def start(
        self,
        recipe: MedicalLabPlan,
        world: World,
        robot: AbstractRobot,
        measurement: LaboratoryPlanWorld | None = None,
    ) -> None:
        """Start an independent native plan against an already constructed world."""
        if self.worker is not None and self.worker.is_alive():
            raise InvalidMedicalLabCommand("A mechanism plan is already active")
        actions = recipe.actions()
        self.plan = sequential(
            actions, context=MedicalLabContext(world=world, robot=robot, executor=self)
        ).plan
        self.active_request = None
        self.acknowledged = self.cancelled = self.finished = False
        self.error = None
        self.measurement = measurement
        self.completed = 0
        self.total = len(actions)
        self.device = recipe.device
        self.state = MedicalLabState.RUNNING
        self.worker = threading.Thread(target=self._perform, daemon=True)
        self.worker.start()

    def _perform(self) -> None:
        """Run the real native action tree in the standard simulated environment."""
        try:
            if self.measurement is not None:
                self.measurement.apply(self.plan.world)
            with replace(simulated_robot_advanced):
                self.plan.perform()
        except Exception as error:
            # Failures must reach the physics owner instead of stranding its request.
            with self.condition:
                if not self.cancelled:
                    self.error = self.error or str(error)
        finally:
            with self.condition:
                self.active_request = None
                self.finished = True
                if self.cancelled:
                    self.state = MedicalLabState.IDLE
                elif (
                    self.error is not None
                    or self.plan.root.status != TaskStatus.SUCCEEDED
                ):
                    self.error = (
                        self.error or "The native mechanism plan did not complete"
                    )
                    self.state = MedicalLabState.FAILED
                else:
                    self.state = MedicalLabState.COMPLETE
                self.condition.notify_all()

    def execute(self, request: MedicalLabMotionRequest) -> None:
        """Wait at a native action boundary until physics acknowledges its result."""
        with self.condition:
            if self.cancelled:
                raise MedicalLabMotionFailed("Mechanism execution was reset")
            self.active_request = request
            self.acknowledged = False
            self.condition.notify_all()
            self.condition.wait_for(
                lambda: self.cancelled
                or self.acknowledged
                and self.state != MedicalLabState.PAUSED
            )
            if self.acknowledged and self.measurement is not None:
                self.measurement.apply(self.plan.world)
            if self.cancelled:
                raise MedicalLabMotionFailed("Mechanism execution was reset")
            if self.error is not None:
                raise MedicalLabMotionFailed(self.error)
            self.active_request = None
            self.completed += 1

    def pending(self) -> MedicalLabMotionRequest | None:
        """Read the current request without blocking the thread stepping physics."""
        with self.condition:
            if (
                self.acknowledged
                or self.finished
                or self.state != MedicalLabState.RUNNING
            ):
                return None
            return self.active_request

    def wait_for_request(self, timeout: float = 3.0) -> MedicalLabMotionRequest | None:
        """Wait briefly for a native request outside the physics-stepping loop."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.finished
                or self.active_request is not None
                and not self.acknowledged,
                timeout=timeout,
            )
            return self.pending()

    def complete(
        self, error: str | None = None, measurement: LaboratoryPlanWorld | None = None
    ) -> None:
        """Acknowledge the current operation using a detached measured world state."""
        with self.condition:
            if self.active_request is None or self.acknowledged or self.finished:
                raise InvalidMedicalLabCommand(
                    "No pending mechanism operation can be acknowledged"
                )
            self.error = error
            self.measurement = measurement
            self.acknowledged = True
            self.condition.notify_all()

    def pause(self) -> None:
        """Pause the native plan while retaining its physical target and progress."""
        with self.condition:
            if self.state == MedicalLabState.RUNNING:
                self.state = MedicalLabState.PAUSED
                self.plan.root.pause()

    def resume(self) -> None:
        """Resume the same native operation without constructing a replacement plan."""
        with self.condition:
            if self.state == MedicalLabState.PAUSED:
                self.state = MedicalLabState.RUNNING
                self.plan.root.resume()
                self.condition.notify_all()

    def cancel(self) -> None:
        """Join the native worker before a reset changes physical mechanism targets."""
        with self.condition:
            self.cancelled = True
            if self.plan is not None and not self.finished:
                self.plan.root.interrupt()
            self.condition.notify_all()
        if self.worker is not None:
            self.worker.join(timeout=5.0)
            if self.worker.is_alive():
                raise RuntimeError(
                    "The native instrument plan did not stop after reset"
                )
        self.state = MedicalLabState.IDLE

    def snapshot(self) -> dict[str, Any]:
        """Expose measured action progress and the native execution framework."""
        with self.condition:
            return {
                MedicalLabPlanField.STATE: self.state,
                MedicalLabPlanField.DEVICE: self.device,
                MedicalLabPlanField.ACTION: (
                    self.active_request.label
                    if self.active_request is not None
                    else None
                ),
                MedicalLabPlanField.COMPLETED: self.completed,
                MedicalLabPlanField.TOTAL: self.total,
                MedicalLabPlanField.ERROR: self.error,
                MedicalLabPlanField.FRAMEWORK: MedicalLabPlanField.CORAPLEX,
                MedicalLabPlanField.NATIVE_STATUS: (
                    self.plan.root.status.name if self.plan is not None else None
                ),
            }
