"""Compose native laboratory actions and exchange measured execution results."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from enum import StrEnum

from typing_extensions import TYPE_CHECKING, Any

from coraplex.execution_environment import simulated_robot_advanced
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.base import ActionDescription
from cramera.laboratory_actions import (
    AssignRotorHolderAction,
    CloseInstrumentAction,
    ConfirmStorageAction,
    DispenseSampleAction,
    LaboratoryActionFailed,
    LaboratoryActionRequest,
    LaboratoryContext,
    LaboratoryGrasp,
    LaboratoryInstrument,
    OpenInstrumentAction,
    ReleaseSampleAction,
    RetrieveRackAction,
    RunCentrifugeAction,
    SampleDestination,
    SecureSampleAction,
    TransferSampleAction,
    VerifyRotorBalanceAction,
    VerifyWorkflowAction,
    WeighSampleAction,
)
from cramera.laboratory_pipetting import PipettingSource
from cramera.laboratory_plan_world import LaboratoryPlanWorld
from cramera.laboratory_world import LaboratoryWorld
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world_description.shape_collection import ShapeCollection

if TYPE_CHECKING:
    from cramera.laboratory_workflow import LaboratoryWorkflowProgram


# %% ordinary reusable recipe
@dataclass
class LaboratoryPlan:
    """Describe sample preparation as independently executable native actions."""

    sample_keys: tuple[str, ...]
    """Samples processed in rack access order."""
    dose_ml: float = 1.0
    """Requested dose from each reservoir into each sample."""
    rotor_rpm: float = 300.0
    """Requested rotor speed subject to the instrument's interlocks."""
    spin_seconds: float = 8.0
    """Measured cycle duration at operating speed."""

    def actions(self) -> list[ActionDescription]:
        """Compose independent actions without requiring a live simulator."""
        actions = [
            OpenInstrumentAction(LaboratoryInstrument.DRAWER),
            RetrieveRackAction(required_sample_keys=self.sample_keys),
            CloseInstrumentAction(LaboratoryInstrument.DRAWER),
        ]
        for index, key in enumerate(self.sample_keys):
            actions.extend(
                [
                    TransferSampleAction(
                        key, SampleDestination.DISPENSER, release_at_destination=False
                    ),
                    DispenseSampleAction(key, PipettingSource.AMBER, self.dose_ml),
                    DispenseSampleAction(key, PipettingSource.TEAL, self.dose_ml),
                    TransferSampleAction(key, SampleDestination.SCALE, from_held=True),
                    WeighSampleAction(key),
                    TransferSampleAction(key, SampleDestination.RACK),
                ]
            )
            if index == 0:
                actions.append(OpenInstrumentAction(LaboratoryInstrument.CENTRIFUGE))
            actions.extend(
                [
                    AssignRotorHolderAction(key),
                    TransferSampleAction(
                        key, SampleDestination.CENTRIFUGE, grasp=LaboratoryGrasp.TOP
                    ),
                    SecureSampleAction(key),
                ]
            )
        actions.extend(
            [
                VerifyRotorBalanceAction(required_sample_keys=self.sample_keys),
                CloseInstrumentAction(LaboratoryInstrument.CENTRIFUGE),
                RunCentrifugeAction(self.rotor_rpm, self.spin_seconds),
                OpenInstrumentAction(LaboratoryInstrument.CENTRIFUGE),
                OpenInstrumentAction(LaboratoryInstrument.REFRIGERATOR),
            ]
        )
        for key in self.sample_keys:
            actions.extend(
                [
                    ReleaseSampleAction(key),
                    TransferSampleAction(
                        key, SampleDestination.RACK, grasp=LaboratoryGrasp.TOP
                    ),
                    TransferSampleAction(key, SampleDestination.REFRIGERATOR),
                    ConfirmStorageAction(key),
                ]
            )
        actions.extend(
            [
                CloseInstrumentAction(LaboratoryInstrument.REFRIGERATOR),
                CloseInstrumentAction(LaboratoryInstrument.CENTRIFUGE),
                VerifyWorkflowAction(),
            ]
        )
        return actions

    def build(self, context: LaboratoryContext) -> Plan:
        """Build a native sequential plan without executing any physical operation."""
        return sequential(self.actions(), context=context).plan


# %% serialized physics boundary
class LaboratoryPlanField(StrEnum):
    """Native execution evidence published by the running laboratory."""

    FRAMEWORK = "framework"
    """Plan implementation owning action ordering."""
    CORAPLEX = "coraplex"
    """Native plan implementation identifier."""
    STATUS = "status"
    """Native root lifecycle state."""
    ACTION = "action"
    """Currently executing native action class."""
    COMPLETED = "completedActions"
    """Number of physical action boundaries acknowledged successfully."""


@dataclass
class LaboratoryPlanExecution:
    """Let a native plan wait while the simulation owner executes one action."""

    plan: Plan | None = field(default=None, init=False)
    """Native action tree and semantic robot context."""
    worker: threading.Thread | None = field(default=None, init=False, repr=False)
    """Plan worker that never reads or mutates MuJoCo dynamic data."""
    condition: threading.Condition = field(
        default_factory=threading.Condition, init=False, repr=False
    )
    """Coordinate action requests and measured results without the session lock."""
    active_request: LaboratoryActionRequest | None = field(default=None, init=False)
    """Immutable request currently owned by the physical simulation thread."""
    completed: int = field(default=0, init=False)
    """Count of successfully measured physical operations."""
    finished: bool = field(default=False, init=False)
    """Whether native plan execution has terminated."""
    cancelled: bool = field(default=False, init=False)
    """Whether the session revoked physical execution ownership."""
    error: str | None = field(default=None, init=False)
    """Failure propagated from a physical action or native plan."""
    acknowledged: bool = field(default=False, init=False)
    """Whether the pending physical operation has reported its outcome."""
    measurement: LaboratoryPlanWorld | None = field(default=None, init=False)
    """Detached physical state applied to the semantic context at action boundaries."""

    def start(self, program: LaboratoryWorkflowProgram) -> None:
        """Build the semantic context and start its native plan before any motion."""
        world = LaboratoryWorld(
            bundle_directory=program.physics.bundle_directory,
            allow_support_contacts=False,
        ).build()
        for body in world.bodies:
            body.visual = ShapeCollection()
        context = LaboratoryContext(
            world=world,
            robot=world.get_semantic_annotations_by_type(PR2)[0],
            executor=self,
        )
        self.plan = sequential(program.make_actions(), context=context).plan
        self.measurement = LaboratoryPlanWorld.capture(program.physics)
        self.worker = threading.Thread(target=self._perform, daemon=True)
        self.worker.start()
        self.wait_for_request()

    def _perform(self) -> None:
        """Execute the actual native tree with the advanced simulated environment."""
        try:
            self.measurement.apply(self.plan.world)
            with replace(simulated_robot_advanced):
                self.plan.perform()
        except Exception as error:
            # The thread boundary must publish a failure instead of stranding physics.
            with self.condition:
                self.error = str(error)
        finally:
            with self.condition:
                self.active_request = None
                self.finished = True
                self.condition.notify_all()

    def execute(self, request: LaboratoryActionRequest) -> None:
        """Wait for this request's measured outcome without touching the simulator."""
        with self.condition:
            if self.cancelled:
                raise LaboratoryActionFailed("Laboratory execution was reset")
            self.active_request = request
            self.acknowledged = False
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.acknowledged or self.cancelled)
            if self.acknowledged and self.measurement is not None:
                self.measurement.apply(self.plan.world)
            if self.cancelled:
                raise LaboratoryActionFailed(
                    self.error or "Laboratory execution was reset"
                )
            if self.error is not None:
                raise LaboratoryActionFailed(self.error)
            self.active_request = None
            self.completed += 1

    def wait_for_request(self) -> LaboratoryActionRequest | None:
        """Wait for the next native leaf without acquiring the simulation lock."""
        with self.condition:
            ready = self.condition.wait_for(
                lambda: (self.active_request is not None and not self.acknowledged)
                or self.finished,
                timeout=30.0,
            )
            if not ready:
                self.cancelled = True
                self.error = "The native laboratory plan stopped responding"
                self.condition.notify_all()
            return self.active_request if ready and not self.finished else None

    def complete(
        self, error: str | None = None, measurement: LaboratoryPlanWorld | None = None
    ) -> None:
        """Acknowledge one measured result to the waiting native action."""
        with self.condition:
            self.error = error
            if measurement is not None:
                self.measurement = measurement
            self.acknowledged = True
            self.condition.notify_all()

    def pause(self) -> None:
        """Expose the physical pause in the owning native plan tree."""
        if self.plan is not None:
            self.plan.root.pause()

    def resume(self) -> None:
        """Resume the same pending action in the native tree."""
        if self.plan is not None:
            self.plan.root.resume()

    def cancel(self) -> None:
        """Join the plan before a reset may replace its physical state."""
        with self.condition:
            self.cancelled = True
            if self.plan is not None and not self.finished:
                self.plan.root.interrupt()
            self.condition.notify_all()
        if self.worker is not None:
            self.worker.join(timeout=30.0)
            if self.worker.is_alive():
                raise RuntimeError(
                    "The laboratory plan did not stop after cancellation"
                )

    def snapshot(self) -> dict[str, Any]:
        """Expose the native lifecycle alongside physical measurement evidence."""
        with self.condition:
            return {
                LaboratoryPlanField.FRAMEWORK: LaboratoryPlanField.CORAPLEX,
                LaboratoryPlanField.STATUS: self.plan.root.status.name,
                LaboratoryPlanField.ACTION: (
                    self.active_request.operation
                    if self.active_request is not None
                    else None
                ),
                LaboratoryPlanField.COMPLETED: self.completed,
            }
