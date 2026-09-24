"""Compose independent contact-physics laboratory operations as native plan actions."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import ClassVar, Protocol

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType, TaskStatus
from coraplex.plans.executables import Executable, GiskardExecutable
from coraplex.plans.factories import sequential
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan_node import ActionNode, ExecutionBoundaryNode, PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from cramera.laboratory_pipetting_command import PipettingSource


# %% operation vocabulary
class LaboratoryOperation(StrEnum):
    """Independent physical operations accepted by the laboratory executor."""

    TRANSFER = "transfer"
    """Move a sample using measured gripper contacts."""
    DISPENSE = "dispense"
    """Deliver one measured reagent dose."""
    WEIGH = "weigh"
    """Record a supported sample's stable scale measurement."""
    ASSIGN_ROTOR_HOLDER = "assign_rotor_holder"
    """Choose a suitable rotor holder for the sample."""
    VERIFY_ROTOR_BALANCE = "verify_rotor_balance"
    """Check the measured opposed rotor loads."""
    SECURE = "secure"
    """Secure a physically seated sample in its rotor holder."""
    RELEASE = "release"
    """Release a stopped rotor holder for retrieval."""
    RUN_CENTRIFUGE = "run_centrifuge"
    """Complete a timed rotor cycle and its braking phase."""
    OPEN_INSTRUMENT = "open_instrument"
    """Manipulate a closed instrument's handle to open it."""
    CLOSE_INSTRUMENT = "close_instrument"
    """Manipulate an open instrument's handle to close it."""
    CONFIRM_STORAGE = "confirm_storage"
    """Check that a sample remains supported in its refrigerator slot."""
    VERIFY_WORKFLOW = "verify_workflow"
    """Check the final processing evidence and closed instruments."""


class SampleDestination(StrEnum):
    """Physical sample supports and the held dosing position."""

    DISPENSER = "dispenser"
    """Aligned receiver position beneath the dosing nozzles."""
    SCALE = "scale"
    """The instrument's weighing pan."""
    RACK = "rack"
    """A supported rack position available for regrasping."""
    CENTRIFUGE = "centrifuge"
    """The sample's assigned rotor holder."""
    REFRIGERATOR = "refrigerator"
    """The sample's assigned refrigerated storage slot."""


class LaboratoryGrasp(StrEnum):
    """Tool approach orientations for a physical sample transfer."""

    FRONT = "front"
    """A horizontal side grasp of the vessel wall."""
    TOP = "top"
    """An inclined upper-wall grasp for access to the rotor."""


class LaboratoryInstrument(StrEnum):
    """Articulated instruments that the robot can open or close."""

    CENTRIFUGE = "centrifuge"
    """The rotor's protective lid."""
    REFRIGERATOR = "refrigerator"
    """The sample cooler's glass door."""


@dataclass(frozen=True)
class LaboratoryActionRequest:
    """Immutable parameters of one operation dispatched at native execution time."""

    operation: LaboratoryOperation
    """The individual physical operation to execute."""
    object_key: str | None = None
    """Authored sample identifier when the operation acts on a vessel."""
    destination: SampleDestination | None = None
    """Support or dosing position requested by a sample transfer."""
    source: PipettingSource | None = None
    """Reagent channel selected for one dose."""
    volume_ml: float | None = None
    """Requested reagent quantity in millilitres."""
    rpm: float | None = None
    """Requested centrifuge speed in revolutions per minute."""
    duration: float | None = None
    """Requested centrifuge operating time in simulation seconds."""
    instrument: LaboratoryInstrument | None = None
    """Articulated instrument selected for opening or closing."""
    grasp: LaboratoryGrasp = LaboratoryGrasp.FRONT
    """Approach orientation used for physical sample contact."""
    from_held: bool = False
    """Whether the transfer starts with an already grasped vessel."""
    release_at_destination: bool = True
    """Whether the vessel is released onto its destination support."""
    required_sample_keys: tuple[str, ...] = ()
    """Optional exact rotor batch whose weighing and holder evidence is required."""


@dataclass
class LaboratoryActionFailed(PlanFailure):
    """A laboratory operation failed its measured physical acceptance condition."""

    message: str
    """Explanation retained in the native failed action node."""

    def error_message(self) -> str:
        """Describe the physical condition that prevented completion."""
        return self.message


# %% native execution boundary
class LaboratoryActionExecutor(Protocol):
    """Execute one request and return only after its measured physical completion."""

    def execute(self, request: LaboratoryActionRequest) -> None:
        """Submit one operation or raise a :class:`LaboratoryActionFailed`."""


@dataclass(eq=False)
class LaboratoryContext(Context):
    """Native robot context with a separately owned physical operation executor."""

    executor: LaboratoryActionExecutor = field(kw_only=True)
    """Serialized physical operation runner, separate from the semantic world."""


@dataclass(eq=False, repr=False)
class LaboratoryOperationNode(ExecutionBoundaryNode):
    """Keep one physical operation between its preceding and following actions."""

    request: LaboratoryActionRequest = field(kw_only=True)
    """Operation parameters that are not issued during plan expansion."""

    def notify(self) -> None:
        """Leave physical state unchanged when the plan expands its actions."""

    def parse(self) -> LaboratoryOperationExecutable:
        """Create a deferred native executable for this individual operation."""
        return LaboratoryOperationExecutable(node=self, context=self.plan.context)


@dataclass
class LaboratoryOperationExecutable(Executable):
    """Dispatch a physical request while preserving native plan lifecycle events."""

    node: LaboratoryOperationNode = field(kw_only=True)
    """Native leaf whose execution records the physical result."""
    PAUSE_INTERVAL: ClassVar[float] = 0.01
    """Wall-time interval for observing a paused plan before dispatch."""

    def execute(self) -> None:
        """Wait for native resume and dispatch only when simulated execution permits."""
        with self.node.execution_scope():
            while self.node.is_paused and not self.node.is_interrupted:
                time.sleep(self.PAUSE_INTERVAL)
            if self.node.is_interrupted:
                self.node.status = TaskStatus.INTERRUPTED
                return
            if GiskardExecutable.execution_type == ExecutionType.NO_EXECUTION:
                return
            if GiskardExecutable.execution_type != ExecutionType.SIMULATED:
                raise LaboratoryActionFailed(
                    "Laboratory contact actions require simulated robot execution"
                )
            if not isinstance(self.context, LaboratoryContext):
                raise LaboratoryActionFailed(
                    "Laboratory contact actions require a laboratory context"
                )
            action_node = self._record_start()
            try:
                self.context.executor.execute(self.node.request)
            finally:
                if action_node is not None:
                    action_node.update_execution_data_post_perform()
                    action_node.execution_data.execution_end_world_state = (
                        self.context.world.state._data.copy()
                    )

    def _record_start(self) -> ActionNode | None:
        """Keep the enclosing action's initial measured state independent of later updates."""
        action_node = next(
            (node for node in self.node.path if isinstance(node, ActionNode)), None
        )
        if action_node is not None:
            action_node.create_execution_data_pre_perform()
            action_node.execution_data.execution_start_world_state = (
                self.context.world.state._data.copy()
            )
        return action_node


# %% independently composable actions
@dataclass
class LaboratoryAction(ActionDescription, ABC):
    """Expand a parameterized physical operation into a native execution boundary."""

    @property
    @abstractmethod
    def operation(self) -> LaboratoryOperation:
        """Identify the independently executable physical operation."""

    @property
    def request(self) -> LaboratoryActionRequest:
        """Capture the action's declared parameters as an immutable command."""
        return LaboratoryActionRequest(self.operation, **self.designator_parameter)

    @property
    def _action_plan(self) -> PlanNode:
        """Keep physical dispatch deferred until native sequential execution."""
        return sequential([LaboratoryOperationNode(request=self.request)])


@dataclass
class TransferSampleAction(LaboratoryAction):
    """Carry one vessel to a selected station with an explicit grasp policy."""

    object_key: str
    """The vessel manipulated by the robot."""
    destination: SampleDestination
    """The selected physical destination."""
    grasp: LaboratoryGrasp = LaboratoryGrasp.FRONT
    """Approach orientation for the grasp."""
    from_held: bool = False
    """Whether the gripper already holds this vessel."""
    release_at_destination: bool = True
    """Whether to open the gripper after supported placement."""
    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.TRANSFER
    """Physical sample transfer dispatch."""


@dataclass
class DispenseSampleAction(LaboratoryAction):
    """Deliver a bounded quantity from one reagent channel into a sample."""

    object_key: str
    """The receiving vessel beneath the nozzles."""
    source: PipettingSource
    """The selected independent reagent reservoir."""
    volume_ml: float
    """Requested volume in millilitres."""
    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.DISPENSE
    """Measured liquid delivery dispatch."""


@dataclass
class WeighSampleAction(LaboratoryAction):
    """Wait for and record a vessel's stable supported scale reading."""

    object_key: str
    """Vessel whose measured gross mass is recorded."""
    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.WEIGH
    """Stable supported weighing dispatch."""


@dataclass
class AssignRotorHolderAction(LaboratoryAction):
    """Choose the sample's rotor holder from the measured load inventory."""

    object_key: str
    """Vessel assigned to a compatible available holder."""
    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.ASSIGN_ROTOR_HOLDER
    """Measured holder assignment dispatch."""


@dataclass
class VerifyRotorBalanceAction(LaboratoryAction):
    """Verify that the secured rotor load satisfies its balance constraints."""

    required_sample_keys: tuple[str, ...] = ()
    """Optional exact batch whose prior weighing and holder assignment must be verified."""
    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.VERIFY_ROTOR_BALANCE
    """Opposed load validation dispatch."""


@dataclass
class SecureSampleAction(LaboratoryAction):
    """Secure a sample after its physical seating has been confirmed."""

    object_key: str
    """Vessel already placed in its assigned rotor holder."""
    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.SECURE
    """Contact-verified holder securing dispatch."""


@dataclass
class ReleaseSampleAction(LaboratoryAction):
    """Release a holder while the rotor is stopped and its lid is open."""

    object_key: str
    """Vessel whose holder is released before retrieval."""
    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.RELEASE
    """Stopped rotor holder release dispatch."""


@dataclass
class RunCentrifugeAction(LaboratoryAction):
    """Complete a balanced timed centrifuge cycle including its braking phase."""

    rpm: float
    """Requested rotor speed in revolutions per minute."""
    duration: float
    """Requested operating interval in simulation seconds."""
    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.RUN_CENTRIFUGE
    """Timed instrument cycle dispatch."""


@dataclass
class OpenInstrumentAction(LaboratoryAction):
    """Open one instrument through its articulated handle manipulation."""

    instrument: LaboratoryInstrument
    """Instrument selected for opening."""
    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.OPEN_INSTRUMENT
    """Articulated opening dispatch."""


@dataclass
class CloseInstrumentAction(LaboratoryAction):
    """Close one instrument through its articulated handle manipulation."""

    instrument: LaboratoryInstrument
    """Instrument selected for closing."""
    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.CLOSE_INSTRUMENT
    """Articulated closing dispatch."""


@dataclass
class ConfirmStorageAction(LaboratoryAction):
    """Confirm that a processed vessel rests in its assigned refrigerated slot."""

    object_key: str
    """Vessel whose supported storage is confirmed."""
    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.CONFIRM_STORAGE
    """Measured refrigerator occupancy dispatch."""


@dataclass
class VerifyWorkflowAction(LaboratoryAction):
    """Validate the final sample inventory, processing evidence and closures."""

    operation: ClassVar[LaboratoryOperation] = LaboratoryOperation.VERIFY_WORKFLOW
    """Final physical acceptance dispatch."""
