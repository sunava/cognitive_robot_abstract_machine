"""Coordinate measured dosing, weighing, centrifugation and sample storage."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import permutations

import mujoco
import numpy as np
from typing_extensions import Any, ClassVar, TYPE_CHECKING

from cramera.laboratory_centrifuge import CentrifugeGeometry
from cramera.laboratory_centrifuge_command import CentrifugeAction
from cramera.laboratory_mixing_program import LaboratoryMixingProgram, MixingField
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import ProgramField, ProgramState
from cramera.laboratory_pipetting import (
    PipettingAction,
    PipettingSource,
    PipettingState,
)
from cramera.laboratory_refrigerator import RefrigeratorGeometry
from cramera.laboratory_refrigerator_command import (
    RefrigeratorAction,
    RefrigeratorField,
)
from cramera.laboratory_world import LaboratoryBody, LaboratoryField
from cramera.laboratory_workflow_command import (
    WorkflowField as WorkflowCommandField,
    WorkflowSource,
)

if TYPE_CHECKING:
    from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
    from cramera.laboratory_sample_transfer import LaboratorySampleTransfer


# %% published workflow vocabulary
class WorkflowField(StrEnum):
    """Fields of the observable multi-instrument recipe."""

    SOURCE = WorkflowSource.PROGRAM
    """Controller identity shared with the workflow request boundary."""
    WORKFLOW = WorkflowCommandField.WORKFLOW
    """Nested per-run instrument and sample evidence."""
    STAGE = "stage"
    """Current user-visible recipe category."""
    STAGE_LABEL = "stageLabel"
    """English explanation of the active finite operation."""
    SAMPLE_KEY = "sampleKey"
    """Current vessel identifier, when an operation has a sample."""
    SAMPLES = "samples"
    """All sample measurements belonging to this execution."""
    KEY = "key"
    """Stable authored vessel identifier."""
    LABEL = "label"
    """Readable vessel name."""
    DOSE_A = "doseAMl"
    """Confirmed received volume from reagent channel A."""
    DOSE_B = "doseBMl"
    """Confirmed received volume from reagent channel B."""
    WEIGHT = "weightGrams"
    """Stable gross contact load in grams."""
    LOCATION = "location"
    """Latest confirmed support or active robot transport."""
    CENTRIFUGE_SLOT = "centrifugeSlot"
    """Assigned mass-matched rotor pocket."""
    RACK_SLOT = "rackSlot"
    """Measured rack support used for a change of grasp direction."""
    REFRIGERATOR_SLOT = "refrigeratorSlot"
    """Assigned sample cooler support."""
    COMPLETED = "completedStages"
    """Finite operation labels accepted so far."""
    FAILURE = "failure"
    """Unmet physical acceptance condition."""
    SCALE = "scale"
    """Authored balance metadata."""
    PAN_POSITION = "panPosition"
    """World coordinates of the weighing support."""


class WorkflowStage(StrEnum):
    """User-visible categories of the sample preparation recipe."""

    FILL = "fill"
    """Move and meter both reagents into each sample."""
    WEIGH = "weigh"
    """Record the released supported sample mass."""
    LOAD = "load_centrifuge"
    """Arrange measured matched pairs in rotor pockets."""
    SPIN = "spin"
    """Complete a timed cycle under the instrument interlocks."""
    STORE = "store"
    """Place processed samples behind the cooler door."""
    COMPLETE = "complete"
    """Every intended sample reached supported closed-door storage."""


class WorkflowLocation(StrEnum):
    """Confirmed support or active transport of a sample."""

    RACK = "rack"
    """Original authored rack support."""
    ROBOT = "robot"
    """Physical robot transfer in progress."""
    DISPENSER = "dispenser"
    """Sample aligned under the reagent outlet."""
    SCALE = "scale"
    """Balance pan supporting a released sample."""
    REGRASP = "regrasp"
    """Sample released at the named front rack support for a grasp change."""
    CENTRIFUGE = "centrifuge"
    """Rotor holder carrying a sample."""
    REFRIGERATOR = "refrigerator"
    """Cooler shelf carrying a sample."""


class WorkflowStep(StrEnum):
    """Finite operations accepted from instrument and contact evidence."""

    DISPENSER = "Move sample to dispenser"
    """Sample aligned under the reagent outlet."""
    DOSE_A = "Dispense 1 ml of reagent A"
    """Confirmed received volume from reagent channel A."""
    DOSE_B = "Dispense 1 ml of reagent B"
    """Confirmed received volume from reagent channel B."""
    SCALE = "Place sample on balance"
    """Release the sample on the contact-sensitive weighing pan."""
    WEIGH = "Wait for stable measured weight"
    """Record the released supported sample mass."""
    REGRASP = "Place weighed sample in A1 for top regrasp"
    """Release a weighed sample on the reachable front rack support."""
    ASSIGN = "Select a measured matching rotor holder"
    """Choose an opposing sample using actual weighing evidence."""
    BALANCE = "Verify four measured balanced samples"
    """Verify measured weights against the occupied opposing holders."""
    OPEN_CENTRIFUGE = "Open centrifuge lid"
    """Request and observe an open instrument cover."""
    LOAD = "Place sample in centrifuge holder"
    """Arrange measured matched pairs in rotor pockets."""
    SECURE = "Secure seated sample"
    """Engage a clamp only on a measured seated sample."""
    CLOSE_CENTRIFUGE = "Close centrifuge lid"
    """Request and observe a closed instrument cover."""
    SPIN = "Run timed centrifuge cycle"
    """Complete a timed cycle under the instrument interlocks."""
    OPEN_REFRIGERATOR = "Open sample cooler door"
    """Request and observe an open cabinet aperture."""
    RELEASE = "Release centrifuge holder"
    """Disengage a stationary rotor fixture before pickup."""
    UNLOAD = "Place processed sample in A1 for front regrasp"
    """Withdraw from the rotor and release on the front rack support."""
    STORE = "Place sample in cooler"
    """Place processed samples behind the cooler door."""
    CONFIRM_STORAGE = "Confirm supported sample in cooler"
    """Require measured upright supported cooler occupancy."""
    CLOSE_REFRIGERATOR = "Close sample cooler door"
    """Request and observe closed refrigerated storage."""


@dataclass(frozen=True)
class WorkflowParameters:
    """Recipe quantities and bounded waits in simulation time."""

    dose_ml: float = 1.0
    """Volume dispensed from each channel into each initially empty tube."""
    rotor_rpm: float = 300.0
    """Requested demonstration rotor speed, limited by the instrument."""
    spin_seconds: float = 8.0
    """Required time in the measured operating speed band."""
    operation_timeout: float = 45.0
    """Maximum wait for a single instrument operation to become ready."""
    transfer_timeout: float = 180.0
    """Maximum duration of a robot pickup and supported placement."""
    volume_tolerance_ml: float = 0.000001
    """Allowed numerical discrepancy in conserved dose accounting."""
    timer_tolerance_seconds: float = 0.000000001
    """Numerical allowance when comparing one physical timer decrement."""
    position_tolerance: float = 0.008
    """Required initial rack alignment of the four upright vessels."""
    upright_tolerance: float = 0.2
    """Maximum initial tube-axis tilt from vertical in radians."""
    mass_tolerance_grams: float = 0.05
    """Maximum difference between stable pan load and the complete sample mass."""


@dataclass
class WorkflowSample:
    """Recorded instrument evidence belonging to a single vessel."""

    key: str
    """Stable body and liquid-ledger identifier."""
    dose_a_ml: float | None = None
    """Measured reagent A received during this run."""
    dose_b_ml: float | None = None
    """Measured reagent B received during this run."""
    weight_grams: float | None = None
    """Stable released pan load measured for this sample."""
    location: WorkflowLocation = WorkflowLocation.RACK
    """Latest confirmed support or active robot transport."""
    centrifuge_slot: str | None = None
    """Assigned holder in a pair chosen from observed weights."""
    refrigerator_slot: str | None = None
    """Assigned supported storage position."""
    rack_slot: str | None = None
    """Confirmed rack support while the sample awaits a change of grasp."""

    def snapshot(self) -> dict[str, Any]:
        """Publish only confirmed doses and weighing results."""
        return {
            WorkflowField.KEY: self.key,
            WorkflowField.LABEL: self.key.replace("tube_", "")
            .replace("_", " ")
            .title(),
            WorkflowField.DOSE_A: self.dose_a_ml,
            WorkflowField.DOSE_B: self.dose_b_ml,
            WorkflowField.WEIGHT: self.weight_grams,
            WorkflowField.LOCATION: self.location,
            WorkflowField.CENTRIFUGE_SLOT: self.centrifuge_slot,
            WorkflowField.REFRIGERATOR_SLOT: self.refrigerator_slot,
            WorkflowField.RACK_SLOT: self.rack_slot,
        }


def _measured_pair_order(
    samples: list[WorkflowSample], tolerance: float
) -> list[WorkflowSample] | None:
    """Find an assignment satisfying both opposing mass limits, if fully observed."""
    if len(samples) != len(LaboratoryWorkflowProgram.SAMPLE_KEYS) or any(
        sample.weight_grams is None or not math.isfinite(sample.weight_grams)
        for sample in samples
    ):
        return None
    return next(
        (
            list(ordered)
            for ordered in permutations(samples)
            if all(
                abs(ordered[index].weight_grams - ordered[index + 1].weight_grams)
                <= tolerance
                for index in range(0, len(ordered), 2)
            )
        ),
        None,
    )


def balanced_sample_order(
    samples: list[WorkflowSample], tolerance: float
) -> list[WorkflowSample]:
    """Order four weighed samples as two mass-matched opposing pairs."""
    ordered = _measured_pair_order(samples, tolerance)
    if ordered is None:
        raise InvalidPhysicsTarget(
            "The measured sample masses cannot form two balanced opposing pairs"
        )
    return ordered


@dataclass(frozen=True)
class WorkflowIntention:
    """One instrument operation or robot transfer in a fixed sample recipe."""

    operation: WorkflowStep
    """Operation whose measured acceptance condition advances the recipe."""
    stage: WorkflowStage
    """Visible category used to explain recipe progress."""
    sample_index: int | None = None
    """Sample record acted on by this intention, when applicable."""


# %% evidence-driven sequence
@dataclass
class LaboratoryWorkflowProgram:
    """Run four empty samples through physical instruments and contact transfers."""

    physics: LaboratoryRobotPhysics
    """Authoritative robot, contacts and instrument controllers."""
    parameters: WorkflowParameters = field(default_factory=WorkflowParameters)
    """Fixed recipe and measured readiness tolerances."""
    state: ProgramState = field(init=False, default=ProgramState.IDLE)
    """Execution lifecycle separate from the ongoing physics simulation."""
    samples: list[WorkflowSample] = field(init=False, default_factory=list)
    """Per-sample measurements retained across pauses."""
    steps: list[WorkflowIntention] = field(init=False, default_factory=list)
    """Finite sequence of manipulation and instrument intentions."""
    step_index: int = field(init=False, default=0)
    """Current intention in the recipe."""
    step_time: float = field(init=False, default=0.0)
    """Active simulation time spent on the current operation."""
    elapsed: float = field(init=False, default=0.0)
    """Active simulation time spent in this recipe."""
    issued: bool = field(init=False, default=False)
    """Whether the current instrument command has already been issued."""
    motion: LaboratorySampleTransfer | None = field(init=False, default=None)
    """Contact-verified robot transfer currently responsible for joint targets."""
    error: str | None = field(init=False, default=None)
    """Physical or inventory condition that stopped the recipe."""
    dose_before: float = field(init=False, default=0.0)
    """Receiver quantity immediately before the current pump request."""
    spin_observed: bool = field(init=False, default=False)
    """Whether the timed cycle entered its measured operating speed band."""
    spin_remaining: float = field(init=False, default=0.0)
    """Previous instrument timer value used to detect interrupted countdowns."""
    spin_timer_complete: bool = field(init=False, default=False)
    """Whether every final timer decrement was observed without an interlock abort."""
    spin_finished: bool = field(init=False, default=False)
    """Whether the observed timed cycle finished with the rotor fully stopped."""
    load_verified: bool = field(init=False, default=False)
    """Whether all four weighed samples were confirmed secured and balanced."""

    SAMPLE_KEYS: ClassVar[tuple[LaboratoryBody, ...]] = (
        LaboratoryBody.CLEAR_TUBE,
        LaboratoryBody.AMBER_TUBE,
        LaboratoryBody.TEAL_TUBE,
        LaboratoryBody.RESERVE_TUBE,
    )
    """Four authored loose vessels included in the demonstration."""
    REGRASP_SLOT: ClassVar[str] = "A1"
    """Front rack support reached by both horizontal and upper-wall grasps."""

    def __post_init__(self) -> None:
        """Build the recipe without changing object poses or liquid quantities."""
        self.reset()

    def reset(self) -> None:
        """Clear recipe evidence while leaving physical reset to the session owner."""
        self.state = ProgramState.IDLE
        self.samples = [WorkflowSample(key) for key in self.SAMPLE_KEYS]
        self.steps = self._recipe()
        self.step_index = 0
        self.step_time = self.elapsed = self.dose_before = 0.0
        self.issued = self.spin_observed = self.spin_finished = False
        self.spin_remaining = 0.0
        self.spin_timer_complete = False
        self.motion = None
        self.error = None
        self.load_verified = False

    def _recipe(self) -> list[WorkflowIntention]:
        """Vacate front rack samples before approaching rear-row samples."""
        steps = []
        for index in range(len(self.samples)):
            for operation, stage in (
                (WorkflowStep.DISPENSER, WorkflowStage.FILL),
                (WorkflowStep.DOSE_A, WorkflowStage.FILL),
                (WorkflowStep.DOSE_B, WorkflowStage.FILL),
                (WorkflowStep.SCALE, WorkflowStage.WEIGH),
                (WorkflowStep.WEIGH, WorkflowStage.WEIGH),
                (WorkflowStep.REGRASP, WorkflowStage.LOAD),
            ):
                steps.append(WorkflowIntention(operation, stage, index))
            if index == 0:
                steps.append(
                    WorkflowIntention(WorkflowStep.OPEN_CENTRIFUGE, WorkflowStage.LOAD)
                )
            for operation in (
                WorkflowStep.ASSIGN,
                WorkflowStep.LOAD,
                WorkflowStep.SECURE,
            ):
                steps.append(WorkflowIntention(operation, WorkflowStage.LOAD, index))
        steps.extend(
            (
                WorkflowIntention(WorkflowStep.BALANCE, WorkflowStage.LOAD),
                WorkflowIntention(WorkflowStep.CLOSE_CENTRIFUGE, WorkflowStage.SPIN),
                WorkflowIntention(WorkflowStep.SPIN, WorkflowStage.SPIN),
                WorkflowIntention(WorkflowStep.OPEN_CENTRIFUGE, WorkflowStage.STORE),
                WorkflowIntention(WorkflowStep.OPEN_REFRIGERATOR, WorkflowStage.STORE),
            )
        )
        for index in range(len(self.samples)):
            for operation in (
                WorkflowStep.RELEASE,
                WorkflowStep.UNLOAD,
                WorkflowStep.STORE,
                WorkflowStep.CONFIRM_STORAGE,
            ):
                steps.append(WorkflowIntention(operation, WorkflowStage.STORE, index))
        steps.extend(
            (
                WorkflowIntention(WorkflowStep.CLOSE_REFRIGERATOR, WorkflowStage.STORE),
                WorkflowIntention(WorkflowStep.CLOSE_CENTRIFUGE, WorkflowStage.STORE),
            )
        )
        return steps

    @property
    def step(self) -> WorkflowIntention:
        """Current operation, retaining the final intention after completion."""
        return self.steps[min(self.step_index, len(self.steps) - 1)]

    @property
    def sample(self) -> WorkflowSample | None:
        """Sample associated with the current operation."""
        return (
            None
            if self.step.sample_index is None
            else self.samples[self.step.sample_index]
        )

    def _rack_position(self, key: str) -> np.ndarray:
        """Resolve an authored sample's original rack support."""
        laboratory = self.physics.scene[LaboratoryField.LABORATORY]
        slot = next(
            item[LaboratoryField.SLOT]
            for item in laboratory[LaboratoryField.TUBES]
            if item[LaboratoryField.KEY] == key
        )
        return np.asarray(
            next(
                item[LaboratoryField.POSE][:3]
                for item in laboratory[MixingField.RACK_SLOTS]
                if item[LaboratoryField.ID] == slot
            )
        )

    def _regrasp_position(self) -> np.ndarray:
        """Resolve the shared authored rack support for changes of grasp."""
        return np.asarray(
            next(
                slot[LaboratoryField.POSE][:3]
                for slot in self.physics.scene[LaboratoryField.LABORATORY][
                    MixingField.RACK_SLOTS
                ]
                if slot[LaboratoryField.ID] == self.REGRASP_SLOT
            )
        )

    def _validate_start(self) -> None:
        """Require explicitly prepared empty tubes and available empty instruments."""
        instruments = (
            self.physics.pipetting,
            self.physics.scale,
            self.physics.centrifuge,
            self.physics.refrigerator,
        )
        if not all(instrument.available for instrument in instruments):
            raise InvalidPhysicsTarget(
                "The full demo needs dispenser, balance, centrifuge and cooler"
            )
        if self.physics.pipetting.state == PipettingState.DISPENSING:
            raise InvalidPhysicsTarget(
                "Stop the dispenser before preparing a full workflow"
            )
        centrifuge = self.physics.centrifuge
        if not centrifuge.stopped or any(
            slot.key is not None for slot in centrifuge.slots.values()
        ):
            raise InvalidPhysicsTarget(
                "The full workflow needs an empty stopped centrifuge"
            )
        if centrifuge.knowledge.active.minimum_samples > len(self.samples):
            raise InvalidPhysicsTarget(
                "The active centrifuge manual requires more samples than this recipe"
            )
        if self.physics.scale.objects or any(
            self.physics.refrigerator.occupants().values()
        ):
            raise InvalidPhysicsTarget(
                "Clear the balance and cooler before preparing the full workflow"
            )
        if len(self.physics.refrigerator.metadata[RefrigeratorField.SLOTS]) < len(
            self.samples
        ):
            raise InvalidPhysicsTarget(
                "The cooler needs four available sample positions"
            )
        if (
            self.parameters.dose_ml * len(PipettingSource)
            > self.physics.liquid.interior.capacity_ml
        ):
            raise InvalidPhysicsTarget(
                "The sample capacity is too small for both reagent doses"
            )
        for source in PipettingSource:
            if (
                self.physics.pipetting.reservoirs[source].volume_ml
                < len(self.samples) * self.parameters.dose_ml
            ):
                raise InvalidPhysicsTarget(
                    "Refill both reagent reservoirs before the four-sample workflow"
                )
        for sample in self.samples:
            if (
                sample.key not in self.physics.objects
                or sample.key not in self.physics.liquid.tubes
            ):
                raise InvalidPhysicsTarget(
                    "The full workflow needs all four authored samples"
                )
            body = self.physics.objects[sample.key].body_id
            if (
                self.physics.liquid.tubes[sample.key].volume_ml
                > self.parameters.volume_tolerance_ml
            ):
                raise InvalidPhysicsTarget(
                    "Prepare four empty samples before starting this recipe"
                )
            if np.linalg.norm(
                self.physics.data.xpos[body] - self._rack_position(sample.key)
            ) > self.parameters.position_tolerance or self.physics.data.xmat[
                body
            ].reshape(
                3, 3
            )[
                2, 2
            ] < math.cos(
                self.parameters.upright_tolerance
            ):
                raise InvalidPhysicsTarget(
                    "Return all four samples upright to their authored rack positions"
                )

    def start(self) -> None:
        """Start a prepared recipe or resume the same measured physical operation."""
        if self.state not in (ProgramState.IDLE, ProgramState.PAUSED):
            return
        if self.state == ProgramState.IDLE:
            self._validate_start()
            self.physics.release()
        elif self.issued and self.step.operation in (
            WorkflowStep.DOSE_A,
            WorkflowStep.DOSE_B,
        ):
            station = self.physics.pipetting
            if (
                station.key != self.sample.key
                or station.source != self._dose_source()
                or abs(station.requested_ml - self.parameters.dose_ml)
                > self.parameters.volume_tolerance_ml
            ):
                raise InvalidPhysicsTarget(
                    "The paused dose changed; reset the workflow before restarting"
                )
            if station.remaining_ml > self.parameters.volume_tolerance_ml:
                self.physics.pipetting_command(PipettingAction.RESUME)
        if self.motion is not None and self.motion.state == ProgramState.PAUSED:
            self.motion.start()
        self.state = ProgramState.RUNNING

    def pause(self) -> None:
        """Hold robot intentions and stop a partially delivered pump request."""
        if self.state != ProgramState.RUNNING:
            return
        if self.physics.pipetting.state == PipettingState.DISPENSING:
            self.physics.pipetting_command(PipettingAction.STOP)
        if self.motion is not None:
            self.motion.pause()
        self.state = ProgramState.PAUSED

    def _fail(self, message: str) -> None:
        """Stop reagent delivery and rotor drive while retaining failure evidence."""
        self.pause()
        if self.physics.centrifuge.target_rpm > 0:
            self.physics.centrifuge_command(CentrifugeAction.STOP)
        self.error = message
        self.state = ProgramState.FAILED

    def advance(self) -> None:
        """Advance robot joint intentions without editing object dynamic poses."""
        if self.state != ProgramState.RUNNING:
            return
        self.step_time += self.physics.timestep
        self.elapsed += self.physics.timestep
        timeout = (
            self.parameters.transfer_timeout
            if self._is_transfer()
            else self.parameters.operation_timeout
        )
        if self.step.operation == WorkflowStep.SPIN:
            timeout += self.parameters.spin_seconds
        if self.step_time > timeout:
            self._fail(f"Timed out: {self.step.operation}")
            return
        if self._is_transfer() and self.motion is None:
            self._start_transfer()
        if self.motion is not None:
            self.motion.advance()

    def _is_transfer(self) -> bool:
        """Identify intentions that must be carried by the physical robot gripper."""
        return self.step.operation in (
            WorkflowStep.DISPENSER,
            WorkflowStep.SCALE,
            WorkflowStep.REGRASP,
            WorkflowStep.LOAD,
            WorkflowStep.UNLOAD,
            WorkflowStep.STORE,
        )

    def _start_transfer(self) -> None:
        """Construct a measured-source contact transfer to the selected support."""
        operation = self.step.operation
        support = None
        if operation == WorkflowStep.DISPENSER:
            destination = self.physics.pipetting.target_position
        elif operation == WorkflowStep.SCALE:
            destination = self.physics.scene[LaboratoryField.LABORATORY][
                WorkflowField.SCALE
            ][WorkflowField.PAN_POSITION]
            support = self.physics.model.body(
                self.physics.model.geom_bodyid[self.physics.scale.pan_id]
            ).name
        elif operation in (WorkflowStep.REGRASP, WorkflowStep.UNLOAD):
            destination = self._regrasp_position()
            support = LaboratoryBody.RACK
        elif operation == WorkflowStep.LOAD:
            destination = self.physics.centrifuge.slot_position(
                self.sample.centrifuge_slot
            )
            support = CentrifugeGeometry.ROTOR_BODY
        else:
            destination = next(
                slot[RefrigeratorField.POSITION]
                for slot in self.physics.refrigerator.metadata[RefrigeratorField.SLOTS]
                if slot[RefrigeratorField.ID] == self.sample.refrigerator_slot
            )
            support = RefrigeratorGeometry.BASE
        motion_module = importlib.import_module("cramera.laboratory_sample_transfer")
        grasp = (
            motion_module.SampleGrasp.TOP
            if operation in (WorkflowStep.LOAD, WorkflowStep.UNLOAD)
            else motion_module.SampleGrasp.FRONT
        )
        self.motion = motion_module.LaboratorySampleTransfer(
            physics=self.physics,
            object_key=self.sample.key,
            destination=tuple(destination),
            support_body=support,
            release_at_destination=operation != WorkflowStep.DISPENSER,
            from_held=operation == WorkflowStep.SCALE,
            grasp=grasp,
        )
        self.motion.start()
        self.sample.location = WorkflowLocation.ROBOT
        self.sample.rack_slot = None

    def observe(self) -> None:
        """Advance only after robot, liquid, load or instrument evidence confirms it."""
        self._observe_cycle()
        if self.state != ProgramState.RUNNING:
            return
        if self.motion is not None:
            self.motion.observe()
            if self.motion.state == ProgramState.FAILED:
                self._fail(
                    self.motion.snapshot()[ProgramField.ERROR]
                    or "Robot transfer failed"
                )
                return
        if self._is_transfer():
            if self.motion is not None and self.motion.state == ProgramState.SUCCEEDED:
                location = {
                    WorkflowStep.DISPENSER: WorkflowLocation.DISPENSER,
                    WorkflowStep.SCALE: WorkflowLocation.SCALE,
                    WorkflowStep.REGRASP: WorkflowLocation.REGRASP,
                    WorkflowStep.LOAD: WorkflowLocation.CENTRIFUGE,
                    WorkflowStep.UNLOAD: WorkflowLocation.REGRASP,
                    WorkflowStep.STORE: WorkflowLocation.REFRIGERATOR,
                }[self.step.operation]
                self.sample.location = location
                if location == WorkflowLocation.REGRASP:
                    self.sample.rack_slot = self.REGRASP_SLOT
                self._next()
            return
        if self._operation_complete():
            self._next()

    def _dose_source(self) -> PipettingSource:
        """Resolve the one channel associated with the current dosing operation."""
        return (
            PipettingSource.AMBER
            if self.step.operation == WorkflowStep.DOSE_A
            else PipettingSource.TEAL
        )

    def _held_sample(self) -> bool:
        """Measure bilateral fingertip ownership during a stationary dose."""
        if self.motion is None:
            return False
        body = self.physics.objects[self.sample.key].body_id
        forces = {name: 0.0 for name in LaboratoryMixingProgram.FINGERTIPS}
        for index, contact in enumerate(self.physics.data.contact):
            first, second = (
                int(self.physics.model.geom_bodyid[geometry])
                for geometry in (contact.geom1, contact.geom2)
            )
            if body not in (first, second) or contact.efc_address < 0:
                continue
            other = self.physics.model.body(second if first == body else first).name
            if other in forces:
                force = np.zeros(6)
                mujoco.mj_contactForce(
                    self.physics.model, self.physics.data, index, force
                )
                forces[other] += max(0.0, float(force[0]))
        return all(
            force > self.motion.parameters.contact_threshold
            for force in forces.values()
        )

    def _dose_complete(self) -> bool:
        """Confirm station accounting and actual receiver gain for a single dose."""
        station = self.physics.pipetting
        source = self._dose_source()
        if not self._held_sample():
            self._fail("The dispenser receiver lost opposing fingertip contacts")
            return False
        if not self.issued:
            reason = station.blocked_reason(
                self.sample.key,
                source,
                self.parameters.dose_ml,
                self.physics._object_poses(),
                frozenset(),
            )
            if reason is not None:
                self._fail(reason)
                return False
            self.dose_before = self.physics.liquid.tubes[self.sample.key].volume_ml
            self.physics.pipetting_command(
                PipettingAction.DISPENSE,
                key=self.sample.key,
                source=source,
                volume_ml=self.parameters.dose_ml,
            )
            self.issued = True
            return False
        if station.state == PipettingState.BLOCKED:
            self._fail(station.error or "The station interrupted the reagent dose")
            return False
        if station.state != PipettingState.COMPLETE:
            return False
        received = (
            self.physics.liquid.tubes[self.sample.key].volume_ml - self.dose_before
        )
        if (
            abs(received - self.parameters.dose_ml)
            > self.parameters.volume_tolerance_ml
            or abs(station.delivered_ml - received)
            > self.parameters.volume_tolerance_ml
        ):
            self._fail("The received quantity did not match the metered dose")
            return False
        if source == PipettingSource.AMBER:
            self.sample.dose_a_ml = received
        else:
            self.sample.dose_b_ml = received
        return True

    def _weight_complete(self) -> bool:
        """Record a stable pan reaction carrying the entire released sample."""
        scale = self.physics.scale
        if not scale.stable or scale.objects != [self.sample.key]:
            return False
        expected = (
            float(
                self.physics.model.body_mass[
                    self.physics.objects[self.sample.key].body_id
                ]
            )
            * 1000
        )
        if abs(scale.gross_grams - expected) > self.parameters.mass_tolerance_grams:
            return False
        self.sample.weight_grams = scale.gross_grams
        return True

    def _assign_next_slot(self) -> bool:
        """Match an already weighed loaded opposite or begin a new empty pair."""
        sample = self.sample
        if sample.weight_grams is None or not math.isfinite(sample.weight_grams):
            self._fail(
                "A stable sample weight is required before selecting its rotor holder"
            )
            return False
        holders = self.physics.centrifuge.slots
        by_key = {item.key: item for item in self.samples}
        free = [slot for slot in holders.values() if slot.key is None]
        matched = [
            slot
            for slot in free
            if holders[slot.opposite].key in by_key
            and by_key[holders[slot.opposite].key].weight_grams is not None
            and abs(
                by_key[holders[slot.opposite].key].weight_grams - sample.weight_grams
            )
            <= self.physics.centrifuge.parameters.balance_grams
        ]
        available = matched or [
            slot for slot in free if holders[slot.opposite].key is None
        ]
        if not available:
            self._fail(
                "No free rotor holder matches this sample's measured opposing mass"
            )
            return False
        sample.centrifuge_slot = available[0].identifier
        sample.refrigerator_slot = self.physics.refrigerator.metadata[
            RefrigeratorField.SLOTS
        ][self.step.sample_index][RefrigeratorField.ID]
        return True

    def _validate_loaded_balance(self) -> bool:
        """Require all four measured samples in genuinely balanced secured pairs."""
        centrifuge = self.physics.centrifuge
        if (
            _measured_pair_order(self.samples, centrifuge.parameters.balance_grams)
            is None
        ):
            self._fail(
                "The measured sample masses cannot form two balanced opposing pairs"
            )
            return False
        by_slot = {sample.centrifuge_slot: sample for sample in self.samples}
        for sample in self.samples:
            if (
                sample.centrifuge_slot not in centrifuge.slots
                or centrifuge.slots[sample.centrifuge_slot].key != sample.key
            ):
                self._fail(
                    "All four weighed samples must be secured before the batch cycle"
                )
                return False
            opposite = by_slot.get(centrifuge.slots[sample.centrifuge_slot].opposite)
            if (
                opposite is None
                or abs(opposite.weight_grams - sample.weight_grams)
                > centrifuge.parameters.balance_grams
            ):
                self._fail(
                    "The loaded opposing pair does not match its measured masses"
                )
                return False
        if not centrifuge.balanced:
            self._fail("The current physical rotor load is not balanced")
            return False
        self.load_verified = True
        return True

    def _seated_in_holder(self) -> bool:
        """Require the exact holder floor, upright pose and low seating velocity."""
        centrifuge = self.physics.centrifuge
        if not centrifuge.stopped or not centrifuge.lid_open:
            return False
        body = self.physics.objects[self.sample.key].body_id
        parameters = centrifuge.parameters
        if np.linalg.norm(
            self.physics.data.xpos[body]
            - centrifuge.slot_position(self.sample.centrifuge_slot)
        ) > parameters.seating_distance or self.physics.data.xmat[body].reshape(3, 3)[
            2, 2
        ] < math.cos(
            parameters.seating_angle
        ):
            return False
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.physics.model,
            self.physics.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body,
            velocity,
            0,
        )
        if np.linalg.norm(velocity[3:]) > parameters.seating_speed:
            return False
        floor = centrifuge.slots[self.sample.centrifuge_slot].floor_id
        support = 0.0
        for index, contact in enumerate(self.physics.data.contact):
            if floor not in (contact.geom1, contact.geom2) or contact.efc_address < 0:
                continue
            other = contact.geom2 if contact.geom1 == floor else contact.geom1
            if self.physics.model.geom_bodyid[other] == body:
                force = np.zeros(6)
                mujoco.mj_contactForce(
                    self.physics.model, self.physics.data, index, force
                )
                support += max(0.0, float(force[0]))
        weight = self.physics.model.body_mass[body] * np.linalg.norm(
            self.physics.model.opt.gravity
        )
        return support >= parameters.support_fraction * weight

    def _operation_complete(self) -> bool:
        """Execute each bounded instrument command once and wait for its observation."""
        operation = self.step.operation
        centrifuge = self.physics.centrifuge
        refrigerator = self.physics.refrigerator
        if operation in (WorkflowStep.DOSE_A, WorkflowStep.DOSE_B):
            return self._dose_complete()
        if operation == WorkflowStep.WEIGH:
            return self._weight_complete()
        if operation == WorkflowStep.ASSIGN:
            return self._assign_next_slot()
        if operation == WorkflowStep.BALANCE:
            return self._validate_loaded_balance()
        if operation in (WorkflowStep.OPEN_CENTRIFUGE, WorkflowStep.CLOSE_CENTRIFUGE):
            if not self.issued:
                if not centrifuge.stopped:
                    return False
                self.physics.centrifuge_command(
                    CentrifugeAction.OPEN
                    if operation == WorkflowStep.OPEN_CENTRIFUGE
                    else CentrifugeAction.CLOSE
                )
                self.issued = True
            return (
                centrifuge.lid_open
                if operation == WorkflowStep.OPEN_CENTRIFUGE
                else centrifuge.lid_closed
            )
        if operation in (
            WorkflowStep.OPEN_REFRIGERATOR,
            WorkflowStep.CLOSE_REFRIGERATOR,
        ):
            if not self.issued:
                self.physics.refrigerator_command(
                    RefrigeratorAction.OPEN
                    if operation == WorkflowStep.OPEN_REFRIGERATOR
                    else RefrigeratorAction.CLOSE
                )
                self.issued = True
            return (
                refrigerator.door_open
                if operation == WorkflowStep.OPEN_REFRIGERATOR
                else refrigerator.door_closed
            )
        if operation == WorkflowStep.SECURE:
            if not self._seated_in_holder():
                return False
            self.physics.centrifuge_command(
                CentrifugeAction.SECURE,
                slot=self.sample.centrifuge_slot,
                key=self.sample.key,
            )
            return centrifuge.slots[self.sample.centrifuge_slot].key == self.sample.key
        if operation == WorkflowStep.RELEASE:
            if not centrifuge.stopped or not centrifuge.lid_open:
                return False
            self.physics.centrifuge_command(
                CentrifugeAction.RELEASE, slot=self.sample.centrifuge_slot
            )
            return centrifuge.slots[self.sample.centrifuge_slot].key is None
        if operation == WorkflowStep.SPIN:
            return self._spin_complete()
        return (
            refrigerator.occupants().get(self.sample.refrigerator_slot)
            == self.sample.key
        )

    def _observe_cycle(self) -> None:
        """Track the physical timer even while robot intentions are paused."""
        if (
            self.state not in (ProgramState.RUNNING, ProgramState.PAUSED)
            or self.step.operation != WorkflowStep.SPIN
            or not self.issued
        ):
            return
        centrifuge = self.physics.centrifuge
        remaining = centrifuge.remaining_seconds
        if (
            self.spin_remaining - remaining
            > self.physics.timestep + self.parameters.timer_tolerance_seconds
        ):
            self._fail(
                "The centrifuge timer was interrupted before the full cycle completed"
            )
            return
        if (
            self.spin_remaining > 0
            and centrifuge.timer_started
            and (not centrifuge.lid_closed or not centrifuge.balanced)
        ):
            self._fail("A centrifuge interlock interrupted the timed cycle")
            return
        self.spin_observed = self.spin_observed or centrifuge.timer_started
        if (
            self.spin_remaining > 0
            and remaining == 0
            and centrifuge.timer_started
            and self.spin_observed
        ):
            self.spin_timer_complete = True
        self.spin_remaining = remaining

    def _spin_complete(self) -> bool:
        """Accept a full timed cycle only after observed speed and complete braking."""
        centrifuge = self.physics.centrifuge
        if not self.issued:
            if not centrifuge.balanced or not centrifuge.lid_closed:
                self._fail(
                    "The measured centrifuge load or lid is not ready for spinning"
                )
                return False
            self.physics.centrifuge_command(
                CentrifugeAction.START,
                rpm=min(self.parameters.rotor_rpm, centrifuge.maximum_rpm),
                duration=self.parameters.spin_seconds,
            )
            self.spin_remaining = self.parameters.spin_seconds
            self.issued = True
            return False
        self.spin_observed = self.spin_observed or centrifuge.timer_started
        if not centrifuge.balanced:
            self._fail("The centrifuge lost its balanced secured load")
            return False
        if centrifuge.stopped and centrifuge.remaining_seconds == 0:
            if (
                not self.spin_timer_complete
                or not self.spin_observed
                or not centrifuge.timer_started
            ):
                self._fail(
                    "The centrifuge stopped before completing its measured timed cycle"
                )
                return False
            self.spin_finished = True
            return True
        return False

    def _next(self) -> None:
        """Retain confirmed evidence and enter the next finite intention."""
        self.step_index += 1
        self.step_time = 0.0
        self.issued = False
        if self.step.operation not in (WorkflowStep.DOSE_A, WorkflowStep.DOSE_B):
            self.motion = None
        if self.step_index == len(self.steps):
            self._verify_complete()

    def _verify_complete(self) -> None:
        """Accept completion only while the final storage and device states hold."""
        if not self.physics.refrigerator.door_closed:
            self._fail("The sample cooler door did not remain closed")
            return
        if not self.physics.centrifuge.lid_closed:
            self._fail("The centrifuge lid did not remain closed")
            return
        if not self.physics.centrifuge.stopped:
            self._fail("The centrifuge rotor did not remain stopped")
            return
        occupants = self.physics.refrigerator.occupants()
        if not self.spin_finished or any(
            occupants.get(sample.refrigerator_slot) != sample.key
            for sample in self.samples
        ):
            self._fail("The processed samples did not remain supported in the cooler")
            return
        if any(
            abs(
                self.physics.liquid.tubes[sample.key].volume_ml
                - len(PipettingSource) * self.parameters.dose_ml
            )
            > self.parameters.volume_tolerance_ml
            for sample in self.samples
        ):
            self._fail("The stored samples did not retain both complete reagent doses")
            return
        self.state = ProgramState.SUCCEEDED

    def _completed_stages(self) -> list[WorkflowStage]:
        """Publish batch completion only after every sample has its evidence."""
        completed = []
        if all(
            sample.dose_a_ml is not None and sample.dose_b_ml is not None
            for sample in self.samples
        ):
            completed.append(WorkflowStage.FILL)
        if all(sample.weight_grams is not None for sample in self.samples):
            completed.append(WorkflowStage.WEIGH)
        if self.load_verified:
            completed.append(WorkflowStage.LOAD)
        if self.spin_finished:
            completed.append(WorkflowStage.SPIN)
        if self.state == ProgramState.SUCCEEDED:
            completed.extend((WorkflowStage.STORE, WorkflowStage.COMPLETE))
        return completed

    def snapshot(self) -> dict[str, Any]:
        """Expose stage progress together with measured sample evidence."""
        status = self.motion.snapshot() if self.motion is not None else {}
        status.update(
            {
                ProgramField.SOURCE: WorkflowField.SOURCE,
                ProgramField.STATE: self.state,
                ProgramField.PHASE: (
                    self.step.stage
                    if self.state != ProgramState.SUCCEEDED
                    else WorkflowStage.COMPLETE
                ),
                ProgramField.PROGRESS: self.step_index / len(self.steps),
                ProgramField.ERROR: self.error,
                ProgramField.TIME: self.elapsed,
                WorkflowField.WORKFLOW: {
                    WorkflowField.STAGE: (
                        self.step.stage
                        if self.state != ProgramState.SUCCEEDED
                        else WorkflowStage.COMPLETE
                    ),
                    WorkflowField.STAGE_LABEL: self.step.operation,
                    WorkflowField.SAMPLE_KEY: (
                        self.sample.key if self.sample is not None else None
                    ),
                    WorkflowField.SAMPLES: [
                        sample.snapshot() for sample in self.samples
                    ],
                    WorkflowField.COMPLETED: self._completed_stages(),
                    WorkflowField.FAILURE: self.error,
                    WorkflowCommandField.RESERVED: self.state
                    in (ProgramState.RUNNING, ProgramState.PAUSED),
                },
            }
        )
        return status
