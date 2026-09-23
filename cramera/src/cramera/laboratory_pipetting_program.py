"""Present a physically grasped glass to a fixed two-source dosing station."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import Any

from cramera.laboratory_mixing_program import (
    LaboratoryMixingProgram,
    MixingField,
    MixingPhase,
    MixingStage,
)
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import (
    JointReference,
    ProgramField,
    ProgramState,
)
from cramera.laboratory_pipetting import (
    PipettingAction,
    PipettingSource,
    PipettingState,
)
from cramera.laboratory_world import LaboratoryBody, LaboratorySlot


# %% station recipe vocabulary
class PipettingPhase(StrEnum):
    """Additional stages needed for fixed-nozzle liquid dispensing."""

    TRANSPORT = "transport"
    """The held upright receiver travels clear of the rack and station."""
    DISPENSING = "dispensing"
    """The held receiver remains aligned while the station meters liquid."""


class PipettingField(StrEnum):
    """Measured dispensing recipe status shared with the visualization."""

    SOURCE = "laboratory_pipetting"
    """Controller identity distinguishing pipetting from manual pouring."""
    DELIVERED = "deliveredMl"
    """Confirmed liquid delivered from each station reservoir."""
    LIQUID_SOURCE = "liquidSource"
    """Reservoir selected by the current dosing stage."""


@dataclass(frozen=True)
class PipettingStage(MixingStage):
    """A contact-driven manipulation stage with an optional metered dose."""

    phase: MixingPhase | PipettingPhase
    """Observable movement or station operation."""
    source: PipettingSource | None = None
    """Liquid reservoir metered while the receiver stays under the nozzle."""


# %% contact-controlled station operation
@dataclass
class LaboratoryPipettingProgram(LaboratoryMixingProgram):
    """Dose two liquids through a nozzle while actuated fingertips hold the glass."""

    doses: dict[PipettingSource, float] = field(
        default_factory=lambda: {source: 1.0 for source in PipettingSource}
    )
    """Requested recipe quantity per reservoir, in milliliters."""
    delivered_ml: dict[PipettingSource, float] = field(init=False, default_factory=dict)
    """Completed physical station deliveries per reservoir, in milliliters."""
    dose_started: bool = field(init=False, default=False)
    """Whether the current stage has issued its single station request."""
    dose_before_ml: float = field(init=False, default=0.0)
    """Receiver contents immediately before the current delivery."""
    supported_release: bool = field(init=False, default=False)
    """Whether rack support was physically confirmed before opening the fingers."""
    station_front_clearance: float = 0.1
    """Forward approach clearance keeping the raised tube away from the nozzle."""
    volume_tolerance_ml: float = 0.000001
    """Maximum numerical discrepancy between metered and received quantities."""

    def reset(self) -> None:
        """Clear recipe evidence while preserving all physical poses and contents."""
        self.destination = self._slot(LaboratorySlot.A1)
        self.delivered_ml = {source: 0.0 for source in self.doses}
        self.dose_started = False
        self.supported_release = False
        super().reset()

    def _recipe(self) -> list[PipettingStage]:
        """Pick A1, enter below the fixed nozzle, dose, and return through open space."""
        source = self._slot(LaboratorySlot.A1)
        station = np.asarray(self.physics.pipetting.target_position)
        lift = np.array([0.0, 0.0, self.parameters.lift_height])
        front = np.array([0.0, -self.parameters.approach_distance, 0.0])
        station_front = station - np.array([0.0, self.station_front_clearance, 0.0])
        rotation = tuple(Rotation.identity().as_quat())
        stages: list[PipettingStage] = []

        def append(
            phase: MixingPhase | PipettingPhase,
            position: np.ndarray,
            duration: float,
            closed: bool,
            reservoir: PipettingSource | None = None,
            joints: dict[str, float] | None = None,
        ) -> None:
            """Append one upright bounded-motion or station-hold intention."""
            stages.append(
                PipettingStage(
                    phase=phase,
                    object_key=LaboratoryBody.CLEAR_TUBE,
                    position=tuple(position),
                    rotation=rotation,
                    duration=duration,
                    closed=closed,
                    joint_targets=joints,
                    source=reservoir,
                )
            )

        append(
            MixingPhase.APPROACH,
            source + lift,
            self.parameters.front_transition_time,
            False,
            joints=self.FRONT_TRANSITION,
        )
        append(MixingPhase.APPROACH, source + front, 5.0, False)
        append(MixingPhase.APPROACH, source, 4.0, False)
        append(MixingPhase.GRASP, source, 2.0, True)
        append(MixingPhase.LIFT, source + lift, 4.0, True)
        append(PipettingPhase.TRANSPORT, station_front + lift, 6.0, True)
        append(MixingPhase.PLACE, station_front, 4.0, True)
        append(MixingPhase.PLACE, station, 4.0, True)
        for reservoir in self.doses:
            append(PipettingPhase.DISPENSING, station, 1.0, True, reservoir)
        append(MixingPhase.PLACE, station_front, 4.0, True)
        append(PipettingPhase.TRANSPORT, station_front + lift, 4.0, True)
        append(MixingPhase.PLACE, source + lift, 6.0, True)
        append(MixingPhase.PLACE, source, 5.0, True)
        append(MixingPhase.RELEASE, source, 1.0, False)
        append(MixingPhase.SETTLE, source + front, 3.0, False)
        return stages

    @property
    def stage(self) -> PipettingStage:
        """Current movement or metered-dose intention, including after completion."""
        return self.stages[min(self.stage_index, len(self.stages) - 1)]

    def _enter_stage(self) -> None:
        """Capture measured motion origins and reset the next station request."""
        self.dose_started = False
        super()._enter_stage()

    def start(self) -> None:
        """Start from the current A1 sample or resume a paused physical delivery."""
        if self.state not in (ProgramState.IDLE, ProgramState.PAUSED):
            return
        if self.state == ProgramState.IDLE:
            station = self.physics.pipetting
            if not station.available:
                raise InvalidPhysicsTarget("This laboratory has no pipetting station")
            if any(
                station.reservoirs[source].volume_ml < quantity
                for source, quantity in self.doses.items()
            ):
                raise InvalidPhysicsTarget(
                    "Refill the reagent reservoirs before starting pipetting"
                )
            if (
                self._receiver_volume() + sum(self.doses.values())
                > self.physics.liquid.interior.capacity_ml
            ):
                raise InvalidPhysicsTarget(
                    "The clear glass has insufficient capacity for both doses"
                )
            identifier = self.physics.objects[LaboratoryBody.CLEAR_TUBE].body_id
            position = self.physics.data.xpos[identifier]
            axis = self.physics.data.xmat[identifier].reshape(3, 3)[:, 2]
            if np.linalg.norm(
                position - self.destination
            ) > self.parameters.destination_tolerance or axis[2] < math.cos(
                self.parameters.upright_tolerance
            ):
                raise InvalidPhysicsTarget(
                    "Return the clear glass upright to rack A1 before starting pipetting"
                )
        elif self.dose_started:
            station = self.physics.pipetting
            if (
                station.key != LaboratoryBody.CLEAR_TUBE
                or station.source != self.stage.source
                or abs(station.requested_ml - self.doses[self.stage.source])
                > self.volume_tolerance_ml
            ):
                raise InvalidPhysicsTarget(
                    "The paused station dose was replaced; reset the robot recipe before restarting"
                )
            if station.remaining_ml > self.volume_tolerance_ml:
                self.physics.pipetting_command(PipettingAction.RESUME)
        actuator = self.physics.model.actuator(JointReference.GRIPPER_JOINT).id
        self.physics.model.actuator_forcerange[actuator] = (
            -self.parameters.finger_torque,
            self.parameters.finger_torque,
        )
        self.physics.release()
        self.state = ProgramState.RUNNING

    def pause(self) -> None:
        """Freeze the robot intention and stop its pending pump without discarding it."""
        if (
            self.state == ProgramState.RUNNING
            and self.dose_started
            and self.physics.pipetting.state == PipettingState.DISPENSING
        ):
            self.physics.pipetting_command(PipettingAction.STOP)
        super().pause()

    def _complete_stage(self) -> None:
        """Advance dosing only after measured alignment and conserved delivery."""
        if self.stage.phase == PipettingPhase.DISPENSING:
            if not self._dose_complete():
                return
        if self._placing_for_release():
            self.supported_release = True
        super()._complete_stage()

    def _dose_complete(self) -> bool:
        """Meter a source once and verify the observed receiver gain before moving."""
        station = self.physics.pipetting
        if not self.dose_started:
            position, rotation = self._pose()
            aligned = (
                np.linalg.norm(position - np.asarray(self.stage.position))
                <= self.parameters.position_tolerance
                and rotation.apply([0.0, 0.0, 1.0])[2]
                >= math.cos(self.parameters.axis_tolerance)
                and self.holding
            )
            if not aligned:
                if (
                    self.stage_time
                    > self.stage.duration + self.parameters.stage_timeout
                ):
                    self._fail(
                        "The held glass did not align under the pipetting nozzle"
                    )
                return False
            self.dose_before_ml = self._receiver_volume()
            reason = station.blocked_reason(
                LaboratoryBody.CLEAR_TUBE,
                self.stage.source,
                self.doses[self.stage.source],
                self.physics._object_poses(),
                frozenset(),
            )
            if reason is not None:
                self._fail(reason)
                return False
            self.physics.pipetting_command(
                PipettingAction.DISPENSE,
                key=LaboratoryBody.CLEAR_TUBE,
                source=self.stage.source,
                volume_ml=self.doses[self.stage.source],
            )
            self.dose_started = True
            return False
        if station.state == PipettingState.BLOCKED:
            self._fail(station.error or "The pipetting station interrupted delivery")
            return False
        if (
            station.state not in (PipettingState.COMPLETE, PipettingState.IDLE)
            or station.remaining_ml > self.volume_tolerance_ml
        ):
            return False
        received = self._receiver_volume() - self.dose_before_ml
        expected = self.doses[self.stage.source]
        if (
            abs(station.delivered_ml - expected) > self.volume_tolerance_ml
            or abs(received - expected) > self.volume_tolerance_ml
        ):
            self._fail("The measured receiver gain did not match the requested dose")
            return False
        self.delivered_ml[self.stage.source] += received
        return True

    def _fail(self, message: str) -> None:
        """Stop delivery immediately when physical manipulation acceptance fails."""
        if self.dose_started:
            self.physics.pipetting_command(PipettingAction.STOP)
        super()._fail(message)

    def _finish(self) -> None:
        """Require both doses and an upright released receiver supported in A1."""
        self.released = self.gripper_force < self.parameters.contact_threshold
        identifier = self.physics.objects[LaboratoryBody.CLEAR_TUBE].body_id
        upright = self.physics.data.xmat[identifier].reshape(3, 3)[2, 2]
        if any(
            abs(self.delivered_ml[source] - quantity) > self.volume_tolerance_ml
            for source, quantity in self.doses.items()
        ):
            self._fail("Both station reservoirs did not deliver the requested doses")
        elif (
            not self.released
            or not self.supported_release
            or self.destination_error > self.parameters.destination_tolerance
        ):
            self._fail("The dosed receiver did not settle released in rack A1")
        elif upright < math.cos(self.parameters.upright_tolerance):
            self._fail("The dosed receiver did not settle upright")
        else:
            self.state = ProgramState.SUCCEEDED

    def snapshot(self) -> dict[str, Any]:
        """Publish manipulation contacts and the station's measured liquid ledger."""
        snapshot = super().snapshot()
        for key in (
            MixingField.CAPTURED,
            MixingField.SWIRL_COMPLETED,
            MixingField.SWIRL_TURNS,
        ):
            del snapshot[key]
        snapshot[ProgramField.SOURCE] = PipettingField.SOURCE
        snapshot[PipettingField.DELIVERED] = dict(self.delivered_ml)
        snapshot[PipettingField.LIQUID_SOURCE] = self.stage.source
        return snapshot
