"""Finite reagent reservoirs and simulation-timed dispensing into measured vessels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cramera.laboratory_liquid import LaboratoryLiquid, TubePose
from cramera.laboratory_pipetting_command import (
    InvalidPipettingCommand,
    PipettingAction,
    PipettingCommand,
    PipettingField,
    PipettingParameters,
    PipettingSource,
    PipettingState,
)


# %% finite reagent inventories
@dataclass
class ReagentReservoir:
    """One independent reagent inventory feeding the stationary nozzle."""

    identifier: PipettingSource
    """Public channel identifier."""
    label: str
    """English reagent description."""
    volume_ml: float
    """Remaining reagent in milliliters."""
    color: np.ndarray
    """Linear RGB color used for conservative mixing."""

    def snapshot(self, capacity_ml: float) -> dict:
        """Return detached inventory values for controls and rendering."""
        return {
            PipettingField.ID: self.identifier,
            PipettingField.LABEL: self.label,
            PipettingField.VOLUME: self.volume_ml,
            PipettingField.CAPACITY: capacity_ml,
            PipettingField.COLOR: self.color.tolist(),
        }


@dataclass
class LaboratoryPipetting:
    """Transfer measured liquid volumes only while receiver interlocks hold."""

    scene: dict
    """Authored station placement and removable stopper information."""
    liquid: LaboratoryLiquid
    """Receiver volumes, colors and authoritative liquid accounting."""
    parameters: PipettingParameters = field(default_factory=PipettingParameters)
    """Pump flow limits and measured alignment tolerances."""
    available: bool = field(init=False)
    """Whether this authored laboratory declares a dispensing station."""
    target_position: tuple[float, float, float] = field(init=False)
    """Required receiver-root position in world coordinates."""
    nozzle_position: tuple[float, float, float] = field(init=False)
    """Common outlet position of the two independent reagent channels."""
    reservoirs: dict[PipettingSource, ReagentReservoir] = field(init=False)
    """Independent remaining reagent quantities."""
    state: PipettingState = field(init=False, default=PipettingState.IDLE)
    """Current pump operating state."""
    source: PipettingSource | None = field(init=False, default=None)
    """Selected channel for the most recently accepted dose."""
    key: str | None = field(init=False, default=None)
    """Receiver of the most recently accepted dose."""
    requested_ml: float = field(init=False, default=0.0)
    """Total requested volume of the current or last dose."""
    delivered_ml: float = field(init=False, default=0.0)
    """Volume already transferred during the current or last dose."""
    error: str | None = field(init=False, default=None)
    """Measured condition that interrupted the latest dose."""

    def __post_init__(self) -> None:
        """Load authored placement and initialize both finite channels."""
        metadata = self.scene.get(PipettingField.LABORATORY, {}).get(
            PipettingField.PIPETTING, {}
        )
        self.available = bool(metadata) and bool(
            metadata.get(PipettingField.AVAILABLE, True)
        )
        self.target_position = tuple(
            metadata.get(
                PipettingField.TARGET_POSITION, self.parameters.target_position
            )
        )
        self.nozzle_position = tuple(
            metadata.get(
                PipettingField.NOZZLE_POSITION, self.parameters.nozzle_position
            )
        )
        self.reservoirs = {
            PipettingSource.AMBER: ReagentReservoir(
                PipettingSource.AMBER,
                "Amber reagent",
                self.parameters.reservoir_ml,
                np.array([0.82, 0.32, 0.055]),
            ),
            PipettingSource.TEAL: ReagentReservoir(
                PipettingSource.TEAL,
                "Teal reagent",
                self.parameters.reservoir_ml,
                np.array([0.12, 0.56, 0.52]),
            ),
        }

    @property
    def remaining_ml(self) -> float:
        """Return the volume still owed by the most recently accepted request."""
        return max(0.0, self.requested_ml - self.delivered_ml)

    def reset(self) -> None:
        """Restore reagent supplies and clear pending delivery after a world reset."""
        for reservoir in self.reservoirs.values():
            reservoir.volume_ml = self.parameters.reservoir_ml
        self.state = PipettingState.IDLE
        self.source = None
        self.key = None
        self.requested_ml = 0.0
        self.delivered_ml = 0.0
        self.error = None

    def command(
        self,
        action: str,
        poses: dict[str, list[float]],
        secured_keys: frozenset[str] = frozenset(),
        key: str | None = None,
        source: str | None = None,
        volume_ml: float | None = None,
    ) -> None:
        """Validate a new operation against the measured receiver and finite supplies."""
        command = PipettingCommand(action, key, source, volume_ml)
        if not self.available:
            raise InvalidPipettingCommand("This laboratory has no pipetting station.")
        if command.action == PipettingAction.STOP:
            self.state = PipettingState.IDLE
            self.error = None
            return
        if self.state == PipettingState.DISPENSING:
            raise InvalidPipettingCommand(
                "Stop the current dose before changing the pump."
            )
        if command.action == PipettingAction.REFILL:
            self.reservoirs[command.source].volume_ml = self.parameters.reservoir_ml
            return
        if command.action == PipettingAction.RESUME:
            if (
                self.key is None
                or self.source is None
                or self.remaining_ml <= self.parameters.volume_tolerance
            ):
                raise InvalidPipettingCommand("There is no incomplete dose to resume.")
            key, source, volume_ml = self.key, self.source, self.remaining_ml
        else:
            key, source, volume_ml = command.key, command.source, command.volume_ml
        reason = self.blocked_reason(key, source, volume_ml, poses, secured_keys)
        if reason is not None:
            raise InvalidPipettingCommand(reason)
        if command.action == PipettingAction.DISPENSE:
            self.key = key
            self.source = PipettingSource(source)
            self.requested_ml = volume_ml
            self.delivered_ml = 0.0
        self.error = None
        self.state = PipettingState.DISPENSING

    def blocked_reason(
        self,
        key: str,
        source: str,
        volume_ml: float,
        poses: dict[str, list[float]],
        secured_keys: frozenset[str] = frozenset(),
    ) -> str | None:
        """Describe a measured obstruction without mutating the pump or liquids."""
        if key not in self.liquid.tubes or key not in poses:
            return "Select an available liquid receiver."
        if key in secured_keys:
            return "Release the receiver from the centrifuge first."
        pose = TubePose.from_values(poses[key])
        if (
            np.linalg.norm(pose.world_T_tube[:3, 3] - self.target_position)
            > self.parameters.position_tolerance
        ):
            return "Place the receiver beneath the pipetting nozzle."
        if pose.world_T_tube[2, 2] < math.cos(self.parameters.upright_tolerance):
            return "Keep the receiver upright beneath the nozzle."
        rim = pose.world_point(np.array([0.0, 0.0, self.liquid.interior.rim]))
        gap = self.nozzle_position[2] - rim[2]
        if (
            np.linalg.norm(rim[:2] - self.nozzle_position[:2])
            > self.liquid.interior.radius
            or not self.parameters.minimum_gap <= gap <= self.parameters.maximum_gap
        ):
            return "Align the open rim directly beneath the nozzle."
        stopper = self.scene.get(PipettingField.LABORATORY, {}).get(
            PipettingField.STOPPER, {}
        )
        stopper_key = stopper.get(PipettingField.KEY)
        if stopper_key in poses:
            local_stopper = pose.local_point(np.asarray(poses[stopper_key][:3]))
            cap_height = self.liquid.interior.rim - stopper.get(
                PipettingField.INSERTION_DEPTH, 0.0115
            )
            if (
                np.linalg.norm(local_stopper[:2]) < 2 * self.liquid.interior.radius
                and abs(local_stopper[2] - cap_height)
                < self.parameters.stopper_clearance
            ):
                return "Remove the stopper before dispensing into this receiver."
        if (
            volume_ml
            > self.liquid.interior.capacity_ml
            - self.liquid.tubes[key].volume_ml
            + self.parameters.volume_tolerance
        ):
            return "The receiver has insufficient free capacity for this dose."
        if (
            volume_ml
            > self.reservoirs[source].volume_ml + self.parameters.volume_tolerance
        ):
            return "The selected reagent reservoir has insufficient liquid."
        return None

    def advance(
        self,
        duration: float,
        poses: dict[str, list[float]],
        secured_keys: frozenset[str] = frozenset(),
    ) -> None:
        """Deliver elapsed simulated flow only while every receiver interlock holds."""
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not 0 <= duration <= self.liquid.parameters.maximum_duration
        ):
            raise InvalidPipettingCommand(
                "Pump duration must be finite and between zero and one second."
            )
        if self.state != PipettingState.DISPENSING or duration == 0:
            return
        reason = self.blocked_reason(
            self.key, self.source, self.remaining_ml, poses, secured_keys
        )
        if reason is not None:
            self.state = PipettingState.BLOCKED
            self.error = reason
            return
        reservoir = self.reservoirs[self.source]
        tube = self.liquid.tubes[self.key]
        amount = min(
            self.remaining_ml,
            self.parameters.flow_rate_ml * duration,
            reservoir.volume_ml,
        )
        accepted = tube.accept(
            amount, reservoir.color, self.liquid.interior.capacity_ml
        )
        reservoir.volume_ml -= accepted
        self.delivered_ml += accepted
        self.liquid.initial_ml += accepted
        tube.offset = self.liquid.interior.surface_offset(tube.normal, tube.volume_ml)
        if self.remaining_ml <= self.parameters.volume_tolerance:
            self.state = PipettingState.COMPLETE

    def snapshot(self) -> dict:
        """Expose measured delivery and finite inventory for the station controls."""
        return {
            PipettingField.AVAILABLE: self.available,
            PipettingField.STATE: self.state,
            PipettingField.SOURCE: self.source,
            PipettingField.KEY: self.key,
            PipettingField.REQUESTED: self.requested_ml,
            PipettingField.DELIVERED: self.delivered_ml,
            PipettingField.REMAINING: self.remaining_ml,
            PipettingField.ERROR: self.error,
            PipettingField.TARGET_POSITION: list(self.target_position),
            PipettingField.NOZZLE_POSITION: list(self.nozzle_position),
            PipettingField.RESERVOIRS: [
                reservoir.snapshot(self.parameters.reservoir_ml)
                for reservoir in self.reservoirs.values()
            ],
            PipettingField.MINIMUM: self.parameters.minimum_ml,
            PipettingField.MAXIMUM: self.parameters.maximum_ml,
            PipettingField.FLOW_RATE: self.parameters.flow_rate_ml,
        }
