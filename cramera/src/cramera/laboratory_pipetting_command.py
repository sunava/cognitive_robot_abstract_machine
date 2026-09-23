"""Lightweight command schemas and shared station vocabulary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


# %% command and state vocabulary
class PipettingAction(StrEnum):
    """Operations accepted by the two-channel dispensing station."""

    DISPENSE = "dispense"
    """Start a bounded dose into an aligned open receiver."""
    STOP = "stop"
    """Halt flow while retaining partial-dose accounting."""
    RESUME = "resume"
    """Resume the remaining quantity of a stopped dose."""
    REFILL = "refill"
    """Explicitly replenish one inactive reagent reservoir."""


class PipettingState(StrEnum):
    """Observable pump states independent of the simulation worker."""

    IDLE = "idle"
    """The pump is inactive or deliberately stopped."""
    DISPENSING = "dispensing"
    """The pump advances against simulated elapsed time."""
    COMPLETE = "complete"
    """The entire requested dose reached the receiver."""
    BLOCKED = "blocked"
    """A measured interlock interrupted the dose."""


class PipettingSource(StrEnum):
    """Independent reagent channels with distinct fixed colors."""

    AMBER = "amber"
    """Amber demonstration reagent."""
    TEAL = "teal"
    """Teal demonstration reagent."""


class PipettingField(StrEnum):
    """Shared metadata, request and authoritative observation fields."""

    PIPETTING = "pipetting"
    """Station metadata and observed state."""
    AVAILABLE = "available"
    """Whether the instrument exists in this scene."""
    ACTION = "action"
    """Requested station operation."""
    STATE = "state"
    """Current pump operating state."""
    SOURCE = "source"
    """Selected reagent channel."""
    KEY = "key"
    """Authored receiver identifier."""
    VOLUME = "volumeMl"
    """Remaining reservoir or requested dose in milliliters."""
    CAPACITY = "capacityMl"
    """Reservoir capacity in milliliters."""
    REQUESTED = "requestedMl"
    """Total volume requested for the current dose."""
    DELIVERED = "deliveredMl"
    """Volume already delivered for the current dose."""
    REMAINING = "remainingMl"
    """Volume still pending for the current dose."""
    ERROR = "error"
    """Measured interlock preventing further flow."""
    TARGET_POSITION = "targetPosition"
    """Receiver-root destination in world coordinates."""
    NOZZLE_POSITION = "nozzlePosition"
    """Reagent outlet position in world coordinates."""
    RESERVOIRS = "reservoirs"
    """Independent reagent inventories."""
    ID = "id"
    """Stable reagent channel identifier."""
    LABEL = "label"
    """English reagent description."""
    COLOR = "color"
    """Linear RGB mixing color."""
    MINIMUM = "minimumMl"
    """Smallest accepted new dose in milliliters."""
    MAXIMUM = "maximumMl"
    """Largest accepted new dose in milliliters."""
    FLOW_RATE = "flowRateMl"
    """Delivery rate in milliliters per simulated second."""
    LABORATORY = "laboratory"
    """Authored laboratory geometry metadata."""
    STOPPER = "stopper"
    """Removable stopper description."""
    INSERTION_DEPTH = "insertionDepth"
    """Stopper insertion depth beneath the vessel rim."""


@dataclass(frozen=True)
class PipettingParameters:
    """Simulation limits and receiver tolerances for the demonstration pump."""

    target_position: tuple[float, float, float] = (-0.10, -0.11, 0.916)
    """Default receiver-root placement for station-enabled prototype bundles."""
    nozzle_position: tuple[float, float, float] = (-0.10, -0.11, 1.080)
    """Default common reagent outlet in world coordinates."""
    stopper_insertion_depth: float = 0.0115
    """Default removable stopper insertion depth in meters."""
    minimum_ml: float = 0.1
    """Smallest new dose in milliliters."""
    maximum_ml: float = 5.0
    """Largest new dose in milliliters."""
    default_ml: float = 1.0
    """Requested dose when the caller omits a quantity."""
    flow_rate_ml: float = 1.0
    """Delivered milliliters per simulated second."""
    reservoir_ml: float = 50.0
    """Each reagent reservoir's capacity in milliliters."""
    position_tolerance: float = 0.008
    """Maximum measured root distance from the receiver position in meters."""
    upright_tolerance: float = 0.05
    """Maximum vessel-axis tilt in radians."""
    minimum_gap: float = 0.002
    """Smallest vertical nozzle-to-rim clearance in meters."""
    maximum_gap: float = 0.03
    """Largest vertical nozzle-to-rim clearance in meters."""
    stopper_clearance: float = 0.02
    """Axial clearance required around a potential stopper insertion in meters."""
    volume_tolerance: float = 1e-10
    """Numerical tolerance for completion and reservoir comparisons."""


@dataclass
class InvalidPipettingCommand(ValueError):
    """A dose violates the request schema or measured operating interlocks."""

    reason: str
    """Validation explanation returned to the operator."""

    def __str__(self) -> str:
        """Expose the actionable validation explanation."""
        return self.reason


@dataclass(frozen=True)
class PipettingCommand:
    """Validated operation carrying only bounded dispensing parameters."""

    action: PipettingAction | str
    """Requested station operation."""
    key: str | None = None
    """Receiver identifier for a new dose."""
    source: PipettingSource | str | None = None
    """Selected reagent channel."""
    volume_ml: float | None = None
    """New dose quantity in milliliters."""

    def __post_init__(self) -> None:
        """Reject malformed commands before any simulation state can change."""
        if not isinstance(self.action, str) or self.action not in PipettingAction:
            raise InvalidPipettingCommand("Unknown pipetting operation.")
        if self.action in (PipettingAction.STOP, PipettingAction.RESUME):
            if any(
                value is not None for value in (self.key, self.source, self.volume_ml)
            ):
                raise InvalidPipettingCommand(
                    "This pump operation accepts no parameters."
                )
            return
        if not isinstance(self.source, str) or self.source not in PipettingSource:
            raise InvalidPipettingCommand("Select the amber or teal reagent channel.")
        if self.action == PipettingAction.REFILL:
            if self.key is not None or self.volume_ml is not None:
                raise InvalidPipettingCommand("Refill accepts only a reagent channel.")
            return
        if not isinstance(self.key, str) or not self.key:
            raise InvalidPipettingCommand("Select a receiver for the dose.")
        parameters = PipettingParameters()
        volume = parameters.default_ml if self.volume_ml is None else self.volume_ml
        if (
            isinstance(volume, bool)
            or not isinstance(volume, (int, float))
            or not math.isfinite(volume)
            or not parameters.minimum_ml <= volume <= parameters.maximum_ml
        ):
            raise InvalidPipettingCommand(
                f"Dose must be between {parameters.minimum_ml:g} and {parameters.maximum_ml:g} mL."
            )
        object.__setattr__(self, "volume_ml", float(volume))

    @classmethod
    def from_payload(cls, payload: dict) -> PipettingCommand:
        """Parse the exact public schema and exclude unsolicited pose or state writes."""
        if not isinstance(payload, dict) or PipettingField.ACTION not in payload:
            raise InvalidPipettingCommand("Expected a pipetting operation.")
        action = payload[PipettingField.ACTION]
        allowed = {
            PipettingAction.DISPENSE: {
                PipettingField.ACTION,
                PipettingField.KEY,
                PipettingField.SOURCE,
                PipettingField.VOLUME,
            },
            PipettingAction.REFILL: {PipettingField.ACTION, PipettingField.SOURCE},
            PipettingAction.STOP: {PipettingField.ACTION},
            PipettingAction.RESUME: {PipettingField.ACTION},
        }
        if (
            not isinstance(action, str)
            or action not in allowed
            or not set(payload) <= allowed[action]
        ):
            raise InvalidPipettingCommand("Unexpected pipetting command fields.")
        return cls(
            action,
            payload.get(PipettingField.KEY),
            payload.get(PipettingField.SOURCE),
            payload.get(PipettingField.VOLUME),
        )
