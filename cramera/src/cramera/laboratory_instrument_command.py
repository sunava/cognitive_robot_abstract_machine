"""Engine-independent requests for robot-operated instrument covers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Any


# %% bounded request vocabulary
class InstrumentName(StrEnum):
    """Devices whose articulated cover can be manipulated by the robot."""

    CENTRIFUGE = "centrifuge"
    """Protective cover of the rotor chamber."""
    REFRIGERATOR = "refrigerator"
    """Glass door of the sample cooler."""


class InstrumentAction(StrEnum):
    """Reusable cover operations available to an operator."""

    OPEN = "open"
    """Open the selected device using its handle."""
    CLOSE = "close"
    """Close the selected device using its handle."""
    OPEN_CLOSE = "open_close"
    """Open the selected device and then close it."""


class InstrumentField(StrEnum):
    """Request fields and the robot's standalone operation marker."""

    INSTRUMENT = "instrument"
    """Requested device, or the standalone command in a robot snapshot."""
    ACTION = "action"
    """Requested cover operation."""


@dataclass
class InvalidInstrumentCommand(ValueError):
    """A request does not identify exactly one supported device operation."""

    reason: str
    """Validation failure returned to the operator."""

    def __str__(self) -> str:
        """Return the reason the request was rejected."""
        return self.reason


@dataclass(frozen=True)
class InstrumentCommand:
    """A robot cover operation preserving the current samples and physical state."""

    instrument: InstrumentName
    """Device whose handle the robot should manipulate."""
    action: InstrumentAction
    """Individual operation or open-and-close sequence."""

    @classmethod
    def from_payload(cls, payload: Any) -> InstrumentCommand:
        """Require exactly the supported instrument and action fields."""
        if (
            not isinstance(payload, dict)
            or set(payload) != {InstrumentField.INSTRUMENT, InstrumentField.ACTION}
            or not isinstance(payload[InstrumentField.INSTRUMENT], str)
            or payload[InstrumentField.INSTRUMENT] not in InstrumentName
            or not isinstance(payload[InstrumentField.ACTION], str)
            or payload[InstrumentField.ACTION] not in InstrumentAction
        ):
            raise InvalidInstrumentCommand(
                "Choose a centrifuge or refrigerator robot action: open, close, or open_close"
            )
        return cls(
            InstrumentName(payload[InstrumentField.INSTRUMENT]),
            InstrumentAction(payload[InstrumentField.ACTION]),
        )

    def snapshot(self) -> dict[str, str]:
        """Describe the selected operation for pause, resume and completion feedback."""
        return {
            InstrumentField.INSTRUMENT: self.instrument,
            InstrumentField.ACTION: self.action,
        }
