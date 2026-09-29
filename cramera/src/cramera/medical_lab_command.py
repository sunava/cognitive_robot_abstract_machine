"""Bounded operator requests for native laboratory mechanism demonstrations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Any


# %% authenticated command contract
class MedicalLabAction(StrEnum):
    """Operator operations accepted by the expanded instrument controller."""

    RUN = "run"
    """Run or resume the selected authored inspection sequence."""
    OPEN = "open"
    """Move the selected device to its inspection joint coordinates."""
    CLOSE = "close"
    """Move the selected device to its rest joint coordinates."""
    PAUSE = "pause"
    """Retain the current operation while pausing native execution."""
    RESET = "reset"
    """Cancel the native plan before restoring authored mechanism coordinates."""


class MedicalLabSelection(StrEnum):
    """Selection that visits every installed actuated instrument."""

    ALL = "all"
    """Every installed resource with at least one moving joint."""


class MedicalLabCommandField(StrEnum):
    """Exact JSON command keys; arbitrary joint coordinates are never accepted."""

    ACTION = "action"
    """Requested lifecycle or target-selection operation."""
    DEVICE = "device"
    """Installed resource identifier or complete-collection selection."""


class InvalidMedicalLabCommand(ValueError):
    """The request is not a supported, explicitly bounded mechanism operation."""


@dataclass(frozen=True)
class MedicalLabCommand:
    """One validated selection or lifecycle command independent of physics."""

    action: MedicalLabAction
    """Requested mechanism operation or lifecycle change."""
    device: str | None = None
    """Installed resource or all, absent from pause and reset requests."""

    def __post_init__(self) -> None:
        """Keep programmatic construction as strict as remote command parsing."""
        if not isinstance(self.action, MedicalLabAction):
            raise InvalidMedicalLabCommand("Choose a supported instrument action")
        if self.action in (MedicalLabAction.PAUSE, MedicalLabAction.RESET):
            if self.device is not None:
                raise InvalidMedicalLabCommand("Pause and reset do not accept a device")
            return
        if (
            not isinstance(self.device, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", self.device) is None
        ):
            raise InvalidMedicalLabCommand("Select an installed instrument")
        if (
            self.device == MedicalLabSelection.ALL
            and self.action != MedicalLabAction.RUN
        ):
            raise InvalidMedicalLabCommand(
                "Only a complete inspection may select all instruments"
            )

    @classmethod
    def from_payload(cls, payload: Any) -> MedicalLabCommand:
        """Reject extra keys, arbitrary targets and unsupported operation names."""
        if (
            not isinstance(payload, dict)
            or payload.get(MedicalLabCommandField.ACTION) not in MedicalLabAction
        ):
            raise InvalidMedicalLabCommand("Choose run, open, close, pause or reset")
        action = MedicalLabAction(payload[MedicalLabCommandField.ACTION])
        required = {MedicalLabCommandField.ACTION}
        if action not in (MedicalLabAction.PAUSE, MedicalLabAction.RESET):
            required.add(MedicalLabCommandField.DEVICE)
        if payload.keys() != required:
            raise InvalidMedicalLabCommand(
                "Instrument command has missing or unexpected fields"
            )
        return cls(action, payload.get(MedicalLabCommandField.DEVICE))
