"""Engine-independent centrifuge command and snapshot vocabulary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import ClassVar

from cramera.laboratory_manual import CentrifugeManual


# %% command and observation fields
class CentrifugeAction(StrEnum):
    """Operations accepted by the laboratory instrument."""

    OPEN = "open"
    """Raise the lid after the rotor has stopped."""
    CLOSE = "close"
    """Lower the lid before operating the rotor."""
    SECURE = "secure"
    """Engage a fixture around an observed seated glass."""
    RELEASE = "release"
    """Release one stopped rotor fixture."""
    START = "start"
    """Run a balanced timed rotor cycle."""
    STOP = "stop"
    """Brake the rotor with bounded drive torque."""


class CentrifugeSlot(StrEnum):
    """Four individually addressable rotor holders."""

    C1 = "C1"
    """First rotor holder, opposed to the third."""
    C2 = "C2"
    """Second rotor holder, opposed to the fourth."""
    C3 = "C3"
    """Third rotor holder, opposed to the first."""
    C4 = "C4"
    """Fourth rotor holder, opposed to the second."""


class CentrifugeField(StrEnum):
    """Public command keys and measured instrument snapshot keys."""

    CENTRIFUGE = "centrifuge"
    """Instrument state within a simulation snapshot."""
    KNOWLEDGE = "knowledge"
    """Read operating manual, rule changes and measured start eligibility."""
    AVAILABLE = "available"
    """Whether the authored scene contains this instrument."""
    RPM = "rpm"
    """Measured or requested revolutions per minute."""
    TARGET_RPM = "targetRpm"
    """Requested rotor speed before acceleration limiting."""
    RELATIVE_FORCE = "rcf"
    """Radial acceleration relative to gravity."""
    LID_ANGLE = "lidAngle"
    """Measured opening angle of the articulated lid."""
    LID_OPEN = "lidOpen"
    """Whether the measured lid position permits loading."""
    LID_CLOSED = "lidClosed"
    """Whether the measured lid position permits rotation."""
    LOCKED = "locked"
    """Whether movement prohibits opening or releasing fixtures."""
    RUNNING = "running"
    """Whether a timed cycle is active."""
    BALANCED = "balanced"
    """Whether opposing secured samples have balanced mass."""
    MAXIMUM_RPM = "maximumRpm"
    """Largest permitted speed of this instrument."""
    SLOTS = "slots"
    """Measured rotor holders and their secured contents."""
    ID = "id"
    """Public identifier of one rotor holder."""
    OPPOSITE = "opposite"
    """Holder requiring a balancing sample."""
    POSITION = "position"
    """Measured desired glass-bottom position in world coordinates."""
    KEY = "key"
    """Authored glass identifier."""
    GRAMS = "grams"
    """Contained glass and liquid mass in grams."""
    SECURED = "secured"
    """Whether the holder fixture is engaged."""
    JOINTS = "joints"
    """Measured articulated angles for rendering."""
    REMAINING_SECONDS = "remainingSeconds"
    """Remaining timed cycle duration in simulation seconds."""
    ACTION = "action"
    """Requested instrument operation."""
    SLOT = "slot"
    """Holder selected for fixture operation."""
    DURATION = "duration"
    """Requested cycle duration in simulation seconds."""


@dataclass
class InvalidCentrifugeCommand(ValueError):
    """A command violates the public request schema."""

    reason: str
    """Validation failure exposed to the caller."""

    def __str__(self) -> str:
        """Return the request validation failure."""
        return self.reason


class ManualRequestField(StrEnum):
    """Accepted sources of a centrifuge operating document."""

    DOCUMENT = "document"
    """Complete Markdown document supplied for validation and reading."""
    EDITION = "edition"
    """Identifier of one bundled demonstration document."""


@dataclass(frozen=True)
class ManualRequest:
    """A complete validated document ready for serialized knowledge loading."""

    document: str
    """The operating document whose rules will replace current knowledge."""

    @classmethod
    def from_payload(cls, payload: dict) -> ManualRequest:
        """Require exactly one document source and validate it before mutation."""
        if not isinstance(payload, dict):
            raise InvalidCentrifugeCommand("Expected a manual document or edition")
        if set(payload) == {ManualRequestField.EDITION}:
            manual = CentrifugeManual.builtin(payload[ManualRequestField.EDITION])
        elif set(payload) == {ManualRequestField.DOCUMENT}:
            manual = CentrifugeManual.from_markdown(
                payload[ManualRequestField.DOCUMENT]
            )
        else:
            raise InvalidCentrifugeCommand(
                "Expected exactly one manual document or edition"
            )
        return cls(manual.document)


# %% validated command
@dataclass(frozen=True)
class CentrifugeCommand:
    """One instrument operation with validated optional cycle settings."""

    action: CentrifugeAction
    """Requested lid, fixture, or rotor operation."""
    slot: CentrifugeSlot | None = None
    """Holder involved in fixture engagement or release."""
    key: str | None = None
    """Existing glass requested for fixture engagement."""
    rpm: float | None = None
    """Requested rotor speed in revolutions per minute."""
    duration: float | None = None
    """Requested timed cycle length in simulation seconds."""

    MINIMUM_DURATION: ClassVar[float] = 1.0
    """Shortest accepted timed cycle in simulation seconds."""
    MAXIMUM_DURATION: ClassVar[float] = 300.0
    """Longest accepted timed cycle in simulation seconds."""

    @classmethod
    def from_payload(cls, payload: dict) -> CentrifugeCommand:
        """Reject unrecognized fields and malformed physical control parameters."""
        if (
            not isinstance(payload, dict)
            or payload.get(CentrifugeField.ACTION) not in CentrifugeAction
        ):
            raise InvalidCentrifugeCommand("A supported centrifuge action is required")
        action = CentrifugeAction(payload[CentrifugeField.ACTION])
        required = {CentrifugeField.ACTION}
        optional = set()
        if action in (CentrifugeAction.SECURE, CentrifugeAction.RELEASE):
            required.add(CentrifugeField.SLOT)
        if action == CentrifugeAction.SECURE:
            required.add(CentrifugeField.KEY)
        if action == CentrifugeAction.START:
            optional.update((CentrifugeField.RPM, CentrifugeField.DURATION))
        if not required <= payload.keys() or not payload.keys() <= required | optional:
            raise InvalidCentrifugeCommand("Unexpected or missing centrifuge fields")
        slot = payload.get(CentrifugeField.SLOT)
        if slot is not None and slot not in CentrifugeSlot:
            raise InvalidCentrifugeCommand("Unknown centrifuge slot")
        if CentrifugeField.SLOT in required and slot is None:
            raise InvalidCentrifugeCommand("A centrifuge slot is required")
        key = payload.get(CentrifugeField.KEY)
        if CentrifugeField.KEY in required and (
            not isinstance(key, str) or not key.strip()
        ):
            raise InvalidCentrifugeCommand("A glass key is required")
        rpm = payload.get(CentrifugeField.RPM)
        duration = payload.get(CentrifugeField.DURATION)
        for name in optional & payload.keys():
            value = payload[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise InvalidCentrifugeCommand(
                    f"{name} must be a finite positive number"
                )
        if (
            duration is not None
            and not cls.MINIMUM_DURATION <= duration <= cls.MAXIMUM_DURATION
        ):
            raise InvalidCentrifugeCommand(
                f"Duration must be between {cls.MINIMUM_DURATION:g} and {cls.MAXIMUM_DURATION:g} seconds"
            )
        return cls(
            action,
            CentrifugeSlot(slot) if slot is not None else None,
            key,
            rpm,
            duration,
        )
