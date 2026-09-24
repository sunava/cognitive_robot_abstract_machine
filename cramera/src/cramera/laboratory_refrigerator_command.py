"""Engine-independent sample refrigerator commands and observation fields."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


# %% command and observation vocabulary
class RefrigeratorAction(StrEnum):
    """Operations permitted on the articulated sample cabinet."""

    OPEN = "open"
    """Swing the glass door towards its mechanical opening limit."""
    CLOSE = "close"
    """Return the door to the closed position using bounded torque."""


class RefrigeratorField(StrEnum):
    """Fields shared by authored geometry, commands and measured cabinet state."""

    REFRIGERATOR = "refrigerator"
    """Instrument state within a laboratory snapshot."""
    ACTION = "action"
    """Requested door operation."""
    AVAILABLE = "available"
    """Whether the current scene contains an articulated refrigerator."""
    BODY = "body"
    """Stationary cabinet body supporting the samples."""
    DOOR_BODY = "doorBody"
    """Articulated glass door body."""
    DOOR_JOINT = "doorJoint"
    """Authored door hinge name."""
    DOOR_ANGLE = "doorAngle"
    """Measured door opening angle in radians."""
    DOOR_OPEN = "doorOpen"
    """Authored opening limit, or measured accessibility in a snapshot."""
    DOOR_CLOSED = "doorClosed"
    """Whether the measured hinge is near its closed configuration."""
    TARGET_TEMPERATURE = "targetTemperatureC"
    """Cabinet cooling target in degrees Celsius."""
    AUTHORED_TEMPERATURE = "targetTemperature"
    """Cooling target supplied by the authored cabinet metadata."""
    TEMPERATURE = "temperatureC"
    """Reduced-model cabinet temperature in degrees Celsius."""
    TEMPERATURE_MODEL = "temperatureModel"
    """Provenance of the displayed temperature."""
    SLOTS = "slots"
    """Supported sample positions and their observed occupants."""
    ID = "id"
    """Human-readable storage position identifier."""
    POSITION = "position"
    """Desired vessel-root support position in world coordinates."""
    KEY = "key"
    """Authored sample identifier, or null for an unoccupied position."""
    JOINTS = "joints"
    """Measured hinge positions for rendering."""


class RefrigeratorTemperatureModel(StrEnum):
    """Provenance of the cabinet temperature observation."""

    SIMULATED = "simulated"
    """A reduced heat-exchange model driven by simulated time and door angle."""


@dataclass
class InvalidRefrigeratorCommand(ValueError):
    """A request violates the refrigerator command schema."""

    reason: str
    """Validation failure exposed to the operator."""

    def __str__(self) -> str:
        """Return the rejected command's validation explanation."""
        return self.reason


@dataclass(frozen=True)
class RefrigeratorCommand:
    """One validated door operation without geometry or pose overrides."""

    action: RefrigeratorAction
    """Requested door operation."""

    @classmethod
    def from_payload(cls, payload: dict) -> RefrigeratorCommand:
        """Require exactly one supported operation before mutating the instrument."""
        if (
            not isinstance(payload, dict)
            or set(payload) != {RefrigeratorField.ACTION}
            or not isinstance(payload[RefrigeratorField.ACTION], str)
            or payload[RefrigeratorField.ACTION] not in RefrigeratorAction
        ):
            raise InvalidRefrigeratorCommand(
                "Expected a refrigerator open or close operation"
            )
        return cls(RefrigeratorAction(payload[RefrigeratorField.ACTION]))
