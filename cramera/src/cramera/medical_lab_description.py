"""Validate authored instrument mechanisms in an expanded medical laboratory."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Any


# %% authored mechanism vocabulary
class MedicalLabField(StrEnum):
    """Fields shared by the expanded scene and its instrument controller."""

    INSTRUMENTS = "instruments"
    """Installed independent instruments and passive accessories."""
    ID = "id"
    """Resource identifier shared by files, metadata and command selections."""
    LABEL = "label"
    """English display name."""
    STATION = "station"
    """Authored workstation containing the instrument."""
    JOINTS = "joints"
    """Actuated scalar mechanisms of one instrument."""
    NAME = "name"
    """Fully prefixed URDF joint name."""
    TYPE = "type"
    """URDF motion kind."""
    REST = "rest"
    """Authored rest coordinate."""
    INSPECTION = "inspection"
    """Authored inspection coordinate."""
    LOWER = "lower"
    """Minimum finite-travel joint coordinate."""
    UPPER = "upper"
    """Maximum finite-travel joint coordinate."""
    VELOCITY = "velocity"
    """Maximum reference speed in SI units."""
    EFFORT = "effort"
    """Maximum motor force or torque in SI units."""


class MedicalLabJointKind(StrEnum):
    """Scalar URDF joints supported by the mechanism demonstrations."""

    REVOLUTE = "revolute"
    """Bounded rotary mechanism."""
    PRISMATIC = "prismatic"
    """Bounded translating mechanism."""
    CONTINUOUS = "continuous"
    """Rotary mechanism without positional stops."""


class InvalidMedicalLabDescription(ValueError):
    """Instrument metadata cannot safely identify bounded physical mechanisms."""


@dataclass(frozen=True)
class MedicalLabJoint:
    """Authored inspection targets and actuator limits for one scalar joint."""

    name: str
    """Fully prefixed joint name shared by URDF, physics and semantic worlds."""
    kind: MedicalLabJointKind
    """Rotary or translational URDF joint kind."""
    rest: float
    """Authored rest coordinate in radians or metres."""
    inspection: float
    """Authored inspection coordinate in radians or metres."""
    lower: float
    """Lower bound for a finite-travel joint."""
    upper: float
    """Upper bound for a finite-travel joint."""
    velocity: float
    """Maximum authored reference velocity in radians or metres per second."""
    effort: float = 20.0
    """Maximum motor effort in Newton-metres or Newtons."""

    def __post_init__(self) -> None:
        """Reject nonfinite targets and coordinates outside mechanical travel."""
        if not isinstance(self.kind, MedicalLabJointKind):
            raise InvalidMedicalLabDescription("Use a supported scalar joint kind")
        values = (
            self.rest,
            self.inspection,
            self.lower,
            self.upper,
            self.velocity,
            self.effort,
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ):
            raise InvalidMedicalLabDescription(
                f"{self.name}: joint parameters must be finite numbers"
            )
        if self.velocity <= 0 or self.effort <= 0:
            raise InvalidMedicalLabDescription(
                f"{self.name}: velocity and effort must be positive"
            )
        if self.kind != MedicalLabJointKind.CONTINUOUS:
            if self.lower >= self.upper or not all(
                self.lower <= value <= self.upper
                for value in (self.rest, self.inspection)
            ):
                raise InvalidMedicalLabDescription(
                    f"{self.name}: inspection and rest must lie within joint travel"
                )

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MedicalLabJoint:
        """Read one scalar joint from its explicit scene metadata."""
        required = {
            MedicalLabField.NAME,
            MedicalLabField.TYPE,
            MedicalLabField.REST,
            MedicalLabField.INSPECTION,
            MedicalLabField.LOWER,
            MedicalLabField.UPPER,
            MedicalLabField.VELOCITY,
        }
        if (
            not isinstance(data, dict)
            or not required <= data.keys()
            or data[MedicalLabField.TYPE] not in MedicalLabJointKind
        ):
            raise InvalidMedicalLabDescription(
                "Incomplete or unsupported instrument joint"
            )
        return cls(
            name=data[MedicalLabField.NAME],
            kind=MedicalLabJointKind(data[MedicalLabField.TYPE]),
            rest=data[MedicalLabField.REST],
            inspection=data[MedicalLabField.INSPECTION],
            lower=data[MedicalLabField.LOWER],
            upper=data[MedicalLabField.UPPER],
            velocity=data[MedicalLabField.VELOCITY],
            effort=data.get(MedicalLabField.EFFORT, 20.0),
        )


@dataclass(frozen=True)
class MedicalLabInstrument:
    """One installed resource and the mechanisms that belong exclusively to it."""

    id: str
    """Standalone resource identifier."""
    label: str
    """English display name."""
    station: str
    """Authored workstation identifier."""
    joints: tuple[MedicalLabJoint, ...]
    """Actuated coordinates; empty for passive accessories."""

    def __post_init__(self) -> None:
        """Reject ambiguous identifiers and mechanisms assigned to another model."""
        if (
            not isinstance(self.id, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", self.id) is None
        ):
            raise InvalidMedicalLabDescription(
                "Instrument identifiers must be local resource names"
            )
        if (
            not isinstance(self.label, str)
            or not self.label
            or not isinstance(self.station, str)
            or not self.station
        ):
            raise InvalidMedicalLabDescription(
                f"{self.id}: a label and station are required"
            )
        names = [joint.name for joint in self.joints]
        if len(set(names)) != len(names) or any(
            not isinstance(name, str)
            or not name.startswith(self.id + "_")
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name) is None
            for name in names
        ):
            raise InvalidMedicalLabDescription(
                f"{self.id}: joint names must be unique and belong to this resource"
            )

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MedicalLabInstrument:
        """Read an installed instrument without inventing controller targets."""
        required = {
            MedicalLabField.ID,
            MedicalLabField.LABEL,
            MedicalLabField.STATION,
            MedicalLabField.JOINTS,
        }
        if (
            not isinstance(data, dict)
            or not required <= data.keys()
            or not isinstance(data[MedicalLabField.JOINTS], list)
        ):
            raise InvalidMedicalLabDescription("Incomplete instrument description")
        return cls(
            data[MedicalLabField.ID],
            data[MedicalLabField.LABEL],
            data[MedicalLabField.STATION],
            tuple(
                MedicalLabJoint.from_json(joint)
                for joint in data[MedicalLabField.JOINTS]
            ),
        )


@dataclass(frozen=True)
class MedicalLabDescription:
    """The installed collection used to validate physical mechanism requests."""

    instruments: tuple[MedicalLabInstrument, ...]
    """Independently named resources in authored inspection order."""

    def __post_init__(self) -> None:
        """Require unique resource identities throughout the expanded room."""
        identifiers = [instrument.id for instrument in self.instruments]
        if len(set(identifiers)) != len(identifiers):
            raise InvalidMedicalLabDescription("Instrument identifiers must be unique")

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MedicalLabDescription:
        """Parse the medicalLab section while leaving room presentation metadata intact."""
        if not isinstance(data, dict) or not isinstance(
            data.get(MedicalLabField.INSTRUMENTS), list
        ):
            raise InvalidMedicalLabDescription(
                "The laboratory needs an instrument list"
            )
        return cls(
            tuple(
                MedicalLabInstrument.from_json(item)
                for item in data[MedicalLabField.INSTRUMENTS]
            )
        )

    def by_id(self, identifier: str) -> MedicalLabInstrument:
        """Resolve an installed instrument or reject an unknown resource."""
        for instrument in self.instruments:
            if instrument.id == identifier:
                return instrument
        raise InvalidMedicalLabDescription(f"Unknown instrument: {identifier}")
