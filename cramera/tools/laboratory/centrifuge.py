"""Shared dimensions and articulated collision description for the centrifuge."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bundle import UrdfDescription


# %% instrument contract


class CentrifugePart(StrEnum):
    """Names shared by authored bodies, joints and generated scene metadata."""

    BASE = "laboratory_centrifuge_base"
    """Stationary housing attached to the bench."""
    ROTOR = "laboratory_centrifuge_rotor"
    """Driven assembly carrying four tube holders."""
    LID = "laboratory_centrifuge_lid"
    """Rear-hinged protective cover."""
    BASE_JOINT = "laboratory_centrifuge_base_joint"
    """Fixed attachment of the instrument."""
    ROTOR_JOINT = "laboratory_centrifuge_rotor_joint"
    """Continuous vertical drive axis."""
    LID_JOINT = "laboratory_centrifuge_lid_joint"
    """Bounded opening hinge."""
    BASE_ASSET = "centrifuge_base"
    """Housing visual asset stem."""
    ROTOR_ASSET = "centrifuge_rotor"
    """Rotor visual asset stem."""
    LID_ASSET = "centrifuge_lid"
    """Cover visual asset stem."""


class CentrifugeField(StrEnum):
    """Scene metadata fields describing the instrument's authored frames."""

    BODY = "body"
    """Stationary housing link."""
    ROTOR_BODY = "rotorBody"
    """Moving rotor link."""
    LID_BODY = "lidBody"
    """Moving cover link."""
    ROTOR_JOINT = "rotorJoint"
    """Continuous drive joint."""
    LID_JOINT = "lidJoint"
    """Bounded cover hinge."""
    ORIGIN = "origin"
    """World translation of the stationary housing."""
    ROTOR_ORIGIN = "rotorOrigin"
    """Drive origin relative to the housing."""
    LID_ORIGIN = "lidOrigin"
    """Hinge origin relative to the housing."""
    OPEN_ANGLE = "openAngle"
    """Cover opening limit in radians."""
    MAXIMUM_RPM = "maximumRpm"
    """Configured maximum drive speed in revolutions per minute."""
    CHAMBER_RADIUS = "chamberRadius"
    """Radial clearance of the open stationary chamber."""
    CHAMBER_HEIGHT = "chamberHeight"
    """Rim elevation relative to the rotor origin."""
    SLOTS = "slots"
    """Ordered sample-holder targets."""
    IDENTIFIER = "id"
    """Visible sample-holder identifier."""
    OPPOSITE = "opposite"
    """Identifier of the opposing sample holder."""
    POSITION = "position"
    """Tube root position relative to the rotor."""
    WORLD_POSITION = "worldPosition"
    """Initial tube root position in the laboratory frame."""


@dataclass(frozen=True)
class CentrifugeDimensions:
    """Metric dimensions accommodating the laboratory's existing tubes."""

    origin: tuple[float, float, float] = (-0.53, -0.18, 0.9)
    """Stationary housing origin on the laboratory work surface."""
    width: float = 0.34
    """Full housing width."""
    depth: float = 0.32
    """Full housing depth."""
    chamber_radius: float = 0.137
    """Clear internal radius of the stationary bowl."""
    chamber_outer_radius: float = 0.15
    """Outer radius of the housing's circular upper shell."""
    chamber_bottom: float = 0.032
    """Height of the chamber floor above the work surface."""
    chamber_top: float = 0.213
    """Upper housing rim height above the work surface."""
    rotor_origin: tuple[float, float, float] = (0.0, 0.0, 0.052)
    """Drive-axis origin relative to the stationary housing."""
    lid_origin: tuple[float, float, float] = (0.0, 0.155, 0.223)
    """Hinge origin relative to the stationary housing."""
    open_angle: float = 1.35
    """Positive opening angle around the negative horizontal axis."""
    handle_center: tuple[float, float, float] = (0.0, -0.210, 0.090)
    """Raised grip centre providing fingertip clearance above the closed cover."""
    handle_size: tuple[float, float, float] = (0.069, 0.019, 0.020)
    """Full dimensions of the horizontal grip bar."""
    handle_mounts: tuple[tuple[float, float, float], ...] = (
        (-0.026, -0.210, 0.047),
        (0.026, -0.210, 0.047),
    )
    """Two supports joining the raised handle to the protective cover."""
    handle_mount_size: tuple[float, float, float] = (0.014, 0.016, 0.086)
    """Full dimensions of each handle support."""
    slot_radius: float = 0.072
    """Distance from the rotor axis to each sample centre."""
    slot_bottom: float = 0.008
    """Tube-root support height relative to the rotor origin."""
    slot_inner_radius: float = 0.0115
    """Minimum free radius inside each vertical sample sleeve."""
    slot_outer_radius: float = 0.016
    """Outside radius of each sample sleeve."""
    slot_height: float = 0.115
    """Sleeve height above its bottom support."""
    slot_count: int = 4
    """Number of evenly spaced sample holders."""
    maximum_revolutions_per_minute: float = 300.0
    """Maximum speed configured for the interactive contact-physics demo."""
    ring_segments: int = 24
    """Tangent collision boxes used for the stationary chamber."""
    sleeve_segments: int = 16
    """Tangent collision boxes used for each sample sleeve."""

    def slot_position(self, index: int) -> tuple[float, float, float]:
        """Return a zero-based sample position in the rotating frame."""
        angle = math.tau * index / self.slot_count
        return (
            self.slot_radius * math.cos(angle),
            self.slot_radius * math.sin(angle),
            self.slot_bottom,
        )

    def metadata(self) -> dict[str, object]:
        """Describe articulated frames and sample targets for runtime controls."""
        slots = []
        for index in range(self.slot_count):
            position = self.slot_position(index)
            slots.append(
                {
                    CentrifugeField.IDENTIFIER: f"C{index + 1}",
                    CentrifugeField.OPPOSITE: f"C{(index + self.slot_count // 2) % self.slot_count + 1}",
                    CentrifugeField.POSITION: list(position),
                    CentrifugeField.WORLD_POSITION: [
                        self.origin[axis] + self.rotor_origin[axis] + position[axis]
                        for axis in range(3)
                    ],
                }
            )
        return {
            CentrifugeField.BODY: CentrifugePart.BASE,
            CentrifugeField.ROTOR_BODY: CentrifugePart.ROTOR,
            CentrifugeField.LID_BODY: CentrifugePart.LID,
            CentrifugeField.ROTOR_JOINT: CentrifugePart.ROTOR_JOINT,
            CentrifugeField.LID_JOINT: CentrifugePart.LID_JOINT,
            CentrifugeField.ORIGIN: list(self.origin),
            CentrifugeField.ROTOR_ORIGIN: list(self.rotor_origin),
            CentrifugeField.LID_ORIGIN: list(self.lid_origin),
            CentrifugeField.OPEN_ANGLE: self.open_angle,
            CentrifugeField.MAXIMUM_RPM: self.maximum_revolutions_per_minute,
            CentrifugeField.CHAMBER_RADIUS: self.chamber_radius,
            CentrifugeField.CHAMBER_HEIGHT: self.chamber_top - self.rotor_origin[2],
            CentrifugeField.SLOTS: slots,
        }


# %% collision and articulation description


@dataclass
class CentrifugeDescription:
    """Open instrument geometry with separately actuated rotor and lid."""

    description: UrdfDescription
    """Environment description receiving the instrument."""
    dimensions: CentrifugeDimensions = CentrifugeDimensions()
    """Shared metric dimensions for both visual and collision bodies."""

    def ring(
        self,
        link: ElementTree.Element,
        name: str,
        center: tuple[float, float, float],
        inner_radius: float,
        outer_radius: float,
        height: float,
        segments: int,
    ) -> None:
        """Form an open annular collision wall from overlapping tangent boxes."""
        radius = (inner_radius + outer_radius) / 2
        size = (
            outer_radius - inner_radius,
            2 * outer_radius * math.tan(math.pi / segments),
            height,
        )
        for index in range(segments):
            angle = math.tau * index / segments
            self.description.box(
                link,
                f"{name}_{index}",
                (
                    center[0] + radius * math.cos(angle),
                    center[1] + radius * math.sin(angle),
                    center[2],
                ),
                size,
                angle,
            )

    def hinge(
        self,
        name: CentrifugePart,
        child: CentrifugePart,
        origin: tuple[float, float, float],
        axis: tuple[float, float, float],
        travel: float | None,
    ) -> None:
        """Attach an explicitly inertial rotating body to the fixed housing."""
        joint = ElementTree.SubElement(
            self.description.root,
            "joint",
            name=name,
            type="continuous" if travel is None else "revolute",
        )
        ElementTree.SubElement(joint, "parent", link=CentrifugePart.BASE)
        ElementTree.SubElement(joint, "child", link=child)
        ElementTree.SubElement(
            joint, "origin", xyz=self.description.coordinates(origin), rpy="0 0 0"
        )
        ElementTree.SubElement(joint, "axis", xyz=self.description.coordinates(axis))
        if travel is not None:
            ElementTree.SubElement(
                joint, "limit", lower="0", upper=str(travel), effort="8", velocity="1"
            )
        ElementTree.SubElement(joint, "dynamics", damping="0.03", friction="0.002")

    def build(self) -> None:
        """Build the housing, supported open sleeves, and hinged protective cover."""
        dimensions = self.dimensions
        base = self.description.link(
            CentrifugePart.BASE, f"assets/{CentrifugePart.BASE_ASSET}.glb"
        )
        self.description.joint(
            CentrifugePart.BASE_JOINT,
            "laboratory_root",
            CentrifugePart.BASE,
            dimensions.origin,
        )
        self.description.box(
            base,
            "chamber_floor",
            (0.0, 0.0, dimensions.chamber_bottom / 2),
            (dimensions.width, dimensions.depth, dimensions.chamber_bottom),
        )
        self.ring(
            base,
            "chamber_wall",
            (0.0, 0.0, (dimensions.chamber_bottom + dimensions.chamber_top) / 2),
            dimensions.chamber_radius,
            dimensions.chamber_outer_radius,
            dimensions.chamber_top - dimensions.chamber_bottom,
            dimensions.ring_segments,
        )
        rotor = self.description.link(
            CentrifugePart.ROTOR, f"assets/{CentrifugePart.ROTOR_ASSET}.glb"
        )
        self.hinge(
            CentrifugePart.ROTOR_JOINT,
            CentrifugePart.ROTOR,
            dimensions.rotor_origin,
            (0.0, 0.0, 1.0),
            None,
        )
        self.description.cylinder(rotor, "rotor_plate", 0.111, 0.006, 0.003)
        self.description.cylinder(rotor, "rotor_hub", 0.023, 0.036, 0.018)
        for index in range(dimensions.slot_count):
            position = dimensions.slot_position(index)
            self.description.box(
                rotor,
                f"slot_{index + 1}_floor",
                (position[0], position[1], dimensions.slot_bottom - 0.003),
                (
                    2 * dimensions.slot_outer_radius,
                    2 * dimensions.slot_outer_radius,
                    0.006,
                ),
            )
            self.ring(
                rotor,
                f"slot_{index + 1}_wall",
                (
                    position[0],
                    position[1],
                    dimensions.slot_bottom + dimensions.slot_height / 2,
                ),
                dimensions.slot_inner_radius,
                dimensions.slot_outer_radius,
                dimensions.slot_height,
                dimensions.sleeve_segments,
            )
        self.description.inertia(rotor, 0.40, (0.0, 0.0, 0.035), (0.222, 0.222, 0.123))
        lid = self.description.link(
            CentrifugePart.LID, f"assets/{CentrifugePart.LID_ASSET}.glb"
        )
        self.hinge(
            CentrifugePart.LID_JOINT,
            CentrifugePart.LID,
            dimensions.lid_origin,
            (-1.0, 0.0, 0.0),
            dimensions.open_angle,
        )
        self.description.box(
            lid,
            "lid_cover",
            (0.0, -dimensions.lid_origin[1], 0.007),
            (0.30, 0.30, 0.014),
        )
        self.description.box(
            lid,
            "lid_handle",
            dimensions.handle_center,
            dimensions.handle_size,
        )
        for index, position in enumerate(dimensions.handle_mounts):
            self.description.box(
                lid, f"lid_handle_mount_{index}", position, dimensions.handle_mount_size
            )
        self.description.inertia(
            lid, 0.55, (0.0, -dimensions.lid_origin[1], 0.007), (0.30, 0.30, 0.014)
        )
