"""Shared cabinet, door and sample-support geometry for refrigerated storage."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bundle import UrdfDescription


# %% instrument contract


class RefrigeratorPart(StrEnum):
    """Stable body, joint, mesh and collision identifiers."""

    BASE = "laboratory_refrigerator_base"
    """Floor-standing insulated cabinet."""
    DOOR = "laboratory_refrigerator_door"
    """Left-hinged glass door."""
    BASE_JOINT = "laboratory_refrigerator_base_joint"
    """Fixed attachment to the laboratory frame."""
    DOOR_JOINT = "laboratory_refrigerator_door_joint"
    """Bounded outward-opening door hinge."""
    BASE_ASSET = "refrigerator_base"
    """Cabinet material-bearing asset stem."""
    DOOR_ASSET = "refrigerator_door"
    """Door material-bearing asset stem."""
    DISPLAY = "Refrigerator display"
    """Fixed screen receiving the current storage temperature."""
    GLASS = "refrigerator_door_glass"
    """Physical glazing preventing access while the door is closed."""


class RefrigeratorField(StrEnum):
    """Authored frame and storage-target metadata."""

    BODY = "body"
    """Fixed cabinet link."""
    DOOR_BODY = "doorBody"
    """Moving glass-door link."""
    JOINT = "joint"
    """Cabinet attachment joint."""
    DOOR_JOINT = "doorJoint"
    """Door hinge joint."""
    ASSET = "asset"
    """Cabinet mesh path."""
    DOOR_ASSET = "doorAsset"
    """Door mesh path."""
    ORIGIN = "origin"
    """Cabinet floor origin in the laboratory frame."""
    DOOR_ORIGIN = "doorOrigin"
    """Left hinge origin relative to the cabinet."""
    DOOR_OPEN = "doorOpen"
    """Maximum positive opening angle."""
    DIMENSIONS = "dimensions"
    """Full cabinet width, depth and height."""
    DISPLAY_MESH = "displayMesh"
    """Authored live-display mesh name."""
    DISPLAY_POSITION = "displayPosition"
    """Display centre in the laboratory frame."""
    DISPLAY_SIZE = "displaySize"
    """Display width and height."""
    TARGET_TEMPERATURE = "targetTemperature"
    """Default storage target in degrees Celsius."""
    SLOTS = "slots"
    """Six independently addressable sample supports."""
    IDENTIFIER = "id"
    """Visible sample-support identifier."""
    POSITION = "position"
    """Supported tube-root target in the laboratory frame."""
    LOCAL_POSITION = "localPosition"
    """Supported tube-root target relative to the cabinet."""


@dataclass(frozen=True)
class RefrigeratorBox:
    """One collision primitive expressed in its supporting link frame."""

    name: str
    """Stable collision identifier."""
    center: tuple[float, float, float]
    """Primitive centre."""
    size: tuple[float, float, float]
    """Full width, depth and height."""
    yaw: float = 0.0
    """Rotation about the vertical axis."""


@dataclass(frozen=True)
class RefrigeratorDimensions:
    """Compact insulated cabinet accommodating existing upright glass tubes."""

    origin: tuple[float, float, float] = (-1.23, 0.0, 0.0)
    """Cabinet floor centre beside the laboratory bench."""
    width: float = 0.54
    """External cabinet width."""
    depth: float = 0.58
    """External cabinet depth."""
    height: float = 0.84
    """External cabinet height."""
    door_origin: tuple[float, float, float] = (-0.25, -0.31, 0.43)
    """Vertical hinge on the front-left cabinet edge."""
    door_open: float = 1.75
    """Positive travel about the negative vertical axis."""
    shelf_heights: tuple[float, ...] = (0.24, 0.48)
    """Upper support surfaces of the two removable shelves."""
    slot_columns: tuple[float, ...] = (-0.14, 0.0, 0.14)
    """Three sample centres across each shelf."""
    slot_depth: float = -0.10
    """Sample centre depth relative to the cabinet."""
    collar_inner_radius: float = 0.0115
    """Clearance inside each open sample collar."""
    collar_outer_radius: float = 0.0185
    """Outer sample collar radius."""
    collar_height: float = 0.04
    """Collar wall height above the shelf."""
    collar_segments: int = 16
    """Number of tangent collision boxes surrounding an open collar."""
    display: tuple[float, float, float] = (0.09, -0.2932, 0.775)
    """Front-facing screen centre in the fixed cabinet header."""
    display_size: tuple[float, float] = (0.145, 0.036)
    """Screen width and height."""
    target_temperature: float = 4.0
    """Default storage temperature in degrees Celsius."""

    def slot_positions(self) -> tuple[tuple[float, float, float], ...]:
        """List supported tube roots in lower-then-upper shelf order."""
        return tuple(
            (column, self.slot_depth, height)
            for height in self.shelf_heights
            for column in self.slot_columns
        )

    def world_position(self, position: tuple[float, float, float]) -> list[float]:
        """Translate a cabinet-local point to the laboratory frame."""
        return [self.origin[axis] + position[axis] for axis in range(3)]

    def cabinet_boxes(self) -> tuple[RefrigeratorBox, ...]:
        """Describe the open chamber walls and two load-bearing shelves."""
        return (
            RefrigeratorBox(
                "refrigerator_left_wall",
                (-0.2525, 0.0, 0.43),
                (0.035, self.depth, 0.82),
            ),
            RefrigeratorBox(
                "refrigerator_right_wall",
                (0.2525, 0.0, 0.43),
                (0.035, self.depth, 0.82),
            ),
            RefrigeratorBox(
                "refrigerator_rear_wall", (0.0, 0.27, 0.43), (0.47, 0.04, 0.82)
            ),
            RefrigeratorBox(
                "refrigerator_compressor", (0.0, 0.0, 0.08), (0.47, self.depth, 0.12)
            ),
            RefrigeratorBox(
                "refrigerator_header", (0.0, 0.0, 0.78), (0.47, self.depth, 0.12)
            ),
            *(
                RefrigeratorBox(
                    f"refrigerator_shelf_{index + 1}",
                    (0.0, -0.005, height - 0.006),
                    (0.454, 0.47, 0.012),
                )
                for index, height in enumerate(self.shelf_heights)
            ),
        )

    def collar_boxes(self) -> tuple[RefrigeratorBox, ...]:
        """Form six hollow collars without filling their sample apertures."""
        radius = (self.collar_inner_radius + self.collar_outer_radius) / 2
        size = (
            self.collar_outer_radius - self.collar_inner_radius,
            2 * self.collar_outer_radius * math.tan(math.pi / self.collar_segments),
            self.collar_height,
        )
        boxes = []
        for slot_index, (horizontal, depth, height) in enumerate(self.slot_positions()):
            for index in range(self.collar_segments):
                angle = math.tau * index / self.collar_segments
                boxes.append(
                    RefrigeratorBox(
                        f"refrigerator_R{slot_index + 1}_collar_{index}",
                        (
                            horizontal + radius * math.cos(angle),
                            depth + radius * math.sin(angle),
                            height + self.collar_height / 2,
                        ),
                        size,
                        angle,
                    )
                )
        return tuple(boxes)

    def door_boxes(self) -> tuple[RefrigeratorBox, ...]:
        """Describe the physical glazing and four enclosing frame rails."""
        return (
            RefrigeratorBox(RefrigeratorPart.GLASS, (0.25, 0, 0), (0.418, 0.006, 0.49)),
            RefrigeratorBox(
                "refrigerator_door_left", (0.018, 0, 0), (0.036, 0.032, 0.58)
            ),
            RefrigeratorBox(
                "refrigerator_door_right", (0.482, 0, 0), (0.036, 0.032, 0.58)
            ),
            RefrigeratorBox(
                "refrigerator_door_top", (0.25, 0, 0.2675), (0.428, 0.032, 0.045)
            ),
            RefrigeratorBox(
                "refrigerator_door_bottom", (0.25, 0, -0.2675), (0.428, 0.032, 0.045)
            ),
            RefrigeratorBox(
                "refrigerator_handle", (0.482, -0.044, 0.0), (0.018, 0.018, 0.203)
            ),
            *(
                RefrigeratorBox(
                    f"refrigerator_handle_mount_{index}",
                    (0.482, -0.026, elevation),
                    (0.016, 0.035, 0.016),
                )
                for index, elevation in enumerate((-0.085, 0.085))
            ),
        )

    def metadata(self) -> dict[str, object]:
        """Expose authored frames, screen dimensions and six sample targets."""
        return {
            RefrigeratorField.BODY: RefrigeratorPart.BASE,
            RefrigeratorField.DOOR_BODY: RefrigeratorPart.DOOR,
            RefrigeratorField.JOINT: RefrigeratorPart.BASE_JOINT,
            RefrigeratorField.DOOR_JOINT: RefrigeratorPart.DOOR_JOINT,
            RefrigeratorField.ASSET: f"assets/{RefrigeratorPart.BASE_ASSET}.glb",
            RefrigeratorField.DOOR_ASSET: f"assets/{RefrigeratorPart.DOOR_ASSET}.glb",
            RefrigeratorField.ORIGIN: list(self.origin),
            RefrigeratorField.DOOR_ORIGIN: list(self.door_origin),
            RefrigeratorField.DOOR_OPEN: self.door_open,
            RefrigeratorField.DIMENSIONS: [self.width, self.depth, self.height],
            RefrigeratorField.DISPLAY_MESH: RefrigeratorPart.DISPLAY,
            RefrigeratorField.DISPLAY_POSITION: self.world_position(self.display),
            RefrigeratorField.DISPLAY_SIZE: list(self.display_size),
            RefrigeratorField.TARGET_TEMPERATURE: self.target_temperature,
            RefrigeratorField.SLOTS: [
                {
                    RefrigeratorField.IDENTIFIER: f"R{index + 1}",
                    RefrigeratorField.POSITION: self.world_position(position),
                    RefrigeratorField.LOCAL_POSITION: list(position),
                }
                for index, position in enumerate(self.slot_positions())
            ],
        }


# %% physical articulation


@dataclass
class RefrigeratorDescription:
    """An accessible cabinet with a separately articulated glazed door."""

    description: UrdfDescription
    """Environment receiving the model."""
    dimensions: RefrigeratorDimensions = RefrigeratorDimensions()
    """Shared authored measurements."""

    def build(self) -> None:
        """Attach the insulated housing, supported shelves and bounded glass door."""
        dimensions = self.dimensions
        base = self.description.link(
            RefrigeratorPart.BASE, f"assets/{RefrigeratorPart.BASE_ASSET}.glb"
        )
        self.description.joint(
            RefrigeratorPart.BASE_JOINT,
            "laboratory_root",
            RefrigeratorPart.BASE,
            dimensions.origin,
        )
        for box in (*dimensions.cabinet_boxes(), *dimensions.collar_boxes()):
            self.description.box(base, box.name, box.center, box.size, box.yaw)
        door = self.description.link(
            RefrigeratorPart.DOOR, f"assets/{RefrigeratorPart.DOOR_ASSET}.glb"
        )
        for box in dimensions.door_boxes():
            self.description.box(door, box.name, box.center, box.size, box.yaw)
        self.description.inertia(door, 3.0, (0.25, 0, 0), (0.5, 0.032, 0.58))
        joint = ElementTree.SubElement(
            self.description.root,
            "joint",
            name=RefrigeratorPart.DOOR_JOINT,
            type="revolute",
        )
        ElementTree.SubElement(joint, "parent", link=RefrigeratorPart.BASE)
        ElementTree.SubElement(joint, "child", link=RefrigeratorPart.DOOR)
        ElementTree.SubElement(
            joint,
            "origin",
            xyz=self.description.coordinates(dimensions.door_origin),
            rpy="0 0 0",
        )
        ElementTree.SubElement(joint, "axis", xyz="0 0 -1")
        ElementTree.SubElement(
            joint,
            "limit",
            lower="0",
            upper=str(dimensions.door_open),
            effort="15",
            velocity="1",
        )
        ElementTree.SubElement(joint, "dynamics", damping="0.2", friction="0.04")
