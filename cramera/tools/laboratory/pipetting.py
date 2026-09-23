"""Shared placement and contact geometry for the dispensing instrument."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bundle import UrdfDescription


# %% authored instrument contract


class PipettingPart(StrEnum):
    """Stable names identifying the stationary instrument and its asset."""

    BODY = "laboratory_pipetting_station"
    """Fixed instrument frame."""
    JOINT = "laboratory_pipetting_station_joint"
    """Attachment to the laboratory root."""
    ASSET = "pipetting_station"
    """Material-bearing visual asset stem."""
    PLATFORM = "pipetting_platform"
    """Load-bearing vessel support."""
    COLUMN = "pipetting_column"
    """Rear housing carrying the reservoirs and cantilever."""
    HEAD = "pipetting_head"
    """Cantilever supporting the independent feed nozzles."""
    DISPLAY = "Pipetting display"
    """Front-facing blank screen receiving runtime information."""


class PipettingField(StrEnum):
    """Metadata shared by visual placement and instrument control."""

    BODY = "body"
    """Stationary instrument link."""
    JOINT = "joint"
    """Fixed attachment joint."""
    ASSET = "asset"
    """Relative material-bearing mesh path."""
    ORIGIN = "origin"
    """Instrument origin in the laboratory frame."""
    TARGET_POSITION = "targetPosition"
    """Receiving vessel root in the laboratory frame."""
    NOZZLE_POSITION = "nozzlePosition"
    """Feed outlet in the laboratory frame."""
    DISPLAY_POSITION = "displayPosition"
    """Screen centre in the laboratory frame."""
    DISPLAY_SIZE = "displaySize"
    """Screen width and height."""
    DISPLAY_MESH = "displayMesh"
    """Mesh receiving the runtime display texture."""
    CHANNELS = "channels"
    """Independent feeds and their visual geometry."""
    IDENTIFIER = "id"
    """Reagent channel identifier."""
    LIQUID_MESH = "liquidMesh"
    """Reservoir liquid mesh whose height follows inventory."""
    LIQUID_BOTTOM = "liquidBottom"
    """Reservoir liquid lower elevation in the instrument frame."""
    LIQUID_HEIGHT = "liquidHeight"
    """Authored reservoir liquid height before level scaling."""


class PipettingChannel(StrEnum):
    """Independent reagent feeds in left-to-right order."""

    AMBER = "amber"
    """Left amber reagent reservoir and feed nozzle."""
    TEAL = "teal"
    """Right teal reagent reservoir and feed nozzle."""


@dataclass(frozen=True)
class PipettingSupport:
    """One rectangular guide forming the front-open receiving cradle."""

    name: str
    """Stable collision and visual part name."""
    center: tuple[float, float, float]
    """Guide centre in the instrument frame."""
    size: tuple[float, float, float]
    """Full guide extents along the instrument axes."""


@dataclass(frozen=True)
class PipettingDimensions:
    """Metre dimensions leaving the front grasp approach unobstructed."""

    origin: tuple[float, float, float] = (-0.10, 0.0, 0.9)
    """Station frame on the work surface."""
    platform_center: tuple[float, float, float] = (0.0, -0.015, 0.008)
    """Centre of the low load-bearing base."""
    platform_size: tuple[float, float, float] = (0.22, 0.29, 0.016)
    """Width, depth and thickness of the base."""
    column_center: tuple[float, float, float] = (0.0, 0.085, 0.133)
    """Centre of the rear instrument housing."""
    column_size: tuple[float, float, float] = (0.22, 0.09, 0.234)
    """Width, depth and height of the rear housing."""
    head_center: tuple[float, float, float] = (0.0, -0.032, 0.219)
    """Centre of the elevated dispensing cantilever."""
    head_size: tuple[float, float, float] = (0.054, 0.194, 0.024)
    """Width, reach and height of the cantilever."""
    target: tuple[float, float, float] = (0.0, -0.11, 0.016)
    """Tube root location on the platform."""
    nozzle: tuple[float, float, float] = (0.0, -0.11, 0.180)
    """Centre between the two independent nozzle outlets."""
    nozzle_offset: float = 0.0018
    """Horizontal offset of each outlet from the tube centre."""
    nozzle_radius: float = 0.0008
    """Outside radius of each stainless dispensing needle."""
    nozzle_length: float = 0.027
    """Needle length below the cantilever."""
    holder_inner_radius: float = 0.0115
    """Minimum clearance from the target centre to each support guide."""
    holder_outer_radius: float = 0.016
    """Outside distance of the vessel support guides."""
    holder_height: float = 0.032
    """Support height above the platform."""
    holder_depth: float = 0.045
    """Length of the two parallel side guides."""
    reservoir_horizontal_offset: float = 0.056
    """Horizontal spacing of each reservoir from the column centre."""
    reservoir_bottom: float = 0.254
    """Base elevation of both reservoir bottles."""
    reservoir_liquid_bottom: float = 0.259
    """Lower elevation of the independently named liquid meshes."""
    reservoir_liquid_height: float = 0.069
    """Authored liquid height at the initial reservoir fill."""
    display: tuple[float, float, float] = (-0.054, 0.0387, 0.156)
    """Centre of the front-facing screen."""
    display_size: tuple[float, float] = (0.078, 0.028)
    """Screen width and height in the instrument frame."""

    def world_position(self, position: tuple[float, float, float]) -> list[float]:
        """Translate an instrument-local position into the laboratory frame."""
        return [self.origin[axis] + position[axis] for axis in range(3)]

    def channel_nozzle(self, channel: PipettingChannel) -> tuple[float, float, float]:
        """Return the outlet position belonging to one independent feed."""
        sign = -1 if channel is PipettingChannel.AMBER else 1
        return (sign * self.nozzle_offset, self.nozzle[1], self.nozzle[2])

    def holder_guides(self) -> tuple[PipettingSupport, ...]:
        """Return two side guides and a backstop with a fully open front."""
        offset = (self.holder_inner_radius + self.holder_outer_radius) / 2
        thickness = self.holder_outer_radius - self.holder_inner_radius
        elevation = self.target[2] + self.holder_height / 2
        return (
            *(
                PipettingSupport(
                    f"pipetting_holder_side_{index}",
                    (
                        self.target[0] + sign * offset,
                        self.target[1]
                        + self.holder_outer_radius
                        - self.holder_depth / 2,
                        elevation,
                    ),
                    (thickness, self.holder_depth, self.holder_height),
                )
                for index, sign in enumerate((-1, 1))
            ),
            PipettingSupport(
                "pipetting_holder_back",
                (self.target[0], self.target[1] + offset, elevation),
                (self.holder_outer_radius * 2, thickness, self.holder_height),
            ),
        )

    def metadata(self) -> dict[str, object]:
        """Describe fixed frames, receiving position, and independently named feeds."""
        return {
            PipettingField.BODY: PipettingPart.BODY,
            PipettingField.JOINT: PipettingPart.JOINT,
            PipettingField.ASSET: f"assets/{PipettingPart.ASSET}.glb",
            PipettingField.ORIGIN: list(self.origin),
            PipettingField.TARGET_POSITION: self.world_position(self.target),
            PipettingField.NOZZLE_POSITION: self.world_position(self.nozzle),
            PipettingField.DISPLAY_POSITION: self.world_position(self.display),
            PipettingField.DISPLAY_SIZE: list(self.display_size),
            PipettingField.DISPLAY_MESH: PipettingPart.DISPLAY,
            PipettingField.CHANNELS: [
                {
                    PipettingField.IDENTIFIER: channel,
                    PipettingField.NOZZLE_POSITION: self.world_position(
                        self.channel_nozzle(channel)
                    ),
                    PipettingField.LIQUID_MESH: f"Pipetting {channel} liquid",
                    PipettingField.LIQUID_BOTTOM: self.reservoir_liquid_bottom,
                    PipettingField.LIQUID_HEIGHT: self.reservoir_liquid_height,
                }
                for channel in PipettingChannel
            ],
        }


# %% open receiving area and fixed collision geometry


@dataclass
class PipettingDescription:
    """Separate instrument collisions supporting the glass below its grasp height."""

    description: UrdfDescription
    """Environment receiving the stationary instrument."""
    dimensions: PipettingDimensions = PipettingDimensions()
    """Authored dimensions shared with the Blender model."""

    def build(self) -> None:
        """Attach the platform, rear column, dispensing head and front-open guides."""
        dimensions = self.dimensions
        body = self.description.link(
            PipettingPart.BODY, f"assets/{PipettingPart.ASSET}.glb"
        )
        self.description.joint(
            PipettingPart.JOINT,
            "laboratory_root",
            PipettingPart.BODY,
            dimensions.origin,
        )
        for name, center, size in (
            (
                PipettingPart.PLATFORM,
                dimensions.platform_center,
                dimensions.platform_size,
            ),
            (PipettingPart.COLUMN, dimensions.column_center, dimensions.column_size),
            (PipettingPart.HEAD, dimensions.head_center, dimensions.head_size),
        ):
            self.description.box(body, name, center, size)
        for guide in dimensions.holder_guides():
            self.description.box(body, guide.name, guide.center, guide.size)
        for channel in PipettingChannel:
            nozzle = dimensions.channel_nozzle(channel)
            self.description.box(
                body,
                f"pipetting_{channel}_nozzle",
                (nozzle[0], nozzle[1], nozzle[2] + dimensions.nozzle_length / 2),
                (
                    dimensions.nozzle_radius * 2,
                    dimensions.nozzle_radius * 2,
                    dimensions.nozzle_length,
                ),
            )
