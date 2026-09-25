"""
Dimensioned surfaces of revolution for hollow laboratory vessels.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin


# %% Sample carrier dimensions
@dataclass(frozen=True)
class RackDimensions:
    """Open rack plate and foot dimensions in its local frame."""

    origin: tuple[float, float, float] = (0.22, -0.04, 0.900)
    """Lower centre of the rack in the laboratory frame."""
    width: float = 0.19
    """Full extent across the three columns."""
    depth: float = 0.11
    """Full extent across the two rows."""
    base_thickness: float = 0.006
    """Thickness of the lower plate."""
    top_bottom: float = 0.062
    """Elevation of the underside of the upper plate."""
    top_thickness: float = 0.006
    """Thickness of the upper plate."""
    hole_radius: float = 0.0115
    """Radius of the open visual holes and half-width of collision apertures."""
    columns: tuple[float, ...] = (-0.06, 0.0, 0.06)
    """Horizontal hole centres."""
    rows: tuple[float, ...] = (-0.025, 0.025)
    """Depth coordinates of the hole centres."""
    handle_center: tuple[float, float, float] = (0.0, -0.125, 0.055)
    """Front carry bar centre providing vertical jaw clearance."""
    handle_size: tuple[float, float, float] = (0.12, 0.024, 0.024)
    """Carry bar extents in rack coordinates."""
    mass: float = 0.18
    """Empty rack mass in kilograms."""
    drawer_offset: tuple[float, float, float] = (0.0, -0.10, 0.0)
    """Stored centre relative to the drawer floor centre."""
    guide_inner_radius: float = 0.0095
    """Straight lower bore radius surrounding each rounded vessel bottom."""
    guide_outer_radius: float = 0.012
    """Outside radius of the lower guide wall."""
    guide_bore_height: float = 0.010
    """Straight guide height above the bottom plate."""
    guide_flare_height: float = 0.006
    """Height of the widening upper entry funnel."""
    guide_entry_radius: float = 0.0115
    """Inside radius at the funnel mouth."""
    guide_segments: int = 12
    """Number of tangent collision segments around each guide."""

    def guide_profile(self) -> tuple[ProfilePoint, ...]:
        """Return the closed hollow guide wall with a widened insertion mouth."""
        lower = self.base_thickness
        shoulder = lower + self.guide_bore_height
        upper = shoulder + self.guide_flare_height
        outer_entry = (
            self.guide_outer_radius + self.guide_entry_radius - self.guide_inner_radius
        )
        return (
            ProfilePoint(self.guide_inner_radius, lower),
            ProfilePoint(self.guide_outer_radius, lower),
            ProfilePoint(self.guide_outer_radius, shoulder),
            ProfilePoint(outer_entry, upper),
            ProfilePoint(self.guide_entry_radius, upper),
            ProfilePoint(self.guide_inner_radius, shoulder),
            ProfilePoint(self.guide_inner_radius, lower),
        )


# %% Revolved cross sections
@dataclass(frozen=True)
class ProfilePoint:
    """
    One radius and elevation in a vessel cross section.
    """

    radius: float
    """
    Distance from the vertical axis, in metres.
    """

    height: float
    """
    Elevation above the vessel bottom, in metres.
    """


@dataclass(frozen=True)
class TubeDimensions:
    """
    Nominal dimensions of a rounded borosilicate tube.
    """

    height: float = 0.150
    """
    Elevation of the uppermost rim.
    """

    outer_radius: float = 0.009
    """
    Outside radius of the cylindrical wall.
    """

    inner_radius: float = 0.0078
    """
    Inside radius of the cylindrical wall.
    """

    curve_segments: int = 16
    """
    Number of segments in each quarter circle.
    """

    @property
    def wall_thickness(self) -> float:
        """
        Return the material thickness at the straight wall and bottom.
        """
        return self.outer_radius - self.inner_radius

    @property
    def bottom_radius(self) -> float:
        """
        Return the elevation of the hemispherical bottom centre.
        """
        return self.outer_radius

    def profile(self) -> tuple[ProfilePoint, ...]:
        """
        Trace the outside, rounded lip and inside of the open vessel.
        """
        points = [ProfilePoint(0, 0)]
        for index in range(1, self.curve_segments + 1):
            angle = -pi / 2 + pi / 2 * index / self.curve_segments
            points.append(
                ProfilePoint(
                    self.outer_radius * cos(angle),
                    self.bottom_radius + self.outer_radius * sin(angle),
                )
            )
        rim_radius = self.wall_thickness / 2
        rim_centre = (self.outer_radius + self.inner_radius) / 2
        points.append(ProfilePoint(self.outer_radius, self.height - rim_radius))
        for index in range(1, self.curve_segments * 2 + 1):
            angle = pi * index / (self.curve_segments * 2)
            points.append(
                ProfilePoint(
                    rim_centre + rim_radius * cos(angle),
                    self.height - rim_radius + rim_radius * sin(angle),
                )
            )
        points.append(ProfilePoint(self.inner_radius, self.bottom_radius))
        for index in range(1, self.curve_segments):
            angle = -pi / 2 * index / self.curve_segments
            points.append(
                ProfilePoint(
                    self.inner_radius * cos(angle),
                    self.bottom_radius + self.inner_radius * sin(angle),
                )
            )
        points.append(ProfilePoint(0, self.wall_thickness))
        return tuple(points)


@dataclass(frozen=True)
class LatheMesh:
    """
    A watertight indexed surface produced by rotating a cross section.
    """

    vertices: tuple[tuple[float, float, float], ...]
    """
    Positions in a metre-scaled, Z-up coordinate frame.
    """

    faces: tuple[tuple[int, ...], ...]
    """
    Triangles at the axis and quadrilaterals between circular rings.
    """

    @classmethod
    def from_profile(
        cls, profile: tuple[ProfilePoint, ...], segments: int = 96
    ) -> LatheMesh:
        """
        Rotate an ordered profile while sharing the axis vertices.
        """
        if segments < 3 or len(profile) < 2:
            raise ValueError(
                "A lathe needs at least three sides and two profile points"
            )
        vertices = []
        rings = []
        for point in profile:
            ring = []
            count = 1 if point.radius == 0 else segments
            for index in range(count):
                angle = 2 * pi * index / count
                ring.append(len(vertices))
                vertices.append(
                    (point.radius * cos(angle), point.radius * sin(angle), point.height)
                )
            rings.append(ring)
        faces = []
        for first, second in zip(rings, rings[1:]):
            for index in range(segments):
                following = (index + 1) % segments
                if len(first) == 1:
                    faces.append((first[0], second[following], second[index]))
                elif len(second) == 1:
                    faces.append((first[index], first[following], second[0]))
                else:
                    faces.append(
                        (
                            first[index],
                            first[following],
                            second[following],
                            second[index],
                        )
                    )
        return cls(tuple(vertices), tuple(faces))
