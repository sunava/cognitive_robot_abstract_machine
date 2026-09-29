"""Dimensioned, renderer-independent descriptions of reusable laboratory devices."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from math import isclose, isfinite, pi


# %% portable identifiers and metric coordinates
class InvalidInstrumentSpec(ValueError):
    """An instrument cannot produce consistent metric geometry and URDF frames."""


def validate_identifier(value: str) -> None:
    """Require a local identifier that is safe in filenames and robot frame names."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", value) is None:
        raise InvalidInstrumentSpec(f"Invalid local instrument identifier: {value!r}")


def validate_vector(value: tuple[float, float, float], owner: str) -> None:
    """Require complete finite coordinates before deriving geometry or transforms."""
    if len(value) != 3 or not all(isfinite(component) for component in value):
        raise InvalidInstrumentSpec(
            f"{owner}: coordinates must contain three finite values"
        )


# %% surface and articulation vocabulary


class Finish(StrEnum):
    """Shared AICOR instrument finishes."""

    WHITE = "white"
    TEAL = "teal"
    PALE_TEAL = "pale_teal"
    STEEL = "steel"
    DARK = "dark"
    GLASS = "glass"
    SCREEN = "screen"
    INK = "ink"
    PAPER = "paper"
    BLUE = "blue"
    AMBER = "amber"
    RED = "red"


class JointKind(StrEnum):
    """URDF articulation kinds supported by standalone instruments."""

    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    CONTINUOUS = "continuous"


class ShapeKind(StrEnum):
    """Manufactured surfaces and non-colliding markings."""

    BOX = "box"
    CYLINDER = "cylinder"
    CONE = "cone"
    RING = "ring"
    SPHERE = "sphere"
    LABEL = "label"
    LOGO = "logo"
    TUBE = "tube"


@dataclass
class ShapeSpec:
    """One surface described in its owning rigid body's frame."""

    kind: ShapeKind
    """Construction primitive."""
    name: str
    """Human-readable name retained in the Blender object."""
    center: tuple[float, float, float]
    """Local position in metres."""
    dimensions: tuple[float, ...]
    """Primitive dimensions: box extents, radii and depth, or marking size."""
    material: Finish
    """Manufactured surface finish."""
    rotation: tuple[float, float, float] = (0, 0, 0)
    """XYZ Euler orientation in radians."""
    bevel: float = 0
    """Edge rounding distance in metres."""
    collision: bool = True
    """Whether the surface contributes contact geometry."""
    text: str = ""
    """Lettering for a physical label."""
    points: tuple[tuple[float, float, float], ...] = ()
    """Control points for a cable or pipe."""

    def validate(self) -> None:
        """Reject malformed dimensions, invalid bores, and nonfinite placements."""
        validate_vector(self.center, self.name)
        validate_vector(self.rotation, self.name)
        dimension_counts = {
            ShapeKind.BOX: 3,
            ShapeKind.CYLINDER: 2,
            ShapeKind.CONE: 3,
            ShapeKind.RING: 3,
            ShapeKind.SPHERE: 1,
            ShapeKind.LABEL: 2,
            ShapeKind.LOGO: 1,
            ShapeKind.TUBE: 1,
        }
        if len(self.dimensions) != dimension_counts[self.kind]:
            raise InvalidInstrumentSpec(f"{self.name}: incomplete primitive dimensions")
        if not all(isfinite(value) for value in self.dimensions):
            raise InvalidInstrumentSpec(f"{self.name}: dimensions must be finite")
        if not isfinite(self.bevel) or self.bevel < 0:
            raise InvalidInstrumentSpec(
                f"{self.name}: bevel must be finite and nonnegative"
            )
        if self.kind == ShapeKind.LABEL:
            if self.dimensions[0] <= 0 or self.dimensions[1] < 0:
                raise InvalidInstrumentSpec(
                    f"{self.name}: invalid lettering dimensions"
                )
            return
        if self.kind == ShapeKind.CONE:
            bottom, top, height = self.dimensions
            if min(bottom, top) < 0 or max(bottom, top) <= 0 or height <= 0:
                raise InvalidInstrumentSpec(f"{self.name}: invalid cone dimensions")
            return
        if any(value <= 0 for value in self.dimensions):
            raise InvalidInstrumentSpec(f"{self.name}: dimensions must be positive")
        if self.kind == ShapeKind.RING and self.dimensions[1] >= self.dimensions[0]:
            raise InvalidInstrumentSpec(f"{self.name}: sleeve must have an open bore")
        if self.kind == ShapeKind.TUBE:
            if len(self.points) < 2:
                raise InvalidInstrumentSpec(
                    f"{self.name}: a tube needs at least two points"
                )
            for point in self.points:
                validate_vector(point, self.name)
            if any(
                first == second for first, second in zip(self.points, self.points[1:])
            ):
                raise InvalidInstrumentSpec(
                    f"{self.name}: adjacent tube points must differ"
                )


@dataclass
class PartSpec:
    """A rigid body and the joint attaching it to its parent."""

    name: str
    """Device-local body name."""
    parent: str | None = None
    """Device-local parent body name, absent for the root."""
    origin: tuple[float, float, float] = (0, 0, 0)
    """Joint-frame translation in the parent body."""
    rotation: tuple[float, float, float] = (0, 0, 0)
    """Joint-frame XYZ Euler rotation in the parent body."""
    joint: JointKind = JointKind.FIXED
    """Kind of relative motion."""
    axis: tuple[float, float, float] = (0, 0, 1)
    """Unit motion axis in the joint frame."""
    lower: float = 0
    """Lower travel limit in metres or radians."""
    upper: float = 0
    """Upper travel limit in metres or radians."""
    open_value: float = 0
    """Useful inspection position for an open-device preview."""
    mass: float = 1
    """Approximate model mass in kilograms; not a calibrated product specification."""
    shapes: list[ShapeSpec] = field(default_factory=list)
    """Visual and collision components belonging to this body."""

    def box(
        self,
        name: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        material: Finish = Finish.WHITE,
        bevel: float = 0.002,
        collision: bool = True,
        rotation: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        """Add a rounded rectangular component with a box contact primitive."""
        self.shapes.append(
            ShapeSpec(
                ShapeKind.BOX,
                name,
                center,
                tuple(size),
                material,
                rotation,
                bevel,
                collision,
            )
        )

    def cylinder(
        self,
        name: str,
        center: tuple[float, float, float],
        radius: float,
        depth: float,
        material: Finish,
        rotation: tuple[float, float, float] = (0, 0, 0),
        collision: bool = True,
    ) -> None:
        """Add a cylinder whose unrotated axis points along local Z."""
        self.shapes.append(
            ShapeSpec(
                ShapeKind.CYLINDER,
                name,
                center,
                (radius, depth),
                material,
                rotation,
                collision=collision,
            )
        )

    def cone(
        self,
        name: str,
        center: tuple[float, float, float],
        radius_bottom: float,
        radius_top: float,
        depth: float,
        material: Finish,
        rotation: tuple[float, float, float] = (0, 0, 0),
        collision: bool = True,
    ) -> None:
        """Add a tapered component with a conservative cylindrical collider."""
        self.shapes.append(
            ShapeSpec(
                ShapeKind.CONE,
                name,
                center,
                (radius_bottom, radius_top, depth),
                material,
                rotation,
                collision=collision,
            )
        )

    def ring(
        self,
        name: str,
        center: tuple[float, float, float],
        outer_radius: float,
        inner_radius: float,
        depth: float,
        material: Finish,
        rotation: tuple[float, float, float] = (0, 0, 0),
        collision: bool = True,
    ) -> None:
        """Add a genuinely hollow sleeve with segmented wall contact geometry."""
        self.shapes.append(
            ShapeSpec(
                ShapeKind.RING,
                name,
                center,
                (outer_radius, inner_radius, depth),
                material,
                rotation,
                collision=collision,
            )
        )

    def sphere(
        self,
        name: str,
        center: tuple[float, float, float],
        radius: float,
        material: Finish,
        collision: bool = True,
    ) -> None:
        """Add a rounded end or control."""
        self.shapes.append(
            ShapeSpec(
                ShapeKind.SPHERE, name, center, (radius,), material, collision=collision
            )
        )

    def label(
        self,
        text: str,
        center: tuple[float, float, float],
        size: float,
        material: Finish = Finish.TEAL,
        rotation: tuple[float, float, float] = (pi / 2, 0, 0),
        width: float | None = None,
    ) -> None:
        """Add centered lettering, optionally constrained to a maximum width."""
        self.shapes.append(
            ShapeSpec(
                ShapeKind.LABEL,
                text,
                center,
                (size, width or 0),
                material,
                rotation,
                collision=False,
                text=text,
            )
        )

    def logo(
        self,
        center: tuple[float, float, float],
        width: float,
        rotation: tuple[float, float, float] = (pi / 2, 0, 0),
    ) -> None:
        """Place the original transparent AICOR artwork on a flat surface."""
        self.shapes.append(
            ShapeSpec(
                ShapeKind.LOGO,
                "AICOR logo",
                center,
                (width,),
                Finish.TEAL,
                rotation,
                collision=False,
            )
        )

    def tube(
        self,
        name: str,
        points: Iterable[tuple[float, float, float]],
        radius: float,
        material: Finish,
        collision: bool = False,
    ) -> None:
        """Add a cable or pipe following local control points."""
        self.shapes.append(
            ShapeSpec(
                ShapeKind.TUBE,
                name,
                (0, 0, 0),
                (radius,),
                material,
                collision=collision,
                points=tuple(points),
            )
        )

    def validate(self) -> None:
        """Require physically usable mass, transforms, travel and primitive surfaces."""
        validate_identifier(self.name)
        validate_vector(self.origin, self.name)
        validate_vector(self.rotation, self.name)
        validate_vector(self.axis, self.name)
        if not isfinite(self.mass) or self.mass <= 0:
            raise InvalidInstrumentSpec(
                f"{self.name}: mass must be finite and positive"
            )
        if self.joint != JointKind.FIXED:
            if not isclose(
                sum(value * value for value in self.axis), 1.0, abs_tol=1e-8
            ):
                raise InvalidInstrumentSpec(
                    f"{self.name}: motion axis must have unit length"
                )
            if not all(
                isfinite(value) for value in (self.lower, self.upper, self.open_value)
            ):
                raise InvalidInstrumentSpec(
                    f"{self.name}: joint coordinates must be finite"
                )
        if self.joint in (JointKind.PRISMATIC, JointKind.REVOLUTE):
            if (
                self.lower >= self.upper
                or not self.lower <= self.open_value <= self.upper
            ):
                raise InvalidInstrumentSpec(f"{self.name}: invalid joint travel")
        for shape in self.shapes:
            shape.validate()


@dataclass
class FrameSpec:
    """A semantic interaction frame attached to an articulated rigid body."""

    name: str
    """Device-local TF frame name."""
    parent: str
    """Owning rigid body."""
    position: tuple[float, float, float]
    """Position in the owning body."""
    role: str
    """English description of the grasp, insertion, control or observation target."""
    rotation: tuple[float, float, float] = (0, 0, 0)
    """XYZ Euler orientation in the owning body."""


@dataclass
class DeviceSpec:
    """An independently loadable instrument or movable accessory."""

    identifier: str
    """Stable resource directory and namespace."""
    title: str
    """English instrument name."""
    description: str
    """Scope of its authored geometry and articulation."""
    parts: list[PartSpec] = field(default_factory=list)
    """Rigid bodies in parent-before-child order."""
    frames: list[FrameSpec] = field(default_factory=list)
    """Interaction targets represented as fixed URDF links."""
    accessories: list[str] = field(default_factory=list)
    """Other independently loadable resource identifiers."""

    def part(
        self,
        name: str,
        parent: str | None = None,
        *,
        origin: tuple[float, float, float] = (0, 0, 0),
        rotation: tuple[float, float, float] = (0, 0, 0),
        joint: JointKind = JointKind.FIXED,
        axis: tuple[float, float, float] = (0, 0, 1),
        lower: float = 0,
        upper: float = 0,
        open_value: float = 0,
        mass: float = 1,
    ) -> PartSpec:
        """Add one rigid body using device-local names."""
        part = PartSpec(
            name, parent, origin, rotation, joint, axis, lower, upper, open_value, mass
        )
        self.parts.append(part)
        return part

    def frame(
        self,
        name: str,
        parent: str,
        position: tuple[float, float, float],
        role: str,
        rotation: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        """Add a target that follows the parent when its joint moves."""
        self.frames.append(FrameSpec(name, parent, position, role, rotation))

    def validate(self) -> None:
        """Reject disconnected hierarchies and unusable primitive dimensions."""
        validate_identifier(self.identifier)
        if not self.parts or self.parts[0].parent is not None:
            raise InvalidInstrumentSpec(
                f"{self.identifier}: first part must be the root"
            )
        root = self.parts[0]
        if (
            root.origin != (0, 0, 0)
            or root.rotation != (0, 0, 0)
            or root.joint != JointKind.FIXED
        ):
            raise InvalidInstrumentSpec(
                f"{self.identifier}: root must be fixed at its local origin"
            )
        known: set[str] = set()
        for index, part in enumerate(self.parts):
            if part.name in known or index and part.parent not in known:
                raise InvalidInstrumentSpec(
                    f"{self.identifier}: disconnected or duplicate part {part.name}"
                )
            part.validate()
            known.add(part.name)
        for frame in self.frames:
            validate_identifier(frame.name)
            validate_vector(frame.position, frame.name)
            validate_vector(frame.rotation, frame.name)
            if frame.parent not in known or frame.name in known:
                raise InvalidInstrumentSpec(f"{frame.name}: invalid interaction frame")
            known.add(frame.name)
        for accessory in self.accessories:
            validate_identifier(accessory)
