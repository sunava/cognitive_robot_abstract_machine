"""Volume models and helpers for container interior clearance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, Tuple, Optional, Iterable

import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Cylinder, Box, Scale


class ContainerVolumeModel(Protocol):
    """Protocol for volume models providing clearance queries."""

    def distance_inside(self, p: np.ndarray) -> float: ...
    def inner_height(self) -> float: ...
    def inner_bounds_xy(self) -> Tuple[float, float]: ...


@dataclass(frozen=True)
class BoxVolumeModel(ContainerVolumeModel):
    """Axis-aligned box volume model."""

    half_extents: np.ndarray

    def distance_inside(self, p: np.ndarray) -> float:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        hx, hy, hz = (
            float(self.half_extents[0]),
            float(self.half_extents[1]),
            float(self.half_extents[2]),
        )
        dx = hx - abs(x)
        dy = hy - abs(y)
        dz = hz - abs(z)
        return min(dx, dy, dz)

    def inner_height(self) -> float:
        return float(self.half_extents[2]) * 2.0

    def inner_bounds_xy(self) -> Tuple[float, float]:
        return float(self.half_extents[0]), float(self.half_extents[1])


@dataclass(frozen=True)
class CylinderVolumeModel(ContainerVolumeModel):
    """Cylindrical volume model aligned to the Z axis."""

    radius: float
    half_height: float

    def distance_inside(self, p: np.ndarray) -> float:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        dr = float(self.radius) - math.sqrt(x * x + y * y)
        dz = float(self.half_height) - abs(z)
        return min(dr, dz)

    def inner_height(self) -> float:
        return float(self.half_height) * 2.0

    def inner_bounds_xy(self) -> Tuple[float, float]:
        r = float(self.radius)
        return r, r


def _as_box_model(scale: Scale, padding: float) -> BoxVolumeModel:
    """Convert a box scale to a padded box volume model."""
    hx = 0.5 * float(scale.x) - float(padding)
    hy = 0.5 * float(scale.y) - float(padding)
    hz = 0.5 * float(scale.z) - float(padding)
    return BoxVolumeModel(
        half_extents=np.array([max(hx, 0.0), max(hy, 0.0), max(hz, 0.0)], dtype=float)
    )


def _as_cyl_model(radius: float, height: float, padding: float) -> CylinderVolumeModel:
    """Convert cylinder dimensions to a padded volume model."""
    r = float(radius) - float(padding)
    h2 = 0.5 * float(height) - float(padding)
    return CylinderVolumeModel(radius=max(r, 0.0), half_height=max(h2, 0.0))


def volume_from_body_collision(
    body: Body, padding: float = 0.0
) -> Optional[ContainerVolumeModel]:
    """Build a container volume model from a body's collision shape."""
    col = body.collision
    if not isinstance(col, ShapeCollection):
        return None

    shapes = list(col.shapes)
    if not shapes:
        return None

    s0 = shapes[0]

    if isinstance(s0, Box):
        return _as_box_model(s0.scale, padding)

    if isinstance(s0, Cylinder):
        return _as_cyl_model(s0.radius, s0.height, padding)

    return None
