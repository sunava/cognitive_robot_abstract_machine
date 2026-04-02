"""Separation and slicing primitive generators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, List, Tuple, Union, Optional

import numpy as np

from pycram.datastructures.pose import PoseStamped
from .contact_manifold import (
    ContactAnchor,
    compile_contact_manifold,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(frozen=True)
class CutPlane:
    """Planar cutting volume definition."""

    frame_id: str
    half_extents: np.ndarray


@dataclass(frozen=True)
class CutAnchor:
    """Anchor position for cut trajectories."""

    frame_id: str
    p: np.ndarray


@dataclass(frozen=True)
class SliceSpec:
    """Parameters for slicing along a box axis."""

    slice_thickness: float
    z_clearance: float
    z_cut: float
    margin_xy: float = 0.0


def new_pose(frame_id: str) -> PoseStamped:
    """Create a PoseStamped with a given frame id."""
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    return ps


def set_xyz(ps: PoseStamped, x: float, y: float, z: float) -> None:
    """Set position fields on a PoseStamped."""
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)


def bind_slice_anchors_along_x(
    obj_frame_id: str,
    half_extents: np.ndarray,
    spec: SliceSpec,
) -> List[CutAnchor]:
    """Compute evenly spaced slice anchors along the X axis."""
    hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])

    t = float(spec.slice_thickness)
    if t <= 0.0:
        return []

    x_min = -(hx - float(spec.margin_xy))
    x_max = +(hx - float(spec.margin_xy))
    if x_max <= x_min:
        return []

    n = int((x_max - x_min) / t)
    if n <= 0:
        return []

    start = x_min + 0.5 * t
    anchors: List[CutAnchor] = []
    for i in range(n):
        x = start + float(i) * t
        anchors.append(
            CutAnchor(frame_id=obj_frame_id, p=np.array([x, 0.0, 0.0], dtype=float))
        )

    return anchors


class SepMode(Enum):
    """Supported separation modes."""

    SLICE = auto()
    SAW = auto()
    PRESS = auto()
    PULL_APART = auto()


@dataclass(frozen=True)
class SeparationSpec:
    """Parameters for separation trajectories."""

    mode: SepMode
    length: float
    depth: float
    n: int = 80
    normal_force: float = 15.0
    tangential_osc_amp: float = 0.01
    tangential_osc_hz: float = 6.0
    pull_gap: float = 0.12


def compile_slice_kernel(
    anchor: CutAnchor,
    obj_half_extents: np.ndarray,
    spec: SliceSpec,
    tilt_y_deg: float = 0.0,
) -> PoseStamped:
    """Compute the target pose for a slice kernel."""
    hz = float(obj_half_extents[2])
    z_cut = float(spec.z_cut)

    x0 = float(anchor.p[0])
    y0 = float(anchor.p[1])

    z_top = hz + float(spec.z_clearance)
    dz = z_top - z_cut
    dx = dz * math.tan(math.radians(float(tilt_y_deg)))

    p = new_pose(anchor.frame_id)
    set_xyz(p, x0 + dx, y0, z_cut)
    return p


def compile_separation(
    spec: SeparationSpec,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Generate XYZ offsets for a separation primitive."""
    t = np.linspace(0.0, 1.0, spec.n, dtype=float)

    z = -float(spec.depth) * t

    if spec.mode == SepMode.PRESS:
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        return np.column_stack([x, y, z])

    if spec.mode == SepMode.SLICE:
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        return np.column_stack([x, y, z])

    if spec.mode == SepMode.SAW:
        x = np.zeros_like(t)
        osc = float(spec.tangential_osc_amp) * np.sin(
            2.0 * math.pi * float(spec.tangential_osc_hz) * t
        )
        y = osc
        return np.column_stack([x, y, z])

    if spec.mode == SepMode.PULL_APART:
        gap = 0.5 * float(spec.pull_gap) * t
        left = np.column_stack([np.zeros_like(t), -gap, np.zeros_like(t)])
        right = np.column_stack([np.zeros_like(t), +gap, np.zeros_like(t)])
        return left, right

    raise ValueError(f"Unhandled mode: {spec.mode}")


def compile_penetration_kernel(
    p0: np.ndarray,
    n: np.ndarray,
    depth: float,
    n_steps: int,
) -> np.ndarray:
    """Generate a straight penetration trace along a normal."""
    nn = np.asarray(n, dtype=float)
    nn = nn / max(np.linalg.norm(nn), 1e-12)
    t = np.linspace(0.0, 1.0, int(n_steps), dtype=float)
    return p0[None, :] - (depth * t)[:, None] * nn[None, :]


def compile_separation_contact(
    frame_id: Body,
    p0: np.ndarray,
    n: np.ndarray,
    spec: SeparationSpec,
    t1: Optional[np.ndarray] = None,
    t2: Optional[np.ndarray] = None,
) -> Union[Iterable[PoseStamped], Tuple[Iterable[PoseStamped], Iterable[PoseStamped]]]:
    """Compile separation trajectories on the contact manifold."""
    p0 = np.asarray(p0, dtype=float).reshape(3)
    n = np.asarray(n, dtype=float).reshape(3)

    t = np.linspace(0.0, 1.0, int(spec.n), dtype=float)
    d = float(spec.depth) * t

    if spec.mode == SepMode.PULL_APART:
        gap = 0.5 * float(spec.pull_gap) * t
        qL = np.column_stack([np.zeros_like(t), -gap])
        qR = np.column_stack([np.zeros_like(t), +gap])
        a = ContactAnchor(frame_id=frame_id, p0=p0, n=n, t1=t1, t2=t2)
        z0 = np.zeros_like(t)
        left = list(compile_contact_manifold(a, z0, qL))
        right = list(compile_contact_manifold(a, z0, qR))
        return left, right

    q = np.zeros((t.shape[0], 2), dtype=float)

    if spec.mode == SepMode.SAW:
        q[:, 1] = float(spec.tangential_osc_amp) * np.sin(
            2.0 * math.pi * float(spec.tangential_osc_hz) * t
        )

    a = ContactAnchor(frame_id=frame_id, p0=p0, n=n, t1=t1, t2=t2)
    return list(compile_contact_manifold(a, d, q))
