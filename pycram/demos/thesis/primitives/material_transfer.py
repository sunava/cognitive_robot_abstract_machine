"""Material transfer primitives for discharge and shake motions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from geometry_msgs.msg import PoseStamped


@dataclass(frozen=True)
class BoundaryAnchor:
    """Anchor on a boundary with a flow normal."""

    frame_id: str
    p: np.ndarray
    n: np.ndarray


@dataclass(frozen=True)
class DischargeSpec:
    """Parameters for boundary discharge motion."""

    steps: int
    f_start: float
    f_step: float
    f_max: Optional[float] = None
    epsilon: float = 0.0


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize a vector and raise on zero length."""
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("zero-length vector")
    return v / n


def _new_pose(frame_id: str) -> PoseStamped:
    """Create a PoseStamped with a given frame id."""
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    return ps


def _set_xyz(ps: PoseStamped, x: float, y: float, z: float) -> None:
    """Set position fields on a PoseStamped."""
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)


def compile_boundary_discharge(
    anchor: BoundaryAnchor,
    spec: DischargeSpec,
) -> Iterable[PoseStamped]:
    """Generate a monotonic discharge along the boundary normal."""
    steps = int(spec.steps)
    if steps <= 0:
        return []

    p0 = np.array(anchor.p, dtype=float).reshape(3)
    n = _unit(np.array(anchor.n, dtype=float).reshape(3))

    out: list[PoseStamped] = []
    f_max = None if spec.f_max is None else float(spec.f_max)

    for i in range(steps):
        f = float(spec.f_start) + float(spec.f_step) * float(i)
        if f_max is not None:
            f = min(f, f_max)
        if f < float(spec.epsilon):
            continue

        p = p0 + f * n

        ps = _new_pose(anchor.frame_id)
        _set_xyz(ps, float(p[0]), float(p[1]), float(p[2]))
        out.append(ps)

    return out


@dataclass(frozen=True)
class ShakeSpec:
    """Parameters for boundary shake motion."""

    steps: int
    f_bias: float
    f_amp: float
    omega: float
    epsilon: float = 0.0


def compile_boundary_shake(
    anchor: BoundaryAnchor,
    spec: ShakeSpec,
) -> Iterable[PoseStamped]:
    """Generate a sinusoidal shake along the boundary normal."""
    steps = int(spec.steps)
    if steps <= 0:
        return []

    p0 = np.array(anchor.p, dtype=float).reshape(3)
    n = _unit(np.array(anchor.n, dtype=float).reshape(3))

    out: list[PoseStamped] = []

    for i in range(steps):
        t = float(i)
        f = float(spec.f_bias) + float(spec.f_amp) * math.sin(float(spec.omega) * t)
        if f < float(spec.epsilon):
            continue

        p = p0 + f * n

        ps = _new_pose(anchor.frame_id)
        _set_xyz(ps, float(p[0]), float(p[1]), float(p[2]))
        out.append(ps)

    return out
