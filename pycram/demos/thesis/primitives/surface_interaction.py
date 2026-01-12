from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from geometry_msgs.msg import PoseStamped

from pycram.demos.thesis.primitives.contact_manifold import (
    ContactAnchor,
    compile_contact_manifold,
)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("zero-length vector")
    return v / n


def _orthonormal_tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _unit(n)
    a = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(a, n))) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=float)
    t1 = _unit(np.cross(n, a))
    t2 = _unit(np.cross(n, t1))
    return t1, t2


@dataclass(frozen=True)
class SurfacePlane:
    frame_id: str
    origin: np.ndarray
    normal: np.ndarray
    half_extents_uv: np.ndarray
    t1: Optional[np.ndarray] = None
    t2: Optional[np.ndarray] = None

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = _unit(np.array(self.normal, dtype=float))
        if self.t1 is None or self.t2 is None:
            t1, t2 = _orthonormal_tangent_basis(n)
            return t1, t2, n
        t1 = _unit(np.array(self.t1, dtype=float))
        t2 = _unit(np.array(self.t2, dtype=float))
        if abs(float(np.dot(t1, n))) > 1e-6 or abs(float(np.dot(t2, n))) > 1e-6:
            raise ValueError("tangent basis is not orthogonal to normal")
        return t1, t2, n


@dataclass(frozen=True)
class SurfaceAnchor:
    frame_id: str
    p: np.ndarray


@dataclass(frozen=True)
class ScrubSpec:
    radius: float
    points_per_cycle: int
    cycles: int
    z_offset: float = 0.0


@dataclass(frozen=True)
class SweepSpec:
    spacing: float
    margin: float = 0.0
    z_offset: float = 0.0


def bind_surface_anchor(
    surface: SurfacePlane, margin: float
) -> Optional[SurfaceAnchor]:
    hu, hv = float(surface.half_extents_uv[0]), float(surface.half_extents_uv[1])
    if hu <= margin or hv <= margin:
        return None
    p = np.array(surface.origin, dtype=float)
    return SurfaceAnchor(frame_id=surface.frame_id, p=p)


def _anchor_uv(surface: SurfacePlane, anchor_p: np.ndarray) -> tuple[float, float]:
    t1, t2, n = surface.basis()
    o = np.array(surface.origin, dtype=float)
    p = np.array(anchor_p, dtype=float)
    d = p - o
    return float(np.dot(d, t1)), float(np.dot(d, t2))


def compile_scrub_circle(
    anchor: SurfaceAnchor,
    surface: SurfacePlane,
    spec: ScrubSpec,
    margin: float,
) -> Iterable[PoseStamped]:
    hu, hv = float(surface.half_extents_uv[0]), float(surface.half_extents_uv[1])
    r_max = max(0.0, min(hu, hv) - float(margin))
    r = min(float(spec.radius), r_max)

    ppc = int(spec.points_per_cycle)
    if ppc <= 0:
        return []

    cycles = int(spec.cycles)
    if cycles <= 0:
        return []

    n_pts = ppc * cycles

    cu, cv = _anchor_uv(surface, anchor.p)

    q = np.zeros((n_pts, 2), dtype=float)
    for i in range(n_pts):
        a = (2.0 * math.pi) * (float(i) / float(ppc))
        u = cu + r * math.cos(a)
        v = cv + r * math.sin(a)
        if abs(u) > hu - margin or abs(v) > hv - margin:
            q[i, 0] = np.nan
            q[i, 1] = np.nan
        else:
            q[i, 0] = u - cu
            q[i, 1] = v - cv

    keep = np.isfinite(q[:, 0]) & np.isfinite(q[:, 1])
    q = q[keep, :]

    d = np.zeros((q.shape[0],), dtype=float)

    t1, t2, n = surface.basis()
    a = ContactAnchor(
        frame_id=anchor.frame_id,
        p0=np.array(anchor.p, dtype=float) + float(spec.z_offset) * n,
        n=n,
        t1=t1,
        t2=t2,
    )
    return list(compile_contact_manifold(a, d, q))


def compile_sweep_raster(
    surface: SurfacePlane,
    spec: SweepSpec,
    start_xy: Optional[np.ndarray] = None,
    end_xy: Optional[np.ndarray] = None,
) -> Iterable[SurfaceAnchor]:
    hu, hv = float(surface.half_extents_uv[0]), float(surface.half_extents_uv[1])
    m = float(spec.margin)
    s = float(spec.spacing)

    if s <= 0.0:
        return []

    if start_xy is None:
        start_xy = np.array([-(hu - m), -(hv - m)], dtype=float)
    if end_xy is None:
        end_xy = np.array([(hu - m), (hv - m)], dtype=float)

    u0, v0 = float(start_xy[0]), float(start_xy[1])
    u1, v1 = float(end_xy[0]), float(end_xy[1])

    u0 = max(-(hu - m), min((hu - m), u0))
    u1 = max(-(hu - m), min((hu - m), u1))
    v0 = max(-(hv - m), min((hv - m), v0))
    v1 = max(-(hv - m), min((hv - m), v1))

    vs = np.arange(min(v0, v1), max(v0, v1) + 1e-9, s, dtype=float)
    anchors: list[SurfaceAnchor] = []

    t1, t2, n = surface.basis()
    o = np.array(surface.origin, dtype=float)
    w = float(spec.z_offset)

    left = min(u0, u1)
    right = max(u0, u1)

    for k, v in enumerate(vs):
        us = [left, right] if (k % 2 == 0) else [right, left]
        for u in us:
            p = o + float(u) * t1 + float(v) * t2 + w * n
            anchors.append(SurfaceAnchor(frame_id=surface.frame_id, p=p))

    return anchors


def compile_wipe_raster_scrub(
    surface: SurfacePlane,
    sweep: SweepSpec,
    scrub: ScrubSpec,
) -> Iterable[PoseStamped]:
    anchors = list(compile_sweep_raster(surface, sweep))
    out: list[PoseStamped] = []

    for a in anchors:
        ps = PoseStamped()
        ps.header.frame_id = a.frame_id
        ps.pose.position.x = float(a.p[0])
        ps.pose.position.y = float(a.p[1])
        ps.pose.position.z = float(a.p[2])
        out.append(ps)
        out.extend(compile_scrub_circle(a, surface, scrub, margin=float(sweep.margin)))

    return out
