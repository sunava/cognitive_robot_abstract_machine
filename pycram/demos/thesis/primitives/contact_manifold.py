from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from pycram.src.pycram.datastructures.pose import PoseStamped
from semantic_digital_twin.world_description.world_entity import Body


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("zero-length vector")
    return v / n


@dataclass(frozen=True)
class ContactAnchor:
    frame_id: Body
    p0: np.ndarray
    n: np.ndarray
    t1: Optional[np.ndarray] = None
    t2: Optional[np.ndarray] = None

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = _unit(np.asarray(self.n, dtype=float).reshape(3))
        if self.t1 is None or self.t2 is None:
            a = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(a, n))) > 0.9:
                a = np.array([0.0, 1.0, 0.0], dtype=float)
            t1 = _unit(np.cross(n, a))
            t2 = _unit(np.cross(n, t1))
            return t1, t2, n
        t1 = _unit(np.asarray(self.t1, dtype=float).reshape(3))
        t2 = _unit(np.asarray(self.t2, dtype=float).reshape(3))
        if abs(float(np.dot(t1, n))) > 1e-6 or abs(float(np.dot(t2, n))) > 1e-6:
            raise ValueError("tangent basis not orthogonal to normal")
        return t1, t2, n


def _new_pose(frame_id: Body) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    return ps


def _set_xyz(ps: PoseStamped, x: float, y: float, z: float) -> None:
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)


def compile_contact_manifold(
    anchor: ContactAnchor,
    d: np.ndarray,
    q_uv: np.ndarray,
) -> Iterable[PoseStamped]:
    p0 = np.asarray(anchor.p0, dtype=float).reshape(3)
    t1, t2, n = anchor.basis()

    dd = np.asarray(d, dtype=float).reshape(-1)
    qq = np.asarray(q_uv, dtype=float)
    if qq.ndim != 2 or qq.shape[1] != 2:
        raise ValueError("q_uv must be (N,2)")
    if qq.shape[0] != dd.shape[0]:
        raise ValueError("d and q_uv must have same length")

    out: list[PoseStamped] = []
    for i in range(dd.shape[0]):
        p = p0 - float(dd[i]) * n + float(qq[i, 0]) * t1 + float(qq[i, 1]) * t2
        ps = _new_pose(anchor.frame_id)
        _set_xyz(ps, float(p[0]), float(p[1]), float(p[2]))
        out.append(ps)

    return out
