"""Standalone demo of primitive trajectories for thesis figures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def unit(v: np.ndarray) -> np.ndarray:
    """Normalize a 3D vector and raise on zero length."""
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("zero-length vector")
    return v / n


def tangent_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an orthonormal tangent basis for a given normal."""
    n = unit(n)
    a = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(a, n))) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=float)
    t1 = unit(np.cross(n, a))
    t2 = unit(np.cross(n, t1))
    return t1, t2, n


@dataclass(frozen=True)
class ContactAnchor:
    """Anchor frame for contact primitives."""

    p0: np.ndarray
    n: np.ndarray
    t1: Optional[np.ndarray] = None
    t2: Optional[np.ndarray] = None

    def basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return tangent basis vectors and the normal."""
        n = unit(self.n)
        if self.t1 is None or self.t2 is None:
            return tangent_basis(n)
        t1 = unit(self.t1)
        t2 = unit(self.t2)
        return t1, t2, n


def g_contact(anchor: ContactAnchor, d: np.ndarray, q_uv: np.ndarray) -> np.ndarray:
    """Map manifold coordinates (d, q_uv) to 3D contact points."""
    p0 = np.asarray(anchor.p0, dtype=float).reshape(3)
    t1, t2, n = anchor.basis()

    d = np.asarray(d, dtype=float).reshape(-1)
    q = np.asarray(q_uv, dtype=float)
    if q.ndim != 2 or q.shape[1] != 2 or q.shape[0] != d.shape[0]:
        raise ValueError("q_uv must be (N,2) and match d length")

    p = (
        p0[None, :]
        - d[:, None] * n[None, :]
        + q[:, 0:1] * t1[None, :]
        + q[:, 1:2] * t2[None, :]
    )
    return p


def g_volume(
    p_star: np.ndarray, r: np.ndarray, theta: np.ndarray, dz: np.ndarray
) -> np.ndarray:
    """Parametrize a helical volume trajectory around p_star."""
    p_star = np.asarray(p_star, dtype=float).reshape(3)
    r = np.asarray(r, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    dz = np.asarray(dz, dtype=float).reshape(-1)
    if not (r.shape[0] == theta.shape[0] == dz.shape[0]):
        raise ValueError("r, theta, dz must have same length")
    xy0 = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    return (
        p_star[None, :]
        + r[:, None] * xy0
        + dz[:, None] * np.array([0.0, 0.0, 1.0], dtype=float)[None, :]
    )


def g_boundary_flow(p_star: np.ndarray, n: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Compute boundary flow points along a normal direction."""
    p_star = np.asarray(p_star, dtype=float).reshape(3)
    n = unit(n)
    f = np.asarray(f, dtype=float).reshape(-1)
    return p_star[None, :] + f[:, None] * n[None, :]


def linspace01(n: int) -> np.ndarray:
    """Return n points between 0 and 1 (inclusive)."""
    if n <= 1:
        return np.zeros((1,), dtype=float)
    return np.linspace(0.0, 1.0, n, dtype=float)


def primitive_press(anchor: ContactAnchor, depth: float, n: int) -> np.ndarray:
    """Linear penetration into the surface without tangential motion."""
    t = linspace01(n)
    d = float(depth) * t
    q = np.zeros((t.shape[0], 2), dtype=float)
    return g_contact(anchor, d, q)


def primitive_saw(
    anchor: ContactAnchor, depth: float, amp: float, hz: float, n: int
) -> np.ndarray:
    """Sawing primitive with tangential oscillation during penetration."""
    t = linspace01(n)
    d = float(depth) * t
    q = np.zeros((t.shape[0], 2), dtype=float)
    q[:, 1] = float(amp) * np.sin(2.0 * math.pi * float(hz) * t)
    return g_contact(anchor, d, q)


def primitive_wipe(
    anchor: ContactAnchor, radius: float, cycles: int, ppc: int
) -> np.ndarray:
    """Circular wipe/scrub trajectory on the contact manifold."""
    n_pts = int(cycles) * int(ppc)
    if n_pts <= 0:
        return np.zeros((0, 3), dtype=float)
    a = 2.0 * math.pi * (np.arange(n_pts, dtype=float) / float(ppc))
    d = np.zeros((n_pts,), dtype=float)
    q = np.column_stack([float(radius) * np.cos(a), float(radius) * np.sin(a)])
    return g_contact(anchor, d, q)


def primitive_volume_agitation(
    p_star: np.ndarray,
    turns: int,
    angle_step_deg: float,
    r0: float,
    dr: float,
    dz: float,
) -> np.ndarray:
    """Create a spiral agitation trajectory inside a volume."""
    angle_step = math.radians(float(angle_step_deg))
    if angle_step <= 0.0:
        return np.zeros((0, 3), dtype=float)
    steps_per_turn = int(round((2.0 * math.pi) / angle_step))
    n = int(turns) * steps_per_turn
    if n <= 0:
        return np.zeros((0, 3), dtype=float)

    i = np.arange(n, dtype=float)
    r = float(r0) + float(dr) * i
    theta = angle_step * i
    dzv = float(dz) * i
    return g_volume(p_star, r, theta, dzv)


def primitive_shake_as_saw_on_spot(
    p_star: np.ndarray, outflow_n: np.ndarray, amp: float, hz: float, n: int
) -> np.ndarray:
    """Boundary shake modeled as a saw motion along the normal."""
    t = linspace01(n)
    f = float(amp) * np.sin(2.0 * math.pi * float(hz) * t)
    return g_boundary_flow(p_star, outflow_n, f)


def plot_3d(xyz: np.ndarray, title: str) -> None:
    """Plot a 3D trajectory with start/end markers."""
    if xyz.shape[0] == 0:
        return
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], marker="o")
    ax.scatter([xyz[-1, 0]], [xyz[-1, 1]], [xyz[-1, 2]], marker="x")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    plt.show()


def plot_3d_phases(phases: list[tuple[str, np.ndarray]]) -> None:
    """Plot multiple 3D phases with labels."""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for name, xyz in phases:
        if xyz.shape[0] == 0:
            continue
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=name)
        ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], marker="o")
        ax.scatter([xyz[-1, 0]], [xyz[-1, 1]], [xyz[-1, 2]], marker="x")

    ax.set_title("Phases composed from primitives")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    plt.show()


def connect_phases(phases: list[np.ndarray]) -> list[np.ndarray]:
    """Translate phases to be contiguous in space."""
    out = []
    prev_end = None
    for xyz in phases:
        if xyz.shape[0] == 0:
            out.append(xyz)
            continue
        if prev_end is None:
            out.append(xyz)
            prev_end = xyz[-1, :]
            continue
        delta = prev_end - xyz[0, :]
        xyz2 = xyz + delta[None, :]
        out.append(xyz2)
        prev_end = xyz2[-1, :]
    return out


def demo() -> None:
    """Run the demo and show plots for multiple primitives."""
    cut_anchor = ContactAnchor(
        p0=np.array([0.0, 0.0, 0.10], dtype=float),
        n=np.array([0.0, 0.0, 1.0], dtype=float),
    )

    surface_anchor = ContactAnchor(
        p0=np.array([0.15, 0.0, 0.10], dtype=float),
        n=np.array([0.2, 0.0, 1.0], dtype=float),
    )

    bowl_anchor = np.array([0.0, -0.15, 0.06], dtype=float)

    rim_anchor = np.array([-0.15, 0.15, 0.10], dtype=float)
    outflow_n = np.array([1.0, 0.2, -0.4], dtype=float)

    press = primitive_press(cut_anchor, depth=0.04, n=120)
    saw = primitive_saw(cut_anchor, depth=0.04, amp=0.010, hz=6.0, n=240)
    wipe = primitive_wipe(surface_anchor, radius=0.03, cycles=3, ppc=80)
    mix = primitive_volume_agitation(
        bowl_anchor, turns=3, angle_step_deg=15.0, r0=0.0, dr=0.0012, dz=0.0003
    )
    shake = primitive_shake_as_saw_on_spot(
        rim_anchor, outflow_n, amp=0.05, hz=5.0, n=300
    )

    plot_3d(press, "Primitive: press (contact manifold, q=0)")
    plot_3d(saw, "Primitive: saw (contact manifold, d+oscillation)")
    plot_3d(wipe, "Primitive: wipe/scrub (contact manifold, d=0)")
    plot_3d(mix, "Primitive: volume agitation (mixing)")
    plot_3d(shake, "Primitive: shake as sawing-on-spot (boundary flow)")

    approach = np.column_stack(
        [
            np.linspace(-0.20, 0.0, 80, dtype=float),
            np.linspace(-0.10, 0.0, 80, dtype=float),
            np.linspace(0.20, 0.10, 80, dtype=float),
        ]
    )
    contact_wipe = wipe
    cut_saw = saw
    discharge_shake = shake

    phases_raw = [approach, contact_wipe, cut_saw, discharge_shake, mix]
    phases = connect_phases(phases_raw)

    named = [
        ("approach", phases[0]),
        ("wipe", phases[1]),
        ("saw cut", phases[2]),
        ("shake/discharge", phases[3]),
        ("mix", phases[4]),
    ]
    plot_3d_phases(named)


if __name__ == "__main__":
    demo()
