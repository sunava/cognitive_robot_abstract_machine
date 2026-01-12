from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped


@dataclass(frozen=True)
class BoundaryAnchor:
    frame_id: str
    p: np.ndarray
    n: np.ndarray


@dataclass(frozen=True)
class DischargeSpec:
    steps: int
    f_start: float
    f_step: float
    f_max: Optional[float] = None
    epsilon: float = 0.0


@dataclass(frozen=True)
class ShakeSpec:
    steps: int
    f_bias: float
    f_amp: float
    omega: float
    epsilon: float = 0.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("zero-length vector")
    return v / n


def _new_pose(frame_id: str) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    return ps


def _set_xyz(ps: PoseStamped, x: float, y: float, z: float) -> None:
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)


def compile_boundary_discharge(
    anchor: BoundaryAnchor, spec: DischargeSpec
) -> Iterable[PoseStamped]:
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


def compile_boundary_shake(
    anchor: BoundaryAnchor, spec: ShakeSpec
) -> Iterable[PoseStamped]:
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


def _xyz(wps: Iterable[PoseStamped]) -> np.ndarray:
    wps = list(wps)
    return np.array(
        [[w.pose.position.x, w.pose.position.y, w.pose.position.z] for w in wps],
        dtype=float,
    )


def plot_3d_path(
    xyz: np.ndarray, title: str, anchor: Optional[np.ndarray] = None
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], marker="o")
    ax.scatter([xyz[-1, 0]], [xyz[-1, 1]], [xyz[-1, 2]], marker="x")
    if anchor is not None:
        ax.scatter([anchor[0]], [anchor[1]], [anchor[2]], marker="^")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    plt.show()


def plot_f(values: np.ndarray, title: str) -> None:
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("f")
    plt.show()


def demo_boundary_flow() -> None:
    anchor = BoundaryAnchor(
        frame_id="cup_rim",
        p=np.array([0.0, 0.0, 0.10], dtype=float),
        n=np.array([1.0, 0.2, -0.3], dtype=float),
    )

    discharge = DischargeSpec(
        steps=120, f_start=0.0, f_step=0.0012, f_max=0.12, epsilon=1e-6
    )
    shake = ShakeSpec(steps=240, f_bias=0.04, f_amp=0.03, omega=0.22, epsilon=0.0)

    wps_lin = list(compile_boundary_discharge(anchor, discharge))
    wps_shk = list(compile_boundary_shake(anchor, shake))

    xyz_lin = _xyz(wps_lin)
    xyz_shk = _xyz(wps_shk)

    plot_3d_path(xyz_lin, "Boundary discharge: p = p* + f(t) n", anchor=anchor.p)
    plot_3d_path(
        xyz_shk, "Boundary shake: p = p* + (bias + amp sin(ωt)) n", anchor=anchor.p
    )

    n = _unit(anchor.n)
    f_lin = np.dot(xyz_lin - anchor.p.reshape(1, 3), n.reshape(3, 1)).reshape(-1)
    f_shk = np.dot(xyz_shk - anchor.p.reshape(1, 3), n.reshape(3, 1)).reshape(-1)

    plot_f(f_lin, "Discharge schedule f(t) over steps")
    plot_f(f_shk, "Shake schedule f(t) over steps")


if __name__ == "__main__":
    demo_boundary_flow()
