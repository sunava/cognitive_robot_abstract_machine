from dataclasses import dataclass

import numpy as np


def ramp(tau, tau_end, d_max):
    if tau <= 0.0:
        return 0.0
    if tau >= tau_end:
        return float(d_max)
    return float(d_max) * (tau / tau_end)


def planar_spiral_xy(tau, r0, r1, cycles):
    r = r0 + (r1 - r0) * tau
    ang = 2.0 * np.pi * cycles * tau
    return np.array([r * np.cos(ang), r * np.sin(ang), 0.0], dtype=float)


def planar_sweep_x(tau, length, cycles):
    s = float(length) * np.sin(2.0 * np.pi * float(cycles) * tau)
    return np.array([s, 0.0, 0.0], dtype=float)


@dataclass(frozen=True)
class ShearProfile:
    depth_max: float
    depth_ramp_end: float
    shear_amp: float
    shear_cycles: float


@dataclass(frozen=True)
class SpiralProfile:
    r0: float
    r1: float
    cycles: float


@dataclass(frozen=True)
class SweepProfile:
    length: float
    cycles: float


def oscillatory_shear_local_profiled(tau, prof: ShearProfile):
    d = ramp(tau, tau_end=prof.depth_ramp_end, d_max=prof.depth_max)
    s = float(prof.shear_amp) * np.sin(2.0 * np.pi * float(prof.shear_cycles) * tau)
    return np.array([s, 0.0, -d], dtype=float)


def sample_local_curve(local_curve, taus):
    pts = np.empty((len(taus), 3), dtype=float)
    for i, u in enumerate(taus):
        pts[i] = local_curve(float(u))
    return pts


def clamp_to_aabb(q_local, mins, maxs, margin=0.0):
    mins = np.asarray(mins, dtype=float) + float(margin)
    maxs = np.asarray(maxs, dtype=float) - float(margin)
    return np.minimum(np.maximum(q_local, mins), maxs)


def clamp_to_cylinder_xy(q_local, radius, z_min, z_max, margin=0.0):
    q = np.asarray(q_local, dtype=float).reshape(3)
    r = float(radius) - float(margin)
    xy = q[:2]
    r_xy = np.linalg.norm(xy)
    if r_xy > r and r_xy > 1e-9:
        xy = (r / r_xy) * xy
    z = np.clip(q[2], float(z_min) + float(margin), float(z_max) - float(margin))
    return np.array([xy[0], xy[1], z], dtype=float)


def make_constrained_curve(local_curve, constraint_fn):
    return lambda tau: constraint_fn(local_curve(float(tau)))
