from dataclasses import dataclass

import numpy as np


def rot_x(angle):
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_y(angle):
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rot_z(angle):
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rpy_matrix(roll=0.0, pitch=0.0, yaw=0.0):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def constant_orientation():
    return lambda tau: np.eye(3, dtype=float)


def fixed_rpy(roll=0.0, pitch=0.0, yaw=0.0):
    R = rpy_matrix(roll=roll, pitch=pitch, yaw=yaw)
    return lambda tau: R


def tilt_about_local_y(max_angle, ramp_in=0.3, hold_until=0.7):
    max_angle = float(max_angle)
    ramp_in = float(ramp_in)
    hold_until = float(hold_until)

    def profile(tau):
        tau = float(np.clip(tau, 0.0, 1.0))
        if tau <= ramp_in:
            angle = max_angle * (tau / max(ramp_in, 1e-6))
        elif tau <= hold_until:
            angle = max_angle
        else:
            angle = max_angle * max(
                0.0, 1.0 - (tau - hold_until) / max(1.0 - hold_until, 1e-6)
            )
        return rot_y(angle)

    return profile


def ramp(tau, tau_end, d_max):
    """Linear ramp from 0 to d_max over tau_end."""
    if tau <= 0.0:
        return 0.0
    if tau >= tau_end:
        return float(d_max)
    return float(d_max) * (tau / tau_end)


def planar_spiral_xy(tau, r0, r1, cycles):
    """Planar spiral in XY with linearly growing radius."""
    r = r0 + (r1 - r0) * tau
    ang = 2.0 * np.pi * cycles * tau
    return np.array([r * np.cos(ang), r * np.sin(ang), 0.0], dtype=float)


def planar_sweep_x(tau, length, cycles):
    """Sinusoidal sweep along the X axis."""
    s = float(length) * np.sin(2.0 * np.pi * float(cycles) * tau)
    return np.array([s, 0.0, 0.0], dtype=float)


def planar_raster_xy(tau, width, height, lanes):
    """Raster scan covering a rectangle in XY."""
    w = float(width)
    h = float(height)
    n = max(2, int(lanes))
    u = float(np.clip(tau, 0.0, 1.0)) * n
    lane = int(np.floor(u))
    if lane >= n:
        lane = n - 1
    local_t = u - lane

    x0 = -0.5 * w
    x1 = 0.5 * w
    if (lane % 2) == 0:
        x = x0 + (x1 - x0) * local_t
    else:
        x = x1 - (x1 - x0) * local_t

    y = -0.5 * h + (h * lane / float(n - 1))
    return np.array([x, y, 0.0], dtype=float)


@dataclass(frozen=True)
class ShearProfile:
    depth_max: float
    depth_ramp_end: float
    shear_amp: float
    shear_cycles: float


@dataclass(frozen=True)
class ShearXYProfile:
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
    """Oscillatory shear with a monotone depth profile."""
    d = ramp(tau, tau_end=prof.depth_ramp_end, d_max=prof.depth_max)
    s = float(prof.shear_amp) * np.sin(2.0 * np.pi * float(prof.shear_cycles) * tau)
    return np.array([s, 0.0, -d], dtype=float)


def oscillatory_shear_xy_profiled(tau, prof: ShearXYProfile):
    """Oscillatory shear in XY plane with no depth change."""
    ang = 2.0 * np.pi * float(prof.shear_cycles) * tau
    s = float(prof.shear_amp)
    return np.array([s * np.sin(ang), s * np.cos(ang), 0.0], dtype=float)


def sample_local_curve(local_curve, taus):
    """Sample a local curve for a list of tau values."""
    pts = np.empty((len(taus), 3), dtype=float)
    for i, u in enumerate(taus):
        pts[i] = local_curve(float(u))
    return pts


def clamp_to_aabb(q_local, mins, maxs, margin=0.0):
    """Clamp a local point into an axis-aligned bounding box."""
    mins = np.asarray(mins, dtype=float) + float(margin)
    maxs = np.asarray(maxs, dtype=float) - float(margin)
    return np.minimum(np.maximum(q_local, mins), maxs)


def clamp_to_cylinder_xy(q_local, radius, z_min, z_max, margin=0.0):
    """Clamp a point to a vertical cylinder in XY and Z bounds."""
    q = np.asarray(q_local, dtype=float).reshape(3)
    r = float(radius) - float(margin)
    xy = q[:2]
    r_xy = np.linalg.norm(xy)
    if r_xy > r and r_xy > 1e-9:
        xy = (r / r_xy) * xy
    z = np.clip(q[2], float(z_min) + float(margin), float(z_max) - float(margin))
    return np.array([xy[0], xy[1], z], dtype=float)


def make_constrained_curve(local_curve, constraint_fn):
    """Wrap a curve so it respects a constraint function."""
    return lambda tau: constraint_fn(local_curve(float(tau)))
