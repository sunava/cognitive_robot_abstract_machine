from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, Optional

import numpy as np
from geometry_msgs.msg import PoseStamped

from pycram.demos.thesis.primitives.seperation_devision import (
    CutAnchor,
    new_pose,
    set_xyz,
    SliceSpec,
    compile_slice_kernel,
)


class CutSide(Enum):
    POS_Y = auto()
    NEG_Y = auto()


@dataclass(frozen=True)
class SawSliceSpec:
    slice_thickness: float
    tool_half_length: float
    prelift_z: float
    insert_z: float
    y_standoff: float
    tilt_y_deg: float
    pitch_x_deg: float
    stroke_x: float
    rotate_y_shift: float
    rotate_x_shift: float
    return_y_shift: float
    return_x_shift: float
    final_lift_z: float


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


def axis_angle_quat(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / max(np.linalg.norm(a), 1e-12)
    th = math.radians(float(angle_deg))
    s = math.sin(0.5 * th)
    return np.array([a[0] * s, a[1] * s, a[2] * s, math.cos(0.5 * th)], dtype=float)


def rot(ps: PoseStamped, q: np.ndarray) -> None:
    ps.pose.orientation.x = float(q[0])
    ps.pose.orientation.y = float(q[1])
    ps.pose.orientation.z = float(q[2])
    ps.pose.orientation.w = float(q[3])


def bind_cut_side_from_robot_y(angle_y: float) -> CutSide:
    return CutSide.NEG_Y if float(angle_y) >= 0.0 else CutSide.POS_Y


def compile_saw_slice_sequence(
    anchor: CutAnchor,
    side: CutSide,
    spec: SawSliceSpec,
    base_orientation: Optional[np.ndarray] = None,
) -> Iterable[PoseStamped]:
    sgn = +1.0 if side is CutSide.POS_Y else -1.0

    x0 = float(anchor.p[0])
    y0 = float(anchor.p[1]) + sgn * (
        float(spec.tool_half_length) + float(spec.y_standoff)
    )
    z_lift = float(spec.prelift_z)
    z_ins = float(spec.insert_z)

    q = (
        np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        if base_orientation is None
        else np.asarray(base_orientation, dtype=float)
    )

    q_tilt = axis_angle_quat(
        np.array([0.0, 1.0, 0.0], dtype=float), float(spec.tilt_y_deg)
    )
    q = quat_mul(q, q_tilt)

    dz = z_lift - z_ins
    dx = dz * math.tan(math.radians(float(spec.tilt_y_deg)))

    ps0 = new_pose(anchor.frame_id)
    set_xyz(ps0, x0, y0, z_lift)
    rot(ps0, q)

    ps1 = new_pose(anchor.frame_id)
    set_xyz(ps1, x0 + dx, y0, z_ins)
    rot(ps1, q)

    ps2 = new_pose(anchor.frame_id)
    set_xyz(ps2, x0 + dx + float(spec.stroke_x), y0, z_ins)
    rot(ps2, q)

    q_pitch = axis_angle_quat(
        np.array([1.0, 0.0, 0.0], dtype=float), float(spec.pitch_x_deg)
    )
    q2 = quat_mul(q, q_pitch)

    ps3 = new_pose(anchor.frame_id)
    set_xyz(
        ps3,
        x0 + dx + float(spec.stroke_x),
        y0 - sgn * float(spec.rotate_y_shift),
        z_ins,
    )
    rot(ps3, q2)

    ps4 = new_pose(anchor.frame_id)
    set_xyz(ps4, x0 + dx, y0 - sgn * float(spec.return_y_shift), z_ins)
    rot(ps4, q2)

    ps5 = new_pose(anchor.frame_id)
    set_xyz(
        ps5,
        x0 + dx - float(spec.return_x_shift),
        y0 - sgn * float(spec.return_y_shift),
        float(spec.final_lift_z),
    )
    rot(ps5, q2)

    ps6 = new_pose(anchor.frame_id)
    set_xyz(ps6, x0, y0, z_lift)
    rot(ps6, q)

    return [ps0, ps1, ps2, ps3, ps4, ps5, ps6]


def compile_slice_phases_basic(
    anchor: CutAnchor,
    obj_half_extents: np.ndarray,
    spec: SliceSpec,
    tilt_y_deg: float = 0.0,
) -> Iterable[PoseStamped]:
    hz = float(obj_half_extents[2])
    z_top = hz + float(spec.z_clearance)

    x0 = float(anchor.p[0])
    y0 = float(anchor.p[1])

    p0 = new_pose(anchor.frame_id)
    set_xyz(p0, x0, y0, z_top)

    p1 = compile_slice_kernel(anchor, obj_half_extents, spec, tilt_y_deg=tilt_y_deg)

    p2 = new_pose(anchor.frame_id)
    set_xyz(p2, p1.pose.position.x, p1.pose.position.y, z_top)

    return [p0, p1, p2]
