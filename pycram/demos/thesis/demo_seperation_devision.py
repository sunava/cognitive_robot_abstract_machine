from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt

from pycram.demos.thesis.geometry.volume_models import volume_from_body_collision
from pycram.demos.thesis.phases.seperation_devision import (
    SawSliceSpec,
    compile_saw_slice_sequence,
    CutSide,
    compile_slice_phases_basic,
)
from pycram.demos.thesis.primitives.seperation_devision import (
    SliceSpec,
    bind_slice_anchors_along_x,
    SepMode,
    SeparationSpec,
    compile_separation,
    compile_slice_kernel,
    compile_penetration_kernel,
)

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Box, Scale


def plot_3d_curve(xyz: np.ndarray, title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))

    ax.view_init(elev=20, azim=-60)
    plt.show()


def plot_pull(left: np.ndarray, right: np.ndarray, title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(left[:, 0], left[:, 1], left[:, 2])
    ax.plot(right[:, 0], right[:, 1], right[:, 2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))

    ax.view_init(elev=20, azim=-60)
    plt.show()


def poses_to_xyz(wps) -> np.ndarray:
    xy = np.array([[wp.pose.position.x, wp.pose.position.y] for wp in wps], dtype=float)
    zz = np.array([wp.pose.position.z for wp in wps], dtype=float)
    return np.column_stack([xy[:, 0], xy[:, 1], zz])


def make_box() -> Body:
    return Body(
        name=PrefixedName("cut_box"),
        collision=ShapeCollection([Box(scale=Scale(0.18, 0.10, 0.06))]),
    )


def box_half_extents(box: Body) -> np.ndarray:
    volume = volume_from_body_collision(box, padding=0.0)
    if volume is None:
        raise RuntimeError("no volume model from body collision")
    return volume.half_extents


def demo_primitives_generic() -> None:
    common = dict(length=0.18, depth=0.03, n=120, normal_force=18.0)

    for mode in (SepMode.PRESS, SepMode.SLICE, SepMode.SAW):
        spec = SeparationSpec(mode=mode, **common)
        xyz = compile_separation(spec)
        plot_3d_curve(xyz, f"Primitive: {mode.name.lower()} in (x,y,z)")

    pull_spec = SeparationSpec(
        mode=SepMode.PULL_APART, length=0.0, depth=0.0, n=120, pull_gap=0.14
    )
    left, right = compile_separation(pull_spec)
    plot_pull(left, right, "Primitive: pull_apart in (x,y,z)")


def demo_primitive_slice_bound_to_box() -> None:
    box = make_box()
    half_extents = box_half_extents(box)

    spec = SliceSpec(slice_thickness=0.03, z_clearance=0.03, z_cut=0.0, margin_xy=0.005)
    anchors = bind_slice_anchors_along_x(
        getattr(box, "tf_frame", "cut_box"), half_extents, spec
    )

    wps = []
    for a in anchors:
        wps.extend(
            list(compile_slice_phases_basic(a, half_extents, spec, tilt_y_deg=15.0))
        )

    xyz = poses_to_xyz(wps)
    plot_3d_curve(xyz, "Primitive (bound): slice down-through-up in (x,y,z)")


def demo_phases_saw_sequence_on_box() -> None:
    box = make_box()
    half_extents = box_half_extents(box)

    slice_spec = SliceSpec(
        slice_thickness=0.03, z_clearance=0.03, z_cut=0.0, margin_xy=0.005
    )
    anchors = bind_slice_anchors_along_x(
        getattr(box, "tf_frame", "cut_box"), half_extents, slice_spec
    )

    phase_spec = SawSliceSpec(
        slice_thickness=0.03,
        tool_half_length=0.10 / 2.0,
        prelift_z=float(half_extents[2]) + 0.06,
        insert_z=float(half_extents[2]) - 0.01,
        y_standoff=0.02,
        tilt_y_deg=15.0,
        pitch_x_deg=90.0,
        stroke_x=0.08,
        rotate_y_shift=0.005,
        rotate_x_shift=0.0,
        return_y_shift=0.02,
        return_x_shift=0.08,
        final_lift_z=float(half_extents[2]) + 0.06,
    )

    wps = []
    for a in anchors[: max(1, len(anchors) // 2)]:
        wps.extend(list(compile_saw_slice_sequence(a, CutSide.POS_Y, phase_spec)))

    xyz = poses_to_xyz(wps)
    plot_3d_curve(xyz, "Phases: saw slicing sequence in (x,y,z)")


def demo_slice_kernel_on_box() -> None:
    box = make_box()
    half_extents = box_half_extents(box)

    spec = SliceSpec(slice_thickness=0.03, z_clearance=0.03, z_cut=0.0, margin_xy=0.005)
    anchors = bind_slice_anchors_along_x(
        getattr(box, "tf_frame", "cut_box"), half_extents, spec
    )

    a0 = anchors[0]
    p = compile_slice_kernel(a0, half_extents, spec, tilt_y_deg=15.0)

    xyz = poses_to_xyz([p])
    plot_3d_curve(xyz, "Primitive: slice kernel (target pose) in (x,y,z)")


def press_trace(depth: float, n_steps: int) -> np.ndarray:
    p0 = np.zeros(3, dtype=float)
    n = np.array([0.0, 0.0, 1.0], dtype=float)
    return compile_penetration_kernel(p0, n, depth, n_steps)


def slice_trace(depth: float, n_steps: int, tilt_y_deg: float) -> np.ndarray:
    p0 = np.zeros(3, dtype=float)
    th = math.radians(float(tilt_y_deg))
    n = np.array([math.sin(th), 0.0, math.cos(th)], dtype=float)
    return compile_penetration_kernel(p0, n, depth, n_steps)


def demo_press_vs_slice_kernel() -> None:
    depth = 0.03
    n_steps = 120

    press_xyz = press_trace(depth=depth, n_steps=n_steps)
    slice_xyz = slice_trace(depth=depth, n_steps=n_steps, tilt_y_deg=4.0)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot(slice_xyz[:, 0], slice_xyz[:, 1], slice_xyz[:, 2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))

    ax.view_init(elev=20, azim=-60)
    plt.show()


def main() -> None:
    demo_primitives_generic()
    demo_primitive_slice_bound_to_box()
    demo_phases_saw_sequence_on_box()
    demo_slice_kernel_on_box()
    demo_press_vs_slice_kernel()


if __name__ == "__main__":
    main()
