"""Demo for separation and division primitives and phases."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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
from pycram.demos.thesis.tools.trajectory_plotter import TrajectoryPlotter

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Box, Scale


@dataclass(frozen=True)
class SeparationDivisionDemo:
    """Runs a set of separation/slicing demo plots."""

    plotter: TrajectoryPlotter

    def demo_primitives_generic(self) -> None:
        """Plot generic separation primitives (press, slice, saw, pull)."""
        common = dict(length=0.18, depth=0.03, n=120, normal_force=18.0)

        for mode in (SepMode.PRESS, SepMode.SLICE, SepMode.SAW):
            separation_specification = SeparationSpec(mode=mode, **common)
            positions = compile_separation(separation_specification)
            self.plotter.plot_positions(
                positions, f"Primitive: {mode.name.lower()} in (x,y,z)"
            )

        pull_specification = SeparationSpec(
            mode=SepMode.PULL_APART, length=0.0, depth=0.0, n=120, pull_gap=0.14
        )
        left_positions, right_positions = compile_separation(pull_specification)
        self.plotter.plot_position_sets(
            [left_positions, right_positions], "Primitive: pull_apart in (x,y,z)"
        )

    def demo_primitive_slice_bound_to_box(self) -> None:
        """Plot slice trajectories bound to a simple box model."""
        box = self._make_box()
        half_extents = self._box_half_extents(box)

        slice_specification = SliceSpec(
            slice_thickness=0.03, z_clearance=0.03, z_cut=0.0, margin_xy=0.005
        )
        anchors = bind_slice_anchors_along_x(
            "cut_box", half_extents, slice_specification
        )

        poses = []
        for anchor in anchors:
            poses.extend(
                list(
                    compile_slice_phases_basic(
                        anchor, half_extents, slice_specification, tilt_y_deg=15.0
                    )
                )
            )

        self.plotter.plot_poses(
            poses, "Primitive (bound): slice down-through-up in (x,y,z)"
        )

    def demo_phases_saw_sequence_on_box(self) -> None:
        """Plot a saw-slice phase sequence over a box."""
        box = self._make_box()
        half_extents = self._box_half_extents(box)

        slice_specification = SliceSpec(
            slice_thickness=0.03, z_clearance=0.03, z_cut=0.0, margin_xy=0.005
        )
        anchors = bind_slice_anchors_along_x(
            "cut_box", half_extents, slice_specification
        )

        phase_specification = SawSliceSpec(
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

        poses = []
        for anchor in anchors[: max(1, len(anchors) // 2)]:
            poses.extend(
                list(
                    compile_saw_slice_sequence(
                        anchor, CutSide.POS_Y, phase_specification
                    )
                )
            )

        self.plotter.plot_poses(poses, "Phases: saw slicing sequence in (x,y,z)")

    def demo_slice_kernel_on_box(self) -> None:
        """Plot the slice kernel pose bound to a box."""
        box = self._make_box()
        half_extents = self._box_half_extents(box)

        slice_specification = SliceSpec(
            slice_thickness=0.03, z_clearance=0.03, z_cut=0.0, margin_xy=0.005
        )
        anchors = bind_slice_anchors_along_x(
            "cut_box", half_extents, slice_specification
        )

        anchor = anchors[0]
        pose = compile_slice_kernel(
            anchor, half_extents, slice_specification, tilt_y_deg=15.0
        )

        self.plotter.plot_poses(
            [pose], "Primitive: slice kernel (target pose) in (x,y,z)"
        )

    def demo_press_vs_slice_kernel(self) -> None:
        """Compare press vs slice penetration kernel traces."""
        depth = 0.03
        step_count = 120

        press_positions = self._press_trace(depth=depth, step_count=step_count)
        slice_positions = self._slice_trace(
            depth=depth, step_count=step_count, tilt_y_deg=4.0
        )

        self.plotter.plot_position_sets(
            [press_positions, slice_positions], "Kernel: press vs slice in (x,y,z)"
        )

    def _make_box(self) -> Body:
        """Create a simple box body for demos."""
        return Body(
            name=PrefixedName("cut_box"),
            collision=ShapeCollection([Box(scale=Scale(0.18, 0.10, 0.06))]),
        )

    def _box_half_extents(self, box: Body) -> np.ndarray:
        """Return the half extents for a box volume model."""
        volume = volume_from_body_collision(box, padding=0.0)
        if volume is None:
            raise RuntimeError("no volume model from body collision")
        return volume.half_extents

    def _press_trace(self, depth: float, step_count: int) -> np.ndarray:
        """Create a straight penetration trace."""
        point = np.zeros(3, dtype=float)
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        return compile_penetration_kernel(point, normal, depth, step_count)

    def _slice_trace(
        self, depth: float, step_count: int, tilt_y_deg: float
    ) -> np.ndarray:
        """Create a tilted penetration trace."""
        point = np.zeros(3, dtype=float)
        tilt = math.radians(float(tilt_y_deg))
        normal = np.array([math.sin(tilt), 0.0, math.cos(tilt)], dtype=float)
        return compile_penetration_kernel(point, normal, depth, step_count)


def main() -> None:
    """Run all separation division demo plots."""
    demo = SeparationDivisionDemo(plotter=TrajectoryPlotter())
    demo.demo_primitives_generic()
    demo.demo_primitive_slice_bound_to_box()
    demo.demo_phases_saw_sequence_on_box()
    demo.demo_slice_kernel_on_box()
    demo.demo_press_vs_slice_kernel()
    demo.plotter.show()


if __name__ == "__main__":
    main()
