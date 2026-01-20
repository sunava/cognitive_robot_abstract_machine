"""Demo for surface scrub primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycram.demos.thesis.primitives.surface_interaction import (
    SurfacePlane,
    bind_surface_anchor,
    ScrubSpec,
    compile_scrub_circle,
)
from pycram.demos.thesis.tools.trajectory_plotter import TrajectoryPlotter


@dataclass(frozen=True)
class SurfaceScrubDemo:
    """Runs scrub trajectory variants over a surface."""

    plotter: TrajectoryPlotter

    def demo_scrub_variants(self) -> None:
        """Plot multiple scrub radius/period variants."""
        surface = self._make_surface()
        anchor = bind_surface_anchor(surface, margin=0.005)
        if anchor is None:
            raise RuntimeError("surface anchor is None")

        variants = [
            ScrubSpec(radius=0.02, points_per_cycle=40, cycles=2),
            ScrubSpec(radius=0.035, points_per_cycle=48, cycles=3),
        ]

        for scrub_specification in variants:
            poses = list(
                compile_scrub_circle(anchor, surface, scrub_specification, margin=0.005)
            )
            title = (
                "Scrub circle: radius="
                f"{scrub_specification.radius}, cycles={scrub_specification.cycles}"
            )
            self.plotter.plot_poses(poses, title, anchor=anchor.p)

        self.plotter.show()

    def _make_surface(self) -> SurfacePlane:
        """Create a simple rectangular surface model."""
        scale_x, scale_y, scale_z = 0.30, 0.10, 0.10
        half_extents = np.array([0.5 * scale_x, 0.5 * scale_y], dtype=float)
        return SurfacePlane(
            frame_id="scrub_surface",
            origin=np.array([0.0, 0.0, 0.5 * scale_z], dtype=float),
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            half_extents_uv=half_extents,
            t1=np.array([1.0, 0.0, 0.0], dtype=float),
            t2=np.array([0.0, 1.0, 0.0], dtype=float),
        )


def main() -> None:
    """Entry point for the surface scrub demo."""
    demo = SurfaceScrubDemo(plotter=TrajectoryPlotter())
    demo.demo_scrub_variants()


if __name__ == "__main__":
    main()
