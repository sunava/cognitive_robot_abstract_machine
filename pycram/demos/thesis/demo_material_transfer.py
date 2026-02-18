"""Demo for material transfer primitives (discharge and shake)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from primitives.material_transfer import (
    BoundaryAnchor,
    DischargeSpec,
    ShakeSpec,
    compile_boundary_discharge,
    compile_boundary_shake,
)
from demos.thesis_new.visual.trajectory_plotter import TrajectoryPlotter


@dataclass(frozen=True)
class MaterialTransferDemo:
    """Runs a simple material transfer demo with plotting."""

    plotter: TrajectoryPlotter

    def demo_boundary_flow(self) -> None:
        """Plot discharge and shake boundary flow trajectories."""
        anchor = BoundaryAnchor(
            frame_id="cup_rim",
            p=np.array([0.0, 0.0, 0.10], dtype=float),
            n=np.array([1.0, 0.2, -0.3], dtype=float),
        )

        discharge_specification = DischargeSpec(
            steps=120, f_start=0.0, f_step=0.0012, f_max=0.12, epsilon=1e-6
        )
        shake_specification = ShakeSpec(
            steps=240, f_bias=0.04, f_amp=0.03, omega=0.22, epsilon=0.0
        )

        discharge_poses = list(
            compile_boundary_discharge(anchor, discharge_specification)
        )
        shake_poses = list(compile_boundary_shake(anchor, shake_specification))

        self.plotter.plot_poses(
            discharge_poses, "Boundary discharge: p = p* + f(t) n", anchor=anchor.p
        )
        self.plotter.plot_poses(
            shake_poses,
            "Boundary shake: p = p* + (bias + amp sin(ωt)) n",
            anchor=anchor.p,
        )
        self.plotter.show()


def main() -> None:
    """Entry point for the material transfer demo."""
    demo = MaterialTransferDemo(plotter=TrajectoryPlotter())
    demo.demo_boundary_flow()


if __name__ == "__main__":
    main()
