"""Utility for plotting 3D trajectories from pose sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as matplotlib_pyplot


@dataclass(frozen=True)
class TrajectoryPlotter:
    """Simple matplotlib-based plotter for pose trajectories."""

    show_plots: bool = True

    def _collect_positions_from_poses(self, poses: Iterable) -> np.ndarray:
        """Convert an iterable of poses into an (N,3) array."""
        positions = [
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            for pose in poses
        ]
        return np.array(positions, dtype=float)

    def _plot_positions(
        self,
        positions: np.ndarray,
        title: str,
        anchor: Optional[np.ndarray] = None,
    ) -> None:
        """Render a 3D plot for a set of positions."""
        if positions.size == 0:
            return

        figure = matplotlib_pyplot.figure()
        axes = figure.add_subplot(projection="3d")
        axes.plot(positions[:, 0], positions[:, 1], positions[:, 2])
        axes.scatter(
            [positions[0, 0]], [positions[0, 1]], [positions[0, 2]], marker="o"
        )
        axes.scatter(
            [positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], marker="x"
        )
        if anchor is not None:
            axes.scatter([anchor[0]], [anchor[1]], [anchor[2]], marker="^")

        axes.set_title(title)
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_zlabel("z")
        axes.set_box_aspect((1, 1, 1))

    def plot_poses(
        self, poses: Iterable, title: str, anchor: Optional[np.ndarray] = None
    ) -> None:
        """Plot an iterable of pose-like objects."""
        positions = self._collect_positions_from_poses(poses)
        self._plot_positions(positions, title, anchor=anchor)

    def plot_positions(
        self, positions: np.ndarray, title: str, anchor: Optional[np.ndarray] = None
    ) -> None:
        """Plot a set of positions directly."""
        positions_array = np.array(positions, dtype=float)
        self._plot_positions(positions_array, title, anchor=anchor)

    def plot_position_sets(
        self,
        position_sets: Sequence[np.ndarray],
        title: str,
        anchors: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        """Plot multiple position sets in one 3D figure."""
        figure = matplotlib_pyplot.figure()
        axes = figure.add_subplot(projection="3d")

        for index, positions in enumerate(position_sets):
            positions_array = np.array(positions, dtype=float)
            if positions_array.size == 0:
                continue
            axes.plot(
                positions_array[:, 0], positions_array[:, 1], positions_array[:, 2]
            )
            axes.scatter(
                [positions_array[0, 0]],
                [positions_array[0, 1]],
                [positions_array[0, 2]],
                marker="o",
            )
            axes.scatter(
                [positions_array[-1, 0]],
                [positions_array[-1, 1]],
                [positions_array[-1, 2]],
                marker="x",
            )
            if anchors is not None and index < len(anchors):
                anchor = anchors[index]
                axes.scatter([anchor[0]], [anchor[1]], [anchor[2]], marker="^")

        axes.set_title(title)
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_zlabel("z")
        axes.set_box_aspect((1, 1, 1))

    def show(self) -> None:
        """Show all open plots if enabled."""
        if not self.show_plots:
            return
        matplotlib_pyplot.show()
