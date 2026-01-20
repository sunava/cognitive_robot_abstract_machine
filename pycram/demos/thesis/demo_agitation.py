"""Demo utilities for volume agitation and mixmap visualization."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pycram.demos.thesis.binding.mixmap import MixmapParams, solve_mixmap
from pycram.demos.thesis.geometry.volume_models import (
    CylinderVolumeModel,
    volume_from_body_collision,
)
from pycram.demos.thesis.primitives.volume_agitation import (
    VolumeAnchor,
    AgitationSpec,
    compile_volume_agitation,
)
from pycram.demos.thesis.tools.visualize_mixmap import clearance_grid

from pycram.demos.thesis.geometry.surface_models import SurfacePlane
from pycram.demos.thesis.primitives.surface_interaction import (
    bind_surface_anchor,
    SweepSpec,
    ScrubSpec,
    compile_wipe_raster_scrub,
)
from pycram.demos.thesis.geometry.volume_models import volume_from_body_collision
from pycram.demos.thesis.primitives.seperation_devision import (
    SliceSpec,
    bind_slice_anchors_along_x,
)

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Box, Scale


def demo_cylinder_mixmap() -> None:
    """Solve and plot the mixmap clearance for a cylinder volume."""
    volume = CylinderVolumeModel(radius=0.12, half_height=0.08)
    params = MixmapParams(z_ratio=0.35, epsilon=0.01, grid_step=0.005)
    result = solve_mixmap(volume, params)
    if result is None:
        raise RuntimeError("no feasible stance in cylinder")

    xs, ys, grid = clearance_grid(volume, result.z_mix, params.grid_step)

    print("CYLINDER MIXMAP")
    print("z_mix:", result.z_mix)
    print("p_in_container:", result.p_in_container)
    print("clearance:", result.clearance)

    plot_mixmap_surface(
        xs,
        ys,
        grid,
        anchor=np.array(
            [result.p_in_container[0], result.p_in_container[1], result.clearance],
            dtype=float,
        ),
        title="Cylinder clearance at z_mix (3D)",
    )


def plot_3d_waypoints(wps, anchor=None, title="Volume agitation (3D)"):
    """Plot waypoint poses as a 3D trajectory."""
    xyz = np.array(
        [[wp.pose.position.x, wp.pose.position.y, wp.pose.position.z] for wp in wps],
        dtype=float,
    )

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


def plot_mixmap_surface(
    xs: np.ndarray,
    ys: np.ndarray,
    grid: np.ndarray,
    anchor: np.ndarray | None = None,
    title: str = "Mixmap clearance surface (3D)",
) -> None:
    """Render the clearance grid as a 3D surface plot."""
    xs_grid, ys_grid = np.meshgrid(xs, ys)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(xs_grid, ys_grid, grid, cmap="viridis", alpha=0.9)
    if anchor is not None:
        ax.scatter([anchor[0]], [anchor[1]], [anchor[2]], marker="x")
    ax.set_title(title)
    ax.set_xlabel("x in container frame")
    ax.set_ylabel("y in container frame")
    ax.set_zlabel("clearance")
    ax.set_box_aspect((1, 1, 1))
    plt.show()


def plot_depth_over_steps(depths: np.ndarray, title: str) -> None:
    """Plot depth values over time/steps in a 3D style plot."""
    steps = np.arange(len(depths), dtype=float)
    zeros = np.zeros_like(steps)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(steps, zeros, depths)
    ax.scatter([steps[0]], [0.0], [depths[0]], marker="o")
    ax.scatter([steps[-1]], [0.0], [depths[-1]], marker="x")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("index")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    plt.show()


def demo_body_box_to_agitation() -> None:
    """Build a box volume, solve mixmap, and plot agitation waypoints."""
    box = Body(
        name=PrefixedName("muh"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
    )

    volume = volume_from_body_collision(box, padding=0.005)
    if volume is None:
        raise RuntimeError("no volume model from body collision")

    params = MixmapParams(z_ratio=0.35, epsilon=0.01, grid_step=0.005)
    result = solve_mixmap(volume, params)
    if result is None:
        raise RuntimeError("no feasible stance in box")

    anchor = VolumeAnchor(
        frame_id=getattr(box, "tf_frame", "box"), p=result.p_in_container
    )
    spec = AgitationSpec(
        turns=3, angle_step_deg=15.0, radius_step=0.0015, z_step=0.0002
    )

    wps = list(compile_volume_agitation(anchor, spec, volume=volume, epsilon=0.01))
    xyz_anchor = np.array(
        [result.p_in_container[0], result.p_in_container[1], result.z_mix], dtype=float
    )
    plot_3d_waypoints(wps, anchor=xyz_anchor, title="Body box -> volume agitation (3D)")

    print("BODY BOX -> VOLUME AGITATION")
    print(
        "bounds_xy:", volume.inner_bounds_xy(), "inner_height:", volume.inner_height()
    )
    print("anchor:", result.p_in_container, "clearance:", result.clearance)
    print("waypoints:", len(wps))

    zz = np.array([wp.pose.position.z for wp in wps], dtype=float)
    plot_depth_over_steps(zz, "Volume agitation depth over steps (3D)")


def main() -> None:
    """Run the mixmap and agitation demos."""
    demo_cylinder_mixmap()
    demo_body_box_to_agitation()
    import inspect

    print("bind_slice_anchors_along_x:", bind_slice_anchors_along_x)
    print("module:", bind_slice_anchors_along_x.__module__)
    print("file:", inspect.getsourcefile(bind_slice_anchors_along_x))
    print("sig:", inspect.signature(bind_slice_anchors_along_x))


if __name__ == "__main__":
    main()
