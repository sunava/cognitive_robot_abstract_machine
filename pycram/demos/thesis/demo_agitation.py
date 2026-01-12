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

    plt.figure()
    plt.imshow(
        grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="equal"
    )
    plt.title("Cylinder clearance at z_mix")
    plt.xlabel("x in container frame")
    plt.ylabel("y in container frame")
    plt.colorbar()
    plt.scatter([result.p_in_container[0]], [result.p_in_container[1]], marker="x")
    plt.show()


def plot_3d_waypoints(wps, anchor=None, title="Volume agitation (3D)"):
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


def demo_body_box_to_agitation() -> None:
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

    xy = np.array([[wp.pose.position.x, wp.pose.position.y] for wp in wps], dtype=float)
    zz = np.array([wp.pose.position.z for wp in wps], dtype=float)

    plt.figure()
    plt.plot(xy[:, 0], xy[:, 1])
    plt.scatter([result.p_in_container[0]], [result.p_in_container[1]], marker="x")
    plt.title("Volume agitation path in object frame (x,y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

    plt.figure()
    plt.plot(zz)
    plt.title("Volume agitation depth over steps (z)")
    plt.xlabel("step")
    plt.ylabel("z")
    plt.show()


def main() -> None:
    demo_cylinder_mixmap()
    demo_body_box_to_agitation()
    import inspect

    print("bind_slice_anchors_along_x:", bind_slice_anchors_along_x)
    print("module:", bind_slice_anchors_along_x.__module__)
    print("file:", inspect.getsourcefile(bind_slice_anchors_along_x))
    print("sig:", inspect.signature(bind_slice_anchors_along_x))


if __name__ == "__main__":
    main()
