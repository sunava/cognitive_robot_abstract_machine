from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from suturo_resources.suturo_map import load_environment

from semantic_digital_twin.adapters.viz_marker import TrajectoryVizPublisher
from simulation_setup import setup_hsrb_in_environment

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Box, Scale

from pycram.demos.thesis.primitives.seperation_devision import (
    SeparationSpec,
    SepMode,
    compile_separation_contact,
)


def add_box(
    world, name: str, scale_xyz: tuple[float, float, float], tf_frame: str
) -> Body:
    body = Body(
        name=PrefixedName(name),
        collision=ShapeCollection([Box(scale=Scale(*scale_xyz))]),
    )
    if hasattr(body, "tf_frame"):
        body.tf_frame = tf_frame
    with world.modify_world():
        world.add_body(body)
    return body


def top_plane_anchor_from_box(box_body):
    shape = box_body.collision.shapes[0]
    hz = 0.5 * float(shape.scale.z)

    frame_id = getattr(box_body, "tf_frame", "map")
    p0 = np.array([0.0, 0.0, hz], dtype=float)
    n = np.array([0.0, 0.0, 1.0], dtype=float)
    t1 = np.array([1.0, 0.0, 0.0], dtype=float)
    t2 = np.array([0.0, 1.0, 0.0], dtype=float)
    return frame_id, p0, n, t1, t2


def main():
    result = setup_hsrb_in_environment(load_environment=load_environment, with_viz=True)
    world = result.world

    box = add_box(world, name="muh_box", scale_xyz=(0.1, 0.1, 0.1), tf_frame="map")
    frame_id, p0, n, t1, t2 = top_plane_anchor_from_box(box)

    spec = SeparationSpec(
        mode=SepMode.SAW,
        slice_thickness=0.01,
        z_clearance=0.02,
        z_cut=0.0,
        margin_xy=0.005,
        saw_amplitude=0.01,
        saw_frequency=8.0,
        saw_cycles=4,
    )

    poses = list(
        compile_separation_contact(
            frame_id=frame_id,
            p0=p0,
            n=n,
            spec=spec,
            t1=t1,
            t2=t2,
        )
    )

    rclpy.init()
    node = Node("demo_semworld_and_traj")
    traj_viz = TrajectoryVizPublisher(node=node)

    traj_viz.publish_line_strip(poses, ns="muh_box_cut", marker_id=0, frame_id=frame_id)

    rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
