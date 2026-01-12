from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from scipy.spatial.transform import Rotation
from std_msgs.msg import ColorRGBA
from trimesh import collision
from trimesh.permutate import transform
from visualization_msgs.msg import Marker, MarkerArray

from suturo_resources.suturo_map import load_environment

from giskardpy.motion_statechart.context import ExecutionContext
from pycram.src.pycram.datastructures.pose import PoseStamped, Point
from pycram.demos.thesis.simulation_setup import (
    setup_hsrb_in_environment,
    add_box,
    BoxSpec,
)
from pycram.demos.thesis.primitives.seperation_devision import (
    SeparationSpec,
    SepMode,
    compile_separation_contact,
)

from pycram.src.pycram.datastructures.enums import Arms, TorsoState
from pycram.src.pycram.language import SequentialPlan
from pycram.src.pycram.process_module import simulated_robot
from pycram.src.pycram.robot_plans import (
    ParkArmsActionDescription,
    MoveTorsoActionDescription,
)
from semantic_digital_twin.adapters.viz_marker import (
    TrajLinePublisher,
    VizMarkerPublisher,
)

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(frozen=True)
class CutPlaneAnchor:
    frame_id: Body
    p0: np.ndarray
    n: np.ndarray
    t1: np.ndarray
    t2: np.ndarray


def top_plane_anchor_from_box(box_body) -> CutPlaneAnchor:
    shape = box_body.collision.shapes[0]
    hz = 0.5 * float(shape.scale.z)
    return CutPlaneAnchor(
        frame_id=box_body,
        p0=np.array([0.0, 0.0, hz], dtype=float),
        n=np.array([0.0, 0.0, 1.0], dtype=float),
        t1=np.array([1.0, 0.0, 0.0], dtype=float),
        t2=np.array([0.0, 1.0, 0.0], dtype=float),
    )


def compile_cut(anchor: CutPlaneAnchor) -> list[PoseStamped]:
    spec = SeparationSpec(
        mode=SepMode.SAW,
        length=0.10,
        depth=0.06,
        n=120,
        tangential_osc_amp=0.01,
        tangential_osc_hz=6.0,
    )

    poses = compile_separation_contact(
        frame_id=anchor.frame_id,
        p0=anchor.p0,
        n=anchor.n,
        spec=spec,
        t1=anchor.t1,
        t2=anchor.t2,
    )

    if isinstance(poses, tuple):
        left, right = poses
        return list(left) + list(right)
    return list(poses)


# def transform_to_pose(transform: np.ndarray) -> Pose:
#     """
#     Converts a 4x4 transformation matrix to a PoseStamped message.
#
#     :param transform: The transformation matrix to convert.
#     :return: A PoseStamped message.
#     """
#     pose = Pose()
#     pose.position = Point(**dict(zip(["x", "y", "z"], transform[:3, 3])))
#     pose.orientation = Quaternion(
#         **dict(
#             zip(
#                 ["x", "y", "z", "w"],
#                 Rotation.from_matrix(transform[:3, :3]).as_quat(),
#             )
#         )
#     )
#     return pose


def main():
    result = setup_hsrb_in_environment(load_environment=load_environment, with_viz=True)
    world: World
    context: ExecutionContext
    viz: VizMarkerPublisher

    world, context, viz = result.world, result.context, result.viz

    if viz is None:
        raise RuntimeError("viz is None although with_viz=True")

    with world.modify_world():
        box = add_box(
            world, BoxSpec(name="muh_box", scale_xyz=(0.3, 0.1, 0.1)), tf_frame="/map"
        )
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.9, y=1.0, z=0.95
                ),
            )
        )
    anchor = top_plane_anchor_from_box(box)
    poses = compile_cut(anchor)

    world_poses: list[PoseStamped] = []
    for p in poses:

        new_pose = PoseStamped.from_spatial_type(
            world.transform(p.to_spatial_type(), world.root)
        )
        # print(new_pose.pose.position.from_numpy_vector())
        # print
        #
        world_poses.append(new_pose)
        #
        # print(base)

    # print(world_poses)
    # print(poses[0].from_spatial_type(world.get_body_by_name("muh_box")))
    # plan = SequentialPlan()
    # ros_poses = [p.ros_message() for p in poses]
    # print(ros_poses[0].tra)
    # world.transform(poses[0], "map")
    # print(poses[0].header.frame_id.)
    traj_pub = TrajLinePublisher(world=world, node=viz.node)

    traj_pub.publish(world_poses, ns="muh_box_cut", marker_id=0)

    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()
        # while True:
        #     traj_pub.publish(poses, ns="muh_box_cut", marker_id=0)


if __name__ == "__main__":
    main()
