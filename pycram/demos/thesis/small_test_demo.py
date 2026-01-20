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
from pycram.demos.thesis.primitives.surface_interaction import (
    SurfacePlane,
    bind_surface_anchor,
    ScrubSpec,
    SweepSpec,
    compile_scrub_circle,
    compile_wipe_raster_scrub,
)
from pycram.demos.thesis.primitives.volume_agitation import (
    VolumeAnchor,
    AgitationSpec,
    compile_volume_agitation,
)
from pycram.demos.thesis.geometry.volume_models import volume_from_body_collision

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
    depth: float


def top_plane_anchor_from_box(box_body) -> CutPlaneAnchor:
    shape = box_body.collision.shapes[0]
    hz = 0.5 * float(shape.scale.z)
    depth = float(shape.scale.z)
    return CutPlaneAnchor(
        frame_id=box_body,
        p0=np.array([0.0, 0.0, hz], dtype=float),
        n=np.array([0.0, 0.0, 1.0], dtype=float),
        t1=np.array([1.0, 0.0, 0.0], dtype=float),
        t2=np.array([0.0, 1.0, 0.0], dtype=float),
        depth=depth,
    )


def compile_cut(anchor: CutPlaneAnchor) -> list[PoseStamped]:
    spec = SeparationSpec(
        mode=SepMode.SAW,
        length=0.10,
        depth=anchor.depth,
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


def convert_ros_poses_to_pycram(
    poses: Iterable[PoseStamped], frame_id: Body
) -> list[PoseStamped]:
    converted: list[PoseStamped] = []
    for pose in poses:
        converted.append(
            PoseStamped.from_list(
                position=[
                    float(pose.pose.position.x),
                    float(pose.pose.position.y),
                    float(pose.pose.position.z),
                ],
                orientation=[
                    float(pose.pose.orientation.x),
                    float(pose.pose.orientation.y),
                    float(pose.pose.orientation.z),
                    float(pose.pose.orientation.w),
                ],
                frame=frame_id,
            )
        )
    return converted


def transform_poses_to_world(
    world: World, poses: Iterable[PoseStamped]
) -> list[PoseStamped]:
    world_poses: list[PoseStamped] = []
    for pose in poses:
        world_poses.append(
            PoseStamped.from_spatial_type(
                world.transform(pose.to_spatial_type(), world.root)
            )
        )
    return world_poses


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
    cut_poses = compile_cut(anchor)
    world_cut_poses = transform_poses_to_world(world, cut_poses)

    shape = box.collision.shapes[0]
    half_extents = np.array(
        [0.5 * float(shape.scale.x), 0.5 * float(shape.scale.y)], dtype=float
    )
    surface = SurfacePlane(
        frame_id=box,
        origin=np.array([0.0, 0.0, 0.5 * float(shape.scale.z)], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        half_extents_uv=half_extents,
        t1=np.array([1.0, 0.0, 0.0], dtype=float),
        t2=np.array([0.0, 1.0, 0.0], dtype=float),
    )
    surface_anchor = bind_surface_anchor(surface, margin=0.005)
    if surface_anchor is None:
        raise RuntimeError("surface anchor is None")

    scrub_spec = ScrubSpec(radius=0.03, points_per_cycle=48, cycles=2)
    scrub_poses = convert_ros_poses_to_pycram(
        compile_scrub_circle(surface_anchor, surface, scrub_spec, margin=0.005), box
    )
    world_scrub_poses = transform_poses_to_world(world, scrub_poses)

    sweep_spec = SweepSpec(spacing=0.03, margin=0.01)
    wipe_poses = convert_ros_poses_to_pycram(
        compile_wipe_raster_scrub(surface, sweep_spec, scrub_spec), box
    )
    world_wipe_poses = transform_poses_to_world(world, wipe_poses)

    volume = volume_from_body_collision(box, padding=0.005)
    if volume is None:
        raise RuntimeError("volume model is None")
    agitation_anchor = VolumeAnchor(
        frame_id=box, p=np.array([0.0, 0.0, 0.0], dtype=float)
    )
    agitation_spec = AgitationSpec(
        turns=2, angle_step_deg=20.0, radius_step=0.002, z_step=0.0005
    )
    agitation_poses = convert_ros_poses_to_pycram(
        compile_volume_agitation(
            agitation_anchor, agitation_spec, volume=volume, epsilon=0.01
        ),
        box,
    )
    world_agitation_poses = transform_poses_to_world(world, agitation_poses)

    # print(world_poses)
    # print(poses[0].from_spatial_type(world.get_body_by_name("muh_box")))
    # plan = SequentialPlan()
    # ros_poses = [p.ros_message() for p in poses]
    # print(ros_poses[0].tra)
    # world.transform(poses[0], "map")
    # print(poses[0].header.frame_id.)
    traj_pub = TrajLinePublisher(world=world, node=viz.node)

    # traj_pub.publish(world_cut_poses, ns="muh_box_cut", marker_id=0)
    # traj_pub.publish(world_scrub_poses, ns="muh_box_scrub", marker_id=1)
    traj_pub.publish(world_wipe_poses, ns="muh_box_wipe", marker_id=2)
    # traj_pub.publish(world_agitation_poses, ns="muh_box_agitation", marker_id=3)

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
