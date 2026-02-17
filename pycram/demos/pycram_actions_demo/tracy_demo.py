import logging
import os
from typing import Optional

import numpy as np

import rclpy

from demos.thesis.simulation_setup import add_box, BoxSpec
from demos.thesis_new.frame_provider import WorldTransformFrameProvider
from demos.thesis_new.motion_presets import build_default_sequence, build_container_sequence
from demos.thesis_new.motion_models import Pose, FixedFrameProvider
from demos.thesis_new.rviz import MotionSequenceRviz
from demos.thesis_new.tool_motion import (
    get_tool_config,
    make_tool_wrist_poses,
    tip_offset_from_body,
)
from demos.thesis_new.world_utils import try_get_body, make_identity_pose_stamped, body_local_aabb
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan, logger
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import MoveTorsoActionDescription, MoveTCPMotion, SimpleMoveTCPAction, MixingActionDescription, \
    WipingActionDescription, SimpleMoveTCPActionDescription, SimpleMoveTCPsActionDescription
from pycram.testing import setup_world, _build_package_resolver
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.abstract_robot import Arm
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection, OmniDrive
from semantic_digital_twin.world_description.geometry import Scale, Color


def tracy_world():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "urdf",
    )
    tracy = os.path.join(urdf_dir, "tracy.urdf")
    tracy_parser = URDFParser.from_file(file_path=tracy)
    world_with_tracy = tracy_parser.parse()
    Tracy.from_world(world_with_tracy)

    apartment_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "worlds",
            "apartment.urdf",
        )
    ).parse()

    with apartment_world.modify_world():
        apartment_world.merge_world_at_pose(
            world_with_tracy,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
            7.4, 3.2, 0, reference_frame=apartment_world.root
            ))
    return apartment_world


def main():
    """Run the RViz demo for default and bowl-constrained sequences."""
    world = tracy_world()

    rclpy.init()
    node = rclpy.create_node("pycram_demo")

    tf_wrapper = TFWrapper(node=node)
    TFPublisher(node=node, world=world)
    VizMarkerPublisher(world, node)
    #
    tf_wrapper.wait_for_transform(
        "apartment/apartment_root",
        "tracy/table",
        timeout=RclpyDuration(seconds=1.0),
        time=Time(),
    )

    spot_pose = PoseStamped()
    spot_pose.header.frame_id = world.root
    spot_pose.pose.position.x = 8.22
    spot_pose.pose.position.y = 3.29
    spot_pose.pose.position.z = 1.33

    spot_T = spot_pose.to_spatial_type()

    offset_T = HomogeneousTransformationMatrix.from_xyz_axis_angle(
        z=-0.03,
        axis=(0, 1, 0),
        angle=np.pi / 2,
        reference_frame=world.root,
    )

    # apply offset in spot_pose frame
    target_T = spot_T @ offset_T
    target_pose = PoseStamped.from_spatial_type(target_T)

    def make_sine_scan_poses(
            anchor: PoseStamped,
            lanes: int = 12,
            lane_spacing: float = 0.06,  # distance between vertical strokes
            y_span: float = 0.35,  # stroke length
            amplitude: float = 0.01,  # sine wiggle left/right
            wiggles: float = 1.0,  # sine cycles per stroke
            points_per_lane: int = 60,
            lane_axis: str = "x",  # "x" or "z"
    ) -> list[PoseStamped]:
        x0 = anchor.pose.position.x
        y0 = anchor.pose.position.y
        z0 = anchor.pose.position.z
        q = anchor.pose.orientation  # keep same orientation

        y_min = y0 - 0.5 * y_span
        y_max = y0 + 0.5 * y_span
        poses = []

        if lane_axis not in ("x", "z"):
            raise ValueError(f"lane_axis must be 'x' or 'z', got: {lane_axis}")

        for i in range(lanes):
            yc = np.linspace(y_min, y_max, points_per_lane)
            if i % 2 == 1:
                yc = yc[::-1]  # serpentine: up, then down, then up...

            phase = 2.0 * np.pi * wiggles * (yc - y_min) / max(y_span, 1e-9)
            wiggle = amplitude * np.sin(phase)

            if lane_axis == "x":
                lane_center = x0 + i * lane_spacing
                xc = lane_center + wiggle
                zc = np.full_like(yc, z0, dtype=float)
            else:
                lane_center = z0 + i * lane_spacing
                zc = lane_center + wiggle
                xc = np.full_like(yc, x0, dtype=float)

            for x, y, z in zip(xc, yc, zc):
                poses.append(
                    PoseStamped.from_list(
                        position=[float(x), float(y), float(z)],
                        orientation=[q.x, q.y, q.z, q.w],
                        frame=anchor.frame_id,
                    )
                )
        return poses

    sin_list_poses = make_sine_scan_poses(target_pose, lane_axis="z")
    context = Context.from_world(world)
    # poses = make_tool_wrist_poses(P_container, world, tip_offset, tool_cfg)
    plan = SequentialPlan(
        context,
        SimpleMoveTCPActionDescription(target_locations=sin_list_poses, arm=Arms.RIGHT)
       # MoveTorsoActionDescription(TorsoState.HIGH)
    )
    with simulated_robot:
        plan.perform()


if __name__ == "__main__":
    main()
