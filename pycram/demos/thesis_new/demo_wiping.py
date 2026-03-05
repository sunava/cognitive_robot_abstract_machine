import os
import numpy as np
import csv
import json

import rclpy
from trimesh.proximity import nearby_faces, closest_point

from demos.thesis.simulation_setup import add_box, BoxSpec
from demos.thesis_new.thesis_math.world_utils import try_get_body
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import MoveTorsoActionDescription, MixingActionDescription, \
    ParkArmsActionDescription, NavigateActionDescription, CuttingActionDescription, WipingActionDescription

from pycram.testing import setup_world
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import  Sponge
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import Body

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)



def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def _set_uniform_scale(body, scale_xyz, color=None):
    scale = Scale(*scale_xyz)
    for shape in body.root.visual.shapes:
        shape.scale = scale
        if color is not None:
            shape.color = color
    for shape in body.root.collision.shapes:
        shape.scale = scale


def setup_complex_world():
    world = setup_world()

    bowl = _parse_stl("objects", "bowl.stl")
    bowl.root.name.name = "bowl_small"
    _set_uniform_scale(bowl, (1.0, 1.0, 1.0), color=Color(R=1, G=1, B=0))

    bowl_middle = _parse_stl("objects", "bowl.stl")
    bowl_middle.root.name.name = "bowl_middle"
    _set_uniform_scale(bowl_middle, (1.3, 1.3, 1.3), color=Color(R=1, G=1, B=0))

    bowl_big = _parse_stl("objects", "bowl.stl")
    bowl_big.root.name.name = "bowl_big"
    _set_uniform_scale(bowl_big, (1.5, 1.5, 1.5), color=Color(R=1, G=1, B=0))

    bread = _parse_stl("pycram_object_gap_demo", "bread.stl")
    bread.root.name.name = "bread_small"
    _set_uniform_scale(bread, (1.0, 1.0, 1.0), color=Color(R=0.76, G=0.60, B=0.42))

    bread_middle = _parse_stl("pycram_object_gap_demo", "bread.stl")
    bread_middle.root.name.name = "bread_middle"
    _set_uniform_scale(
        bread_middle,
        (1.3, 1.3, 1.3),
        color=Color(R=0.76, G=0.60, B=0.42),
    )

    bread_big = _parse_stl("pycram_object_gap_demo", "bread.stl")
    bread_big.root.name.name = "bread_big"
    _set_uniform_scale(bread_big, (1.5, 1.5, 1.5), color=Color(R=0.76, G=0.60, B=0.42))



    sponge = add_box(
        world, BoxSpec(name="sponge", scale_xyz=(0.05, 0.05, 0.05)), tf_frame="/map", color=Color(R=1, G=1, B=0)
    )


    #
    # whisk = STLParser(
    #     os.path.join(
    #         os.path.dirname(__file__), "../..", "..", "resources", "pycram_object_gap_demo", "whisk.stl"
    #     )
    # ).parse()


    l_robot_tip = world.get_body_by_name("l_gripper_tool_frame")
    r_robot_tip = world.get_body_by_name("r_gripper_tool_frame")
    sponge = add_box(
        world, BoxSpec(name="sponge", scale_xyz=(0.05, 0.05, 0.05)), tf_frame="/map", color=Color(R=1, G=1, B=0)
    )

    with world.modify_world():
        # connection_knife = FixedConnection(
        #     parent=r_robot_tip, child=knife.root,
        #     parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.08, y=0, z=0, roll=0, pitch=0, yaw=0,
        #                                                  reference_frame=r_robot_tip)
        # )
        # connection_whisk = FixedConnection(
        #     parent=l_robot_tip, child=whisk.root,
        #     parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(x=0, y=0, z=0.08, roll=0, pitch=-np.pi/2, yaw=0,
        #                                                  reference_frame=l_robot_tip)
        # )
        # world.merge_world(sponge)
        world.add_kinematic_structure_entity(sponge)
        connection_sponge = FixedConnection(
            parent=l_robot_tip, child=sponge,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_axis_angle(axis=(0, 1, 0),
                                                                                               angle=np.pi / 2,
                                                                                               reference_frame=l_robot_tip
                                                                                               ))
        world.add_connection(connection_sponge)
        # world.merge_world(knife, connection_knife)
        # world.merge_world(whisk, connection_whisk)
        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                3.1, 2, 1, reference_frame=world.root
            ),
        )
        world.merge_world_at_pose(
            bowl_middle,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                3.1, 2.4, 1, reference_frame=world.root
            ),
        )
        world.merge_world_at_pose(
            bowl_big,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                3.1, 2.8, 1, reference_frame=world.root
            ),
        )

        world.merge_world_at_pose(
            bread,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.5, 2, 1, reference_frame=world.root
            ),
        )
        world.merge_world_at_pose(
            bread_middle,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.5, 2.4, 1, reference_frame=world.root
            ),
        )
        world.merge_world_at_pose(
            bread_big,
            HomogeneousTransformationMatrix.from_xyz_rpy(x=2.5, y=2.8, z=1, roll=0, pitch=0, yaw=-np.pi/2,
                                                         reference_frame=world.root),

        )
    return world


def _dump_experiment_metrics_csv(node, out_path: str) -> None:
    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return value

    records = getattr(node, "_experiment_metrics", [])
    if not records:
        return

    rows = []
    all_keys = set()
    for record in records:
        flat = {}
        for k, v in record.items():
            if isinstance(v, (dict, list, tuple)):
                flat[k] = json.dumps(_json_safe(v))
            else:
                flat[k] = _json_safe(v)
        rows.append(flat)
        all_keys.update(flat.keys())

    preferred_order = [
        "action",
        "action_success",
        "container",
        "container_pose",
        "tool",
        "robot_pose",
    ]
    fieldnames = [k for k in preferred_order if k in all_keys] + sorted(
        k for k in all_keys if k not in preferred_order
    )
    write_header = (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _record_failed_action(node, action, exc: Exception, step_index: int = None) -> None:
    records = getattr(node, "_experiment_metrics", None)
    if records is None:
        node._experiment_metrics = []
        records = node._experiment_metrics

    action_name = action.__class__.__name__
    if hasattr(action, "performable") and action.performable is not None:
        action_name = action.performable.__name__

    container_obj = getattr(action, "container", None)
    if container_obj is None and hasattr(action, "kwargs"):
        container_obj = action.kwargs.get("container")

    container_name = None
    if container_obj is not None:
        try:
            container_name = str(container_obj.name)
        except Exception:
            container_name = str(container_obj)

    tool_obj = getattr(action, "tool", None)
    if tool_obj is None and hasattr(action, "kwargs"):
        tool_obj = action.kwargs.get("tool")

    tool_name = None
    if tool_obj is not None:
        try:
            tool_name = str(tool_obj.root.name)
        except Exception:
            tool_name = str(tool_obj)

    records.append(
        {
            "action": action_name,
            "action_success": False,
            "overall_success": False,
            "geometric_success": False,
            "geometric_failed_checks": ["execution_exception"],
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "container": container_name,
            "tool": tool_name,
            "failed_step_index": step_index,
        }
    )


def main():
    world = setup_complex_world()

    rclpy.init()
    node = rclpy.create_node("pycram_demo")

    tf_wrapper = TFWrapper(node=node)
    TFPublisher(node=node, world=world)
    VizMarkerPublisher(world, node)

    tf_wrapper.wait_for_transform(
        "apartment/apartment_root",
        "pr2/base_footprint",
        timeout=RclpyDuration(seconds=1.0),
        time=Time(),
    )

    PR2.from_world(world)
    context = Context.from_world(world)
    #
    #
    # knife_body = try_get_body(world, "big-knife.stl")
    # knife = Knife(root=knife_body)
    # whisk_body = try_get_body(world, "whisk.stl")
    # whisk = Whisk(root=whisk_body)
    sponge_body = try_get_body(world, "sponge")
    sponge = Sponge(root=sponge_body)
    bread_body = try_get_body(world, "bread_small")
    bread_middle_body = try_get_body(world, "bread_middle")
    bread_big_body = try_get_body(world, "bread_big")
    #
    bowl_small_body = try_get_body(world, "bowl_small")
    bowl_middle_body = try_get_body(world, "bowl_middle")
    bowl_big_body = try_get_body(world, "bowl_big")
    clean_up_pose = PoseStamped.from_list([2.5,4,0.95])
    context.ros_node = node
    print(PoseStamped.from_spatial_type(context.robot.root.global_pose))

    actions = [
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
        NavigateActionDescription(
            PoseStamped.from_list([2.0 ,2.5,0.0], [0, 0, 0, 1], world.root),
            True,
        ),
        # NavigateActionDescription(target_location=PoseStamped.from_list([1.5,2.5,0.0])),
        # CuttingActionDescription(
        #     container=bread_body,
        #     arm=Arms.RIGHT,
        #     tool=knife,
        #     technique="saw",
        #     clear_viz=True,
        #     pointer_stride=10,
        #     num_cuts_x=5,
        # ),
        # CuttingActionDescription(
        #     container=bread_middle_body,
        #     arm=Arms.RIGHT,
        #     tool=knife,
        #     technique="saw",
        #     pointer_stride=10,
        #     num_cuts_x=5,
        # ),
        # CuttingActionDescription(
        #     container=bread_big_body,
        #     arm=Arms.RIGHT,
        #     tool=knife,
        #     technique="saw",
        #     pointer_stride=10,
        #     num_cuts_x=5,
        # ),
        # NavigateActionDescription(
        #     PoseStamped.from_spatial_type(
        #         HomogeneousTransformationMatrix.from_xyz_rpy(
        #             4.0, 2.5, 0.0, roll=0, pitch=0, yaw=180, reference_frame=world.root
        #         )
        #     ),
        #     True,
        # ),
        # MixingActionDescription(
        #     container=bowl_small_body,
        #     arm=Arms.LEFT,
        #     tool=whisk,
        #     pointer_stride=3,
        #     mix_duration_s=12.0,
        # ),
        # MixingActionDescription(
        #     container=bowl_middle_body,
        #     arm=Arms.LEFT,
        #     tool=whisk,
        #     pointer_stride=3,
        #     mix_duration_s=12.0,
        # ),
        # MixingActionDescription(
        #     container=bowl_big_body,
        #     arm=Arms.LEFT,
        #     tool=whisk,
        #     pointer_stride=3,
        #     mix_duration_s=12.0,
        # ),
        #
        WipingActionDescription(
            target_pose=clean_up_pose,
            arm=Arms.LEFT,
            tool=None,
        )
        # SimpleMoveTCPAction(target_location=poses[0], arm=Arms.RIGHT),
    ]

    failed_actions = []
    with simulated_robot:
        for idx, action in enumerate(actions, start=1):
            action_name = action.__class__.__name__
            if hasattr(action, "performable") and action.performable is not None:
                action_name = action.performable.__name__
            try:
                SequentialPlan(context, action).perform()
            except Exception as exc:
                _record_failed_action(node, action, exc, step_index=idx)
                failed_actions.append(
                    {
                        "index": idx,
                        "action": action_name,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
                print(
                    f"[FAIL] Step {idx} ({action_name}) failed with "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

    if failed_actions:
        print("\nFailed actions summary:")
        for fail in failed_actions:
            print(
                f"  - #{fail['index']} {fail['action']}: "
                f"{fail['exception_type']} - {fail['exception_message']}"
            )

    _dump_experiment_metrics_csv(
        node,
        os.path.join(os.path.dirname(__file__), "experiment_metrics.csv"),
    )

if __name__ == "__main__":
    main()
