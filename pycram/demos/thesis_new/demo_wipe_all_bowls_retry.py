import os
import time
import numpy as np
from geometry_msgs.msg import Pose as RosPose
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from demos.thesis.simulation_setup import BoxSpec, add_box
from demos.thesis_new.spawn_random_bowls import sample_random_bowl_poses
from demos.thesis_new.utils.demo_utils import (
    commit_plan_to_db,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
)
from demos.thesis_new.utils.experiment_logging import (
    BASE_RESULT_FIELDNAMES,
    append_csv_row,
    body_name as _body_name,
    build_base_result_row,
    format_attempt_error as _format_attempt_error,
    initialize_csv,
    is_collision_like_failure as _is_collision_like_failure,
    new_run_id,
    robot_name as _robot_name,
    tool_name as _tool_name,
)
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.designators.location_designator import CostmapLocation
from pycram.language import SequentialPlan
from pycram.motion_executor import (
    simulated_robot_without_collision,
    simulated_robot_with_collision,
)
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker
from pycram.robot_plans import (
    MoveTorsoActionDescription,
    NavigateActionDescription,
    ParkArmsActionDescription,
    WipingActionDescription,
)

from pycram.tf_transformations import quaternion_from_euler, quaternion_multiply
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Sponge
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color

ACTIVE_TARGET_COLOR = Color(R=0.52, G=0.82, B=0.98)
FAILED_TARGET_COLOR = Color(R=0.95, G=0.20, B=0.20)
SUCCESS_TARGET_COLOR = Color(R=0.62, G=0.92, B=0.62)
DEFAULT_TARGET_COLOR = Color(R=0.78, G=0.80, B=0.86)

RECORDS_DIR = os.path.join(os.path.dirname(__file__), "records")
RESULTS_CSV_PATH = os.path.join(RECORDS_DIR, "wipe_all_bowls_results.csv")
EXPERIMENT_CONDITION = "full_system"
BASELINE_NAME = "task_knowledge+htn+constraint_planning"
TASK_NAME = "bowl_wiping"
POINTER_STRIDE = 3
TARGET_MARKER_TOPIC = "/pycram/wipe_targets"
session = None


def _record_bowl_result(
    results,
    bowl_name,
    robot_name,
    outcome,
    succeeded_arm,
    tool_name,
    phase,
    failures,
    **kwargs,
):
    return build_base_result_row(
        results,
        robot_name,
        outcome,
        succeeded_arm,
        tool_name,
        phase,
        failures,
        bowl_name=bowl_name,
        **kwargs,
    )


def _results_csv_fieldnames():
    return [
        "bowl_name",
        "spawn_pose_xyz",
        *BASE_RESULT_FIELDNAMES,
    ]


def _create_target_pose_marker_publisher(node):
    qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    return node.create_publisher(MarkerArray, TARGET_MARKER_TOPIC, qos)


def _marker_color_for_target(
    bowl_name,
    active_bowl_name,
    failed_bowl_names,
    successful_bowl_names,
):
    if bowl_name == active_bowl_name:
        return ACTIVE_TARGET_COLOR
    if bowl_name in failed_bowl_names:
        return FAILED_TARGET_COLOR
    if bowl_name in successful_bowl_names:
        return SUCCESS_TARGET_COLOR
    return DEFAULT_TARGET_COLOR


def _publish_target_pose_markers(
    node,
    publisher,
    world,
    sampled_bowls,
    *,
    active_bowl_name=None,
    failed_bowl_names=None,
    successful_bowl_names=None,
):
    failed_bowl_names = failed_bowl_names or set()
    successful_bowl_names = successful_bowl_names or set()
    frame_id = str(world.root.name)
    marker_array = MarkerArray()

    clear = Marker()
    clear.action = Marker.DELETEALL
    marker_array.markers.append(clear)

    for idx, bowl_data in enumerate(sampled_bowls):
        world_pose = bowl_data["world_pose"]
        position = world_pose.to_position()
        orientation = world_pose.to_quaternion()
        color = _marker_color_for_target(
            bowl_data["bowl_name"],
            active_bowl_name,
            failed_bowl_names,
            successful_bowl_names,
        )
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.ns = "wipe_targets"
        marker.id = idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = RosPose()
        marker.pose.position.x = float(position.x)
        marker.pose.position.y = float(position.y)
        marker.pose.position.z = float(position.z)
        marker.pose.orientation.x = float(orientation.x)
        marker.pose.orientation.y = float(orientation.y)
        marker.pose.orientation.z = float(orientation.z)
        marker.pose.orientation.w = float(orientation.w)
        marker.scale.x = 0.10
        marker.scale.y = 0.10
        marker.scale.z = 0.005
        marker.color.r = color.R
        marker.color.g = color.G
        marker.color.b = color.B
        marker.color.a = 0.85
        marker.lifetime = RclpyDuration(seconds=0.0).to_msg()
        marker.frame_locked = False
        marker_array.markers.append(marker)

    publisher.publish(marker_array)
    print(
        f"[viz] published {len(sampled_bowls)} wipe target markers on {TARGET_MARKER_TOPIC}"
    )


def _attach_bimanual_sponges(world):
    left_sponge = add_box(
        world,
        BoxSpec(name="sponge_left", scale_xyz=(0.05, 0.05, 0.05)),
        tf_frame="/map",
        color=Color(R=1, G=1, B=0),
    )
    right_sponge = add_box(
        world,
        BoxSpec(name="sponge_right", scale_xyz=(0.05, 0.05, 0.05)),
        tf_frame="/map",
        color=Color(R=1, G=1, B=0),
    )

    l_robot_tip = world.get_body_by_name("l_gripper_tool_frame")
    r_robot_tip = world.get_body_by_name("r_gripper_tool_frame")

    with world.modify_world():
        world.add_kinematic_structure_entity(left_sponge)
        world.add_kinematic_structure_entity(right_sponge)
        world.add_connection(
            FixedConnection(
                parent=l_robot_tip,
                child=left_sponge,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_axis_angle(
                    axis=(0, 1, 0), angle=np.pi / 2, reference_frame=l_robot_tip
                ),
            )
        )
        world.add_connection(
            FixedConnection(
                parent=r_robot_tip,
                child=right_sponge,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_axis_angle(
                    axis=(0, 1, 0), angle=np.pi / 2, reference_frame=r_robot_tip
                ),
            )
        )

    return Sponge(root=right_sponge), Sponge(root=left_sponge)


def _try_wipe(context, target_pose, arm, tool):
    pickup_loc = CostmapLocation(
        target=target_pose,
        reachable_arm=arm,
        reachable_for=context.robot,
    )
    with simulated_robot_without_collision:
        SequentialPlan(
            context,
            ParkArmsActionDescription(Arms.BOTH),
            NavigateActionDescription(pickup_loc, True),
        ).perform()
    print(target_pose)
    with simulated_robot_with_collision:
        current_plan = SequentialPlan(
            context,
            WipingActionDescription(
                target_pose=target_pose,
                arm=arm,
                tool=tool,
                pointer_stride=POINTER_STRIDE,
            ),
        )
        current_plan.perform()

    commit_plan_to_db(session, current_plan)


def _rotate_bowl_180deg_z(world, bowl):
    pose = bowl.global_pose
    pos = np.asarray(pose.to_position().to_np(), dtype=float).reshape(-1)[:3]
    quat = np.asarray(pose.to_quaternion().to_np(), dtype=float).reshape(-1)[:4]
    rot_quat = quaternion_from_euler(0.0, 0.0, np.pi)
    new_quat = quaternion_multiply(rot_quat, quat)
    rotated_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=float(pos[0]),
        pos_y=float(pos[1]),
        pos_z=float(pos[2]),
        quat_x=float(new_quat[0]),
        quat_y=float(new_quat[1]),
        quat_z=float(new_quat[2]),
        quat_w=float(new_quat[3]),
        reference_frame=world.root,
    )
    with world.modify_world():
        bowl.parent_connection.origin = rotated_pose


def main_wiping(seed=None):
    global session
    if session is None:
        session = pycram_sessionmaker()()
        Base.metadata.create_all(session.bind)
        session.commit()
    effective_seed = (
        int(seed)
        if seed is not None
        else int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    )
    world, sampled_bowls, surface_plan = sample_random_bowl_poses(seed=effective_seed)

    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_wipe_all_bowls_retry",
    )
    target_marker_pub = _create_target_pose_marker_publisher(node)

    right_sponge, left_sponge = _attach_bimanual_sponges(world)
    _publish_target_pose_markers(node, target_marker_pub, world, sampled_bowls)

    context = Context.from_world(world)
    context.ros_node = node
    robot_name = _robot_name(context.robot)
    world_name = _body_name(world.root)
    run_id = new_run_id()

    print("[setup] surface plan:")
    print(f"[setup] seed: {effective_seed}")
    for surface_name, area_m2, target_count, placed_count in surface_plan:
        print(
            f"  - {surface_name}: area={area_m2:.3f}m^2 target={target_count} placed={placed_count}"
        )
    print(f"[setup] sampled bowl poses to wipe: {len(sampled_bowls)}")

    success_primary = 0
    success_fallback = 0
    failed = 0
    failed_bowl_names = set()
    successful_bowl_names = set()
    bowl_results = []
    initialize_csv(RESULTS_CSV_PATH, _results_csv_fieldnames())

    with simulated_robot_without_collision:
        SequentialPlan(
            context,
            ParkArmsActionDescription(Arms.BOTH),
            MoveTorsoActionDescription(TorsoState.HIGH),
        ).perform()

    for bowl_data in sampled_bowls:
        attempt_failures = []
        attempt_count = 0
        collision_failure_count = 0
        bowl_start_time = time.perf_counter()
        perturbation_applied = False
        perturbation_type = ""
        bowl_name = bowl_data["bowl_name"]
        _publish_target_pose_markers(
            node,
            target_marker_pub,
            world,
            sampled_bowls,
            active_bowl_name=bowl_name,
            failed_bowl_names=failed_bowl_names,
            successful_bowl_names=successful_bowl_names,
        )
        target_pose = PoseStamped.from_spatial_type(bowl_data["world_pose"])
        spawn_xyz = np.round(np.asarray(bowl_data["pose_xyz"], dtype=float), 4).tolist()
        common_result_kwargs = {
            "task_name": TASK_NAME,
            "run_id": run_id,
            "task_instance_id": bowl_name,
            "seed": effective_seed,
            "world_name": world_name,
            "experiment_condition": EXPERIMENT_CONDITION,
            "baseline_name": BASELINE_NAME,
            "spawn_pose_xyz": spawn_xyz,
            "knowledge_query_success": True,
            "knowledge_query_error": "",
            "knowledge_prior_task": "",
            "knowledge_cutting_tool": "",
            "knowledge_cutting_position": "",
            "knowledge_repetition": "",
            "required_prerequisite": "",
            "prerequisite_source": "none",
            "prerequisite_satisfied_initially": True,
            "autonomous_execution_feasible": True,
            "assistance_type": "",
        }

        print(f"[wipe] {bowl_name}: try RIGHT arm at spawn pose {spawn_xyz}")
        try:
            attempt_count += 1
            _try_wipe(context, target_pose, Arms.RIGHT, right_sponge)
            success_primary += 1
            successful_bowl_names.add(bowl_name)
            _publish_target_pose_markers(
                node,
                target_marker_pub,
                world,
                sampled_bowls,
                failed_bowl_names=failed_bowl_names,
                successful_bowl_names=successful_bowl_names,
            )
            result_row = _record_bowl_result(
                bowl_results,
                bowl_name,
                robot_name,
                "success",
                "RIGHT",
                _tool_name(right_sponge),
                "primary",
                attempt_failures,
                **common_result_kwargs,
                feasibility_reason="ok",
                robot_decision="wipe",
                decision_reason="primary_success",
                assistance_requested=False,
                assistance_completed=False,
                task_blocked_by_prerequisite=False,
                task_resumed_after_assistance=False,
                final_success=True,
                total_attempts=attempt_count,
                retry_count=max(0, attempt_count - 1),
                collision_failure_count=collision_failure_count,
                recovery_used=False,
                recovery_success=False,
                perturbation_applied=perturbation_applied,
                perturbation_type=perturbation_type,
                execution_time_s=time.perf_counter() - bowl_start_time,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            print(f"[ok] {bowl_name}: wiped with RIGHT arm")
            continue
        except TimeoutError as exc_right_timeout:
            collision_failure_count += 1
            attempt_failures.append(
                f"RIGHT primary -> {_format_attempt_error(exc_right_timeout)}"
            )
            print(
                f"[retry] {bowl_name}: RIGHT timed out "
                f"({type(exc_right_timeout).__name__}: {exc_right_timeout})"
            )
        except Exception as exc_right:
            if _is_collision_like_failure(exc_right):
                collision_failure_count += 1
            attempt_failures.append(
                f"RIGHT primary -> {_format_attempt_error(exc_right)}"
            )
            print(
                f"[retry] {bowl_name}: RIGHT failed "
                f"({type(exc_right).__name__}: {exc_right})"
            )

        print(f"[wipe] {bowl_name}: try LEFT arm")
        try:
            attempt_count += 1
            _try_wipe(context, target_pose, Arms.LEFT, left_sponge)
            success_fallback += 1
            successful_bowl_names.add(bowl_name)
            _publish_target_pose_markers(
                node,
                target_marker_pub,
                world,
                sampled_bowls,
                failed_bowl_names=failed_bowl_names,
                successful_bowl_names=successful_bowl_names,
            )
            result_row = _record_bowl_result(
                bowl_results,
                bowl_name,
                robot_name,
                "success",
                "LEFT",
                _tool_name(left_sponge),
                "fallback",
                attempt_failures,
                **common_result_kwargs,
                feasibility_reason="ok",
                robot_decision="retry_with_left_arm",
                decision_reason="right_arm_failed",
                assistance_requested=False,
                assistance_completed=False,
                task_blocked_by_prerequisite=False,
                task_resumed_after_assistance=False,
                final_success=True,
                total_attempts=attempt_count,
                retry_count=max(0, attempt_count - 1),
                collision_failure_count=collision_failure_count,
                recovery_used=True,
                recovery_success=True,
                perturbation_applied=perturbation_applied,
                perturbation_type=perturbation_type,
                execution_time_s=time.perf_counter() - bowl_start_time,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            print(f"[ok] {bowl_name}: wiped with LEFT arm (fallback)")
            continue
        except TimeoutError as exc_left_timeout:
            collision_failure_count += 1
            attempt_failures.append(
                f"LEFT fallback -> {_format_attempt_error(exc_left_timeout)}"
            )
            print(
                f"[fail] {bowl_name}: LEFT timed out "
                f"({type(exc_left_timeout).__name__}: {exc_left_timeout})"
            )
        except Exception as exc_left:
            if _is_collision_like_failure(exc_left):
                collision_failure_count += 1
            attempt_failures.append(
                f"LEFT fallback -> {_format_attempt_error(exc_left)}"
            )
            print(
                f"[fail] {bowl_name}: LEFT failed "
                f"({type(exc_left).__name__}: {exc_left})"
            )
        failed += 1
        failed_bowl_names.add(bowl_name)
        _publish_target_pose_markers(
            node,
            target_marker_pub,
            world,
            sampled_bowls,
            failed_bowl_names=failed_bowl_names,
            successful_bowl_names=successful_bowl_names,
        )
        result_row = _record_bowl_result(
            bowl_results,
            bowl_name,
            robot_name,
            "failed",
            "",
            _tool_name(left_sponge),
            "fallback",
            attempt_failures,
            **common_result_kwargs,
            feasibility_reason="collision_or_motion_failure",
            robot_decision="task_failed",
            decision_reason="all_wipe_attempts_failed",
            assistance_requested=False,
            assistance_completed=False,
            task_blocked_by_prerequisite=False,
            task_resumed_after_assistance=False,
            final_success=False,
            total_attempts=attempt_count,
            retry_count=max(0, attempt_count - 1),
            collision_failure_count=collision_failure_count,
            recovery_used=True,
            recovery_success=False,
            perturbation_applied=perturbation_applied,
            perturbation_type=perturbation_type,
            execution_time_s=time.perf_counter() - bowl_start_time,
        )
        append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)

    print("[summary]")
    print(f"  total sampled bowl poses: {len(sampled_bowls)}")
    print(f"  success primary (RIGHT): {success_primary}")
    print(f"  success fallback (LEFT): {success_fallback}")
    print(f"  failed both arms: {failed}")
    print(f"  results csv: {RESULTS_CSV_PATH}")

    shutdown_experiment_runtime(node)

def main_mixing(seed=None):
    return main_wiping(seed=seed)


if __name__ == "__main__":
    main_wiping()
