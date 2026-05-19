import csv
import math
import os
import uuid

from pycram.plans.failures import NavigationGoalNotReachedError


BASE_RESULT_FIELDNAMES = [
    "task_name",
    "run_id",
    "task_instance_id",
    "seed",
    "world_name",
    "experiment_condition",
    "baseline_name",
    "knowledge_query_success",
    "knowledge_query_error",
    "knowledge_prior_task",
    "knowledge_cutting_tool",
    "knowledge_cutting_position",
    "knowledge_repetition",
    "required_prerequisite",
    "prerequisite_source",
    "prerequisite_satisfied_initially",
    "autonomous_execution_feasible",
    "feasibility_reason",
    "robot_decision",
    "decision_reason",
    "assistance_requested",
    "assistance_type",
    "assistance_completed",
    "task_blocked_by_prerequisite",
    "task_resumed_after_assistance",
    "final_success",
    "robot_name",
    "robot_base_before_action_x",
    "robot_base_before_action_y",
    "robot_base_before_action_z",
    "robot_base_before_action_quat_x",
    "robot_base_before_action_quat_y",
    "robot_base_before_action_quat_z",
    "robot_base_before_action_quat_w",
    "robot_base_before_action_roll_rad",
    "robot_base_before_action_pitch_rad",
    "robot_base_before_action_yaw_rad",
    "robot_base_before_action_reference_frame",
    "outcome",
    "succeeded_arm",
    "tool_name",
    "phase",
    "total_attempts",
    "retry_count",
    "collision_failure_count",
    "recovery_used",
    "recovery_success",
    "perturbation_applied",
    "perturbation_type",
    "object_aabb_min_x",
    "object_aabb_min_y",
    "object_aabb_min_z",
    "object_aabb_max_x",
    "object_aabb_max_y",
    "object_aabb_max_z",
    "object_size_x",
    "object_size_y",
    "object_size_z",
    "object_volume_aabb",
    "target_world_x",
    "target_world_y",
    "target_world_z",
    "support_surface_name",
    "support_world_x",
    "support_world_y",
    "support_world_z",
    "support_yaw_rad",
    "support_size_x",
    "support_size_y",
    "support_size_z",
    "object_world_x",
    "object_world_y",
    "object_world_z",
    "object_quat_x",
    "object_quat_y",
    "object_quat_z",
    "object_quat_w",
    "object_roll_rad",
    "object_pitch_rad",
    "object_yaw_rad",
    "anchor_local_x",
    "anchor_local_y",
    "anchor_local_z",
    "anchor_norm_x",
    "anchor_norm_y",
    "anchor_norm_z",
    "cut_normal_local_x",
    "cut_normal_local_y",
    "cut_normal_local_z",
    "cut_normal_world_x",
    "cut_normal_world_y",
    "cut_normal_world_z",
    "cut_normal_world_yaw_rad",
    "technique_name",
    "slice_thickness_m",
    "num_cuts_x",
    "pointer_stride",
    "execution_time_s",
    "failure_count",
    "failure_reasons",
]


def body_name(body):
    maybe_name = getattr(body, "name", None)
    if hasattr(maybe_name, "name"):
        maybe_name = maybe_name.name
    return maybe_name if isinstance(maybe_name, str) else ""


def robot_name(robot):
    if robot is None:
        return ""
    return body_name(robot) or getattr(robot.__class__, "__name__", "")


def tool_name(tool):
    if tool is None:
        return ""
    root = getattr(tool, "root", None)
    if root is not None:
        return body_name(root)
    return getattr(tool.__class__, "__name__", "")


def format_attempt_error(exc):
    if isinstance(exc, TimeoutError):
        return f"CollisionError (reported as timeout): {exc}"
    return f"{type(exc).__name__}: {exc}"


def is_collision_like_failure(exc):
    return isinstance(exc, (TimeoutError, NavigationGoalNotReachedError))


def required_prerequisite_text(knowledge):
    return "|".join(knowledge.get("required_prerequisites", []))


def knowledge_source(knowledge):
    return "knowledge_base" if knowledge.get("query_success") else "none"


def assistance_type_from_knowledge(knowledge):
    return "|".join(knowledge.get("required_prerequisites", []))


def new_run_id():
    return str(uuid.uuid4())


def initialize_csv(csv_path, fieldnames):
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        ensure_csv_header(csv_path, fieldnames)
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()


def append_csv_row(csv_path, fieldnames, row):
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    file_exists = os.path.exists(csv_path)
    needs_header = (not file_exists) or os.path.getsize(csv_path) == 0
    if file_exists and not needs_header:
        ensure_csv_header(csv_path, fieldnames)
        fieldnames = _read_csv_fieldnames(csv_path) or fieldnames
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _read_csv_fieldnames(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        return csv.DictReader(csv_file).fieldnames


def ensure_csv_header(csv_path, fieldnames):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        existing_fieldnames = reader.fieldnames or []
        if all(field in existing_fieldnames for field in fieldnames):
            return
        rows = list(reader)

    merged_fieldnames = list(existing_fieldnames)
    for field in fieldnames:
        if field not in merged_fieldnames:
            merged_fieldnames.append(field)

    tmp_path = f"{csv_path}.tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=merged_fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp_path, csv_path)


def robot_base_pose_row(context):
    empty = {
        "robot_base_before_action_x": "",
        "robot_base_before_action_y": "",
        "robot_base_before_action_z": "",
        "robot_base_before_action_quat_x": "",
        "robot_base_before_action_quat_y": "",
        "robot_base_before_action_quat_z": "",
        "robot_base_before_action_quat_w": "",
        "robot_base_before_action_roll_rad": "",
        "robot_base_before_action_pitch_rad": "",
        "robot_base_before_action_yaw_rad": "",
        "robot_base_before_action_reference_frame": "",
    }
    try:
        robot = context.robot
        pose_owner = getattr(robot, "base", None) or getattr(robot, "root", None)
        pose = getattr(pose_owner, "global_pose", None)
        if pose is None:
            pose = getattr(robot.root, "global_pose")
        position = pose.to_position().to_np()
        quaternion = pose.to_quaternion().to_np()
        qx, qy, qz, qw = [float(value) for value in quaternion[:4]]
        roll, pitch, yaw = _quaternion_to_rpy(qx, qy, qz, qw)
        reference_frame = getattr(pose, "reference_frame", "")
        return {
            "robot_base_before_action_x": round(float(position[0]), 6),
            "robot_base_before_action_y": round(float(position[1]), 6),
            "robot_base_before_action_z": round(float(position[2]), 6),
            "robot_base_before_action_quat_x": round(qx, 6),
            "robot_base_before_action_quat_y": round(qy, 6),
            "robot_base_before_action_quat_z": round(qz, 6),
            "robot_base_before_action_quat_w": round(qw, 6),
            "robot_base_before_action_roll_rad": round(roll, 6),
            "robot_base_before_action_pitch_rad": round(pitch, 6),
            "robot_base_before_action_yaw_rad": round(yaw, 6),
            "robot_base_before_action_reference_frame": str(reference_frame),
        }
    except Exception:
        return empty


def _quaternion_to_rpy(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def build_base_result_row(
    results,
    robot_name,
    outcome,
    succeeded_arm,
    tool_name,
    phase,
    failures,
    **kwargs,
):
    row = {
        "robot_name": robot_name,
        "outcome": outcome,
        "succeeded_arm": succeeded_arm,
        "tool_name": tool_name,
        "phase": phase,
        "failure_count": len(failures),
        "failure_reasons": " | ".join(failures),
        **kwargs,
    }
    results.append(row)
    return row
