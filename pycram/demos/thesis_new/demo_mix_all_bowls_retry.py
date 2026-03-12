import os
import time
import numpy as np
import rclpy

import pycram
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
    MixingActionDescription,
    MoveTorsoActionDescription,
    NavigateActionDescription,
    ParkArmsActionDescription,
)
from pycram.external_interfaces.sparql_queries.mixing import safe_get_mixing_knowledge
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time

from pycram.tf_transformations import quaternion_from_euler, quaternion_multiply
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Whisk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color

from demos.thesis_new.spawn_random_bowls import _parse_stl, setup_random_bowl_world
from demos.thesis_new.tool_mounts import get_tool_mount_pose_kwargs
from demos.thesis_new.world_setup import resolve_robot_name
from demos.thesis_new.utils.demo_utils import (
    attach_available_tools,
    collect_named_targets,
    commit_plan_to_db,
    get_park_arms_argument,
    highlight_current_target,
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

ACTIVE_BOWL_COLOR = Color(R=0.52, G=0.82, B=0.98)
FAILED_BOWL_COLOR = Color(R=0.95, G=0.20, B=0.20)
SUCCESS_BOWL_COLOR = Color(R=0.62, G=0.92, B=0.62)
DEFAULT_BOWL_COLOR = Color(R=0.78, G=0.80, B=0.86)
RECORDS_DIR = os.path.join(os.path.dirname(__file__), "records")
RESULTS_CSV_PATH = os.path.join(RECORDS_DIR, "mix_all_bowls_results.csv")
EXPERIMENT_CONDITION = "full_system"
BASELINE_NAME = "task_knowledge+htn+constraint_planning"
TASK_NAME = "bowl_mixing"
MIX_DURATION_S = 6.0
POINTER_STRIDE = 3
MIXING_QUERY_TASK = "Whisking"
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
        "knowledge_motion",
        "knowledge_mixing_tool",
        *BASE_RESULT_FIELDNAMES,
    ]


def _try_mix(context, bowl, arm, tool):
    pickup_loc = CostmapLocation(
        target=PoseStamped.from_spatial_type(bowl.global_pose),
        reachable_arm=arm,
        reachable_for=context.robot,
    )
    with simulated_robot_without_collision:
        SequentialPlan(
            context,
            ParkArmsActionDescription(get_park_arms_argument(context.world)),
            NavigateActionDescription(pickup_loc, True),
        ).perform()

    with simulated_robot_with_collision:
        current_plan = SequentialPlan(
            context,
            MixingActionDescription(
                container=bowl,
                arm=arm,
                tool=tool,
                pointer_stride=POINTER_STRIDE,
                mix_duration_s=MIX_DURATION_S,
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


def main_mixing(seed=None, robot_name=None):
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
    world, _, surface_plan = setup_random_bowl_world(
        seed=effective_seed, robot_name=robot_name
    )

    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_mix_all_bowls_retry",
    )

    resolved_robot_name = resolve_robot_name(robot_name)
    arm_tools = attach_available_tools(
        world,
        _parse_stl,
        mesh_parts=("pycram_object_gap_demo", "whisk.stl"),
        right_name="whisk_right",
        left_name="whisk_left",
        right_pose_kwargs=get_tool_mount_pose_kwargs("mix", resolved_robot_name, Arms.RIGHT),
        left_pose_kwargs=get_tool_mount_pose_kwargs("mix", resolved_robot_name, Arms.LEFT),
        tool_cls=Whisk,
    )
    bowls = collect_named_targets(world, "bowl_")

    context = Context.from_world(world)
    context.ros_node = node
    robot_name = _robot_name(context.robot)
    world_name = _body_name(world.root)
    run_id = new_run_id()
    mixing_knowledge = safe_get_mixing_knowledge(MIXING_QUERY_TASK)

    print("[setup] surface plan:")
    print(f"[setup] seed: {effective_seed}")
    for surface_name, area_m2, target_count, placed_count in surface_plan:
        print(
            f"  - {surface_name}: area={area_m2:.3f}m^2 target={target_count} placed={placed_count}"
        )
    print(f"[setup] bowls to mix: {len(bowls)}")

    success_primary = 0
    success_fallback = 0
    failed = 0
    failed_bowls = set()
    successful_bowls = set()
    bowl_results = []
    initialize_csv(RESULTS_CSV_PATH, _results_csv_fieldnames())

    with simulated_robot_without_collision:
        SequentialPlan(
            context,
            ParkArmsActionDescription(get_park_arms_argument(world)),
            MoveTorsoActionDescription(TorsoState.HIGH),
        ).perform()

    for bowl in bowls:
        attempt_failures = []
        attempt_count = 0
        collision_failure_count = 0
        bowl_start_time = time.perf_counter()
        perturbation_applied = False
        perturbation_type = ""
        highlight_current_target(
            world,
            bowls,
            bowl,
            default_color=DEFAULT_BOWL_COLOR,
            active_color=ACTIVE_BOWL_COLOR,
            failed_color=FAILED_BOWL_COLOR,
            success_color=SUCCESS_BOWL_COLOR,
            failed_targets=failed_bowls,
            successful_targets=successful_bowls,
        )
        bowl_name = _body_name(bowl)
        common_result_kwargs = {
            "task_name": TASK_NAME,
            "run_id": run_id,
            "task_instance_id": bowl_name,
            "seed": effective_seed,
            "world_name": world_name,
            "experiment_condition": EXPERIMENT_CONDITION,
            "baseline_name": BASELINE_NAME,
            "knowledge_motion": mixing_knowledge.get("motion") or "",
            "knowledge_mixing_tool": mixing_knowledge.get("mixing_tool") or "",
            "knowledge_query_success": mixing_knowledge.get("query_success", False),
            "knowledge_query_error": mixing_knowledge.get("query_error", ""),
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

        for attempt_index, (arm, tool) in enumerate(arm_tools):
            phase = "primary" if attempt_index == 0 else "fallback"
            decision = "mix" if attempt_index == 0 else "retry_with_left_arm"
            decision_reason = "primary_success" if attempt_index == 0 else "right_arm_failed"
            print(f"[mix] {bowl_name}: try {arm.name} arm")
            try:
                attempt_count += 1
                _try_mix(context, bowl, arm, tool)
                if attempt_index == 0:
                    success_primary += 1
                else:
                    success_fallback += 1
                successful_bowls.add(bowl)
                result_row = _record_bowl_result(
                    bowl_results,
                    bowl_name,
                    robot_name,
                    "success",
                    arm.name,
                    _tool_name(tool),
                    phase,
                    attempt_failures,
                    **common_result_kwargs,
                    feasibility_reason="ok",
                    robot_decision=decision,
                    decision_reason=decision_reason,
                    assistance_requested=False,
                    assistance_completed=False,
                    task_blocked_by_prerequisite=False,
                    task_resumed_after_assistance=False,
                    final_success=True,
                    total_attempts=attempt_count,
                    retry_count=max(0, attempt_count - 1),
                    collision_failure_count=collision_failure_count,
                    recovery_used=attempt_index > 0,
                    recovery_success=attempt_index > 0,
                    perturbation_applied=perturbation_applied,
                    perturbation_type=perturbation_type,
                    execution_time_s=time.perf_counter() - bowl_start_time,
                )
                append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
                suffix = "" if attempt_index == 0 else " (fallback)"
                print(f"[ok] {bowl_name}: mixed with {arm.name} arm{suffix}")
                break
            except TimeoutError as exc:
                collision_failure_count += 1
                attempt_failures.append(
                    f"{arm.name} {phase} -> {_format_attempt_error(exc)}"
                )
                print(
                    f"[{'retry' if attempt_index < len(arm_tools) - 1 else 'fail'}] {bowl_name}: {arm.name} timed out "
                    f"({type(exc).__name__}: {exc})"
                )
            except Exception as exc:
                if _is_collision_like_failure(exc):
                    collision_failure_count += 1
                attempt_failures.append(
                    f"{arm.name} {phase} -> {_format_attempt_error(exc)}"
                )
                print(
                    f"[{'retry' if attempt_index < len(arm_tools) - 1 else 'fail'}] {bowl_name}: {arm.name} failed "
                    f"({type(exc).__name__}: {exc})"
                )
        else:
            failed += 1
            failed_bowls.add(bowl)
            last_tool = arm_tools[-1][1]
            result_row = _record_bowl_result(
                bowl_results,
                bowl_name,
                robot_name,
                "failed",
                "",
                _tool_name(last_tool),
                "fallback" if len(arm_tools) > 1 else "primary",
                attempt_failures,
                **common_result_kwargs,
                feasibility_reason="collision_or_motion_failure",
                robot_decision="task_failed",
                decision_reason="all_mix_attempts_failed",
                assistance_requested=False,
                assistance_completed=False,
                task_blocked_by_prerequisite=False,
                task_resumed_after_assistance=False,
                final_success=False,
                total_attempts=attempt_count,
                retry_count=max(0, attempt_count - 1),
                collision_failure_count=collision_failure_count,
                recovery_used=len(arm_tools) > 1,
                recovery_success=False,
                perturbation_applied=perturbation_applied,
                perturbation_type=perturbation_type,
                execution_time_s=time.perf_counter() - bowl_start_time,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            continue

    highlight_current_target(
        world,
        bowls,
        None,
        default_color=DEFAULT_BOWL_COLOR,
        active_color=ACTIVE_BOWL_COLOR,
        failed_color=FAILED_BOWL_COLOR,
        success_color=SUCCESS_BOWL_COLOR,
        failed_targets=failed_bowls,
        successful_targets=successful_bowls,
    )

    print("[summary]")
    print(f"  total bowls: {len(bowls)}")
    print(f"  success primary (RIGHT): {success_primary}")
    print(f"  success fallback (LEFT): {success_fallback}")
    print(f"  failed both arms: {failed}")
    print(f"  results csv: {RESULTS_CSV_PATH}")

    shutdown_experiment_runtime(node)

    # if __name__ == "__main__":
    #     session = pycram_sessionmaker()()
    #     Base.metadata.create_all(session.bind)
    #     session.commit()
    # main_mixing()
