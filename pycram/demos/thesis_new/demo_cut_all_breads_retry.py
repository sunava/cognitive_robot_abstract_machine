import time


from pycram.designators.location_designator import CostmapLocation

from pycram.motion_executor import (
    simulated_robot_without_collision,
    simulated_robot_with_collision,
)

from pycram.external_interfaces.sparql_queries.cutting import safe_get_cutting_knowledge
from pycram.tf_transformations import quaternion_from_euler, quaternion_multiply


from demos.thesis_new.spawn_random_breads import setup_random_bread_world
from demos.thesis_new.utils.demo_utils import (
    attach_bimanual_tools,
    collect_named_targets,
    commit_plan_to_db,
    highlight_current_target,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
)
from demos.thesis_new.utils.experiment_logging import (
    BASE_RESULT_FIELDNAMES,
    append_csv_row,
    assistance_type_from_knowledge,
    body_name as _body_name,
    build_base_result_row,
    format_attempt_error as _format_attempt_error,
    initialize_csv,
    is_collision_like_failure as _is_collision_like_failure,
    knowledge_source,
    new_run_id,
    required_prerequisite_text,
    robot_name as _robot_name,
    tool_name as _tool_name,
)
import os
import numpy as np
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker
from pycram.robot_plans import (
    MoveTorsoActionDescription,
    MixingActionDescription,
    ParkArmsActionDescription,
    NavigateActionDescription,
    CuttingActionDescription,
    WipingActionDescription,
)

from semantic_digital_twin.adapters.mesh import STLParser

from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Knife, Whisk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color, Scale

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)
DEFAULT_BREAD_COLOR = Color(R=0.76, G=0.60, B=0.42)
ACTIVE_BREAD_COLOR = Color(R=0.52, G=0.82, B=0.98)
FAILED_BREAD_COLOR = Color(R=0.95, G=0.20, B=0.20)
SUCCESS_BREAD_COLOR = Color(R=0.62, G=0.92, B=0.62)
RECORDS_DIR = os.path.join(os.path.dirname(__file__), "records")
RESULTS_CSV_PATH = os.path.join(RECORDS_DIR, "cut_all_breads_results.csv")
EXPERIMENT_CONDITION = "full_system"
BASELINE_NAME = "base_system"
TASK_NAME = "bread_cutting"
CUTTING_QUERY_VERB = "cut:Slicing"
CUTTING_QUERY_FOODON = "FOODON_00003523"
session = None


def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def _record_bread_result(
    results,
    bread_name,
    robot_name,
    outcome,
    succeeded_arm,
    tool_name,
    phase,
    failures,
    *,
    task_instance_id,
    experiment_condition,
    baseline_name,
    task_name,
    seed,
    world_name,
    run_id,
    knowledge_query_success,
    knowledge_query_error,
    knowledge_prior_task,
    knowledge_cutting_tool,
    knowledge_cutting_position,
    knowledge_repetition,
    required_prerequisite,
    prerequisite_source,
    prerequisite_satisfied_initially,
    autonomous_execution_feasible,
    feasibility_reason,
    robot_decision,
    decision_reason,
    assistance_requested,
    assistance_type,
    assistance_completed,
    task_blocked_by_prerequisite,
    task_resumed_after_assistance,
    final_success,
    total_attempts,
    retry_count,
    collision_failure_count,
    recovery_used,
    recovery_success,
    perturbation_applied,
    perturbation_type,
    execution_time_s,
):
    return build_base_result_row(
        results,
        robot_name,
        outcome,
        succeeded_arm,
        tool_name,
        phase,
        failures,
        task_name=task_name,
        run_id=run_id,
        task_instance_id=task_instance_id,
        bread_name=bread_name,
        seed=seed if seed is not None else "",
        world_name=world_name,
        experiment_condition=experiment_condition,
        baseline_name=baseline_name,
        knowledge_query_success=knowledge_query_success,
        knowledge_query_error=knowledge_query_error,
        knowledge_prior_task=knowledge_prior_task,
        knowledge_cutting_tool=knowledge_cutting_tool,
        knowledge_cutting_position=knowledge_cutting_position,
        knowledge_repetition=knowledge_repetition,
        required_prerequisite=required_prerequisite,
        prerequisite_source=prerequisite_source,
        prerequisite_satisfied_initially=prerequisite_satisfied_initially,
        autonomous_execution_feasible=autonomous_execution_feasible,
        feasibility_reason=feasibility_reason,
        robot_decision=robot_decision,
        decision_reason=decision_reason,
        assistance_requested=assistance_requested,
        assistance_type=assistance_type,
        assistance_completed=assistance_completed,
        task_blocked_by_prerequisite=task_blocked_by_prerequisite,
        task_resumed_after_assistance=task_resumed_after_assistance,
        final_success=final_success,
        total_attempts=total_attempts,
        retry_count=retry_count,
        collision_failure_count=collision_failure_count,
        recovery_used=recovery_used,
        recovery_success=recovery_success,
        perturbation_applied=perturbation_applied,
        perturbation_type=perturbation_type,
        execution_time_s=round(execution_time_s, 4),
    )


def _results_csv_fieldnames():
    return ["bread_name", *BASE_RESULT_FIELDNAMES]


def _try_cut(context, bread, arm, tool):
    pickup_loc = CostmapLocation(
        target=PoseStamped.from_spatial_type(bread.global_pose),
        reachable_arm=arm,
        reachable_for=context.robot,
    )
    with simulated_robot_without_collision:
        SequentialPlan(
            context,
            ParkArmsActionDescription(Arms.BOTH),
            NavigateActionDescription(pickup_loc, True),
        ).perform()

    with simulated_robot_with_collision:
        current_plan = SequentialPlan(
            context,
            CuttingActionDescription(
                container=bread,
                arm=arm,
                tool=tool,
                technique="saw",
                pointer_stride=13,
                num_cuts_x=1,
            ),
        )
        current_plan.perform()

    commit_plan_to_db(session, current_plan)


def _rotate_bread_180deg_z(world, bread):
    print("rotating bread")
    pose = bread.global_pose
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
    print(f"rotated object")
    with world.modify_world():
        bread.parent_connection.origin = rotated_pose


def main_cutting(seed=None):
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
    world, _, surface_plan = setup_random_bread_world(seed=effective_seed)

    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_cut_all_breads_retry",
    )

    right_knife, left_knife = attach_bimanual_tools(
        world,
        _parse_stl,
        mesh_parts=("pycram_object_gap_demo", "big-knife.stl"),
        right_name="knife_right",
        left_name="knife_left",
        right_pose_kwargs={
            "x": 0.0,
            "y": 0.0,
            "z": 0.08,
            "roll": 0.0,
            "pitch": -np.pi / 2,
            "yaw": 0.0,
        },
        left_pose_kwargs={
            "x": 0.0,
            "y": 0.0,
            "z": -0.08,
            "roll": np.pi,
            "pitch": np.pi / 2,
            "yaw": 0.0,
        },
        tool_cls=Knife,
    )
    breads = collect_named_targets(world, "bread_")

    context = Context.from_world(world)
    context.ros_node = node
    robot_name = _robot_name(context.robot)
    world_name = _body_name(world.root)
    run_id = new_run_id()
    cutting_knowledge = safe_get_cutting_knowledge(
        CUTTING_QUERY_VERB, CUTTING_QUERY_FOODON
    )

    print("[setup] surface plan:")
    print(f"[setup] seed: {effective_seed}")
    for surface_name, area_m2, target_count, placed_count in surface_plan:
        print(
            f"  - {surface_name}: area={area_m2:.3f}m^2 target={target_count} placed={placed_count}"
        )
    print(f"[setup] breads to cut: {len(breads)}")

    success_primary = 0
    success_fallback = 0
    success_rotated_right = 0
    success_rotated_left = 0
    failed = 0
    failed_breads = set()
    successful_breads = set()
    bread_results = []
    initialize_csv(RESULTS_CSV_PATH, _results_csv_fieldnames())

    with simulated_robot_without_collision:
        SequentialPlan(
            context,
            ParkArmsActionDescription(Arms.BOTH),
            MoveTorsoActionDescription(TorsoState.HIGH),
        ).perform()
    for bread in breads:
        attempt_failures = []
        attempt_count = 0
        collision_failure_count = 0
        bread_start_time = time.perf_counter()
        perturbation_applied = False
        perturbation_type = ""
        highlight_current_target(
            world,
            breads,
            bread,
            default_color=DEFAULT_BREAD_COLOR,
            active_color=ACTIVE_BREAD_COLOR,
            failed_color=FAILED_BREAD_COLOR,
            success_color=SUCCESS_BREAD_COLOR,
            failed_targets=failed_breads,
            successful_targets=successful_breads,
        )
        bread_name = _body_name(bread)
        common_result_kwargs = {
            "task_instance_id": bread_name,
            "experiment_condition": EXPERIMENT_CONDITION,
            "baseline_name": BASELINE_NAME,
            "task_name": TASK_NAME,
            "seed": effective_seed,
            "world_name": world_name,
            "run_id": run_id,
            "knowledge_query_success": cutting_knowledge.get("query_success", False),
            "knowledge_query_error": cutting_knowledge.get("query_error", ""),
            "knowledge_prior_task": cutting_knowledge.get("prior_task") or "",
            "knowledge_cutting_tool": cutting_knowledge.get("cutting_tool") or "",
            "knowledge_cutting_position": cutting_knowledge.get("cutting_position")
            or "",
            "knowledge_repetition": cutting_knowledge.get("repetition") or "",
            "required_prerequisite": required_prerequisite_text(cutting_knowledge),
            "prerequisite_source": knowledge_source(cutting_knowledge),
            "prerequisite_satisfied_initially": not bool(
                cutting_knowledge.get("required_prerequisites")
            ),
            "autonomous_execution_feasible": not bool(
                cutting_knowledge.get("required_prerequisites")
            ),
            "assistance_type": assistance_type_from_knowledge(cutting_knowledge),
        }
        print(f"[cut] {bread_name}: try RIGHT arm")
        try:
            attempt_count += 1
            _try_cut(context, bread, Arms.RIGHT, right_knife)
            success_primary += 1
            successful_breads.add(bread)
            result_row = _record_bread_result(
                bread_results,
                bread_name,
                robot_name,
                "success",
                "RIGHT",
                _tool_name(right_knife),
                "primary",
                attempt_failures,
                **common_result_kwargs,
                feasibility_reason="ok",
                robot_decision="cut",
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
                execution_time_s=time.perf_counter() - bread_start_time,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            print(f"[ok] {bread_name}: cut with RIGHT arm")
            continue

        except TimeoutError as exc_right_timeout:
            collision_failure_count += 1
            attempt_failures.append(
                f"RIGHT primary -> {_format_attempt_error(exc_right_timeout)}"
            )
            print(
                f"[retry] {bread_name}: RIGHT timed out "
                f"({type(exc_right_timeout).__name__}: {exc_right_timeout})"
            )
        except Exception as exc_right:
            if _is_collision_like_failure(exc_right):
                collision_failure_count += 1
            attempt_failures.append(
                f"RIGHT primary -> {_format_attempt_error(exc_right)}"
            )
            print(
                f"[retry] {bread_name}: RIGHT failed "
                f"({type(exc_right).__name__}: {exc_right})"
            )

        print(f"[cut] {bread_name}: try LEFT arm")
        try:
            attempt_count += 1
            _try_cut(context, bread, Arms.LEFT, left_knife)
            success_fallback += 1
            successful_breads.add(bread)
            result_row = _record_bread_result(
                bread_results,
                bread_name,
                robot_name,
                "success",
                "LEFT",
                _tool_name(left_knife),
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
                execution_time_s=time.perf_counter() - bread_start_time,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            print(f"[ok] {bread_name}: cut with LEFT arm (fallback)")
            continue
        except TimeoutError as exc_left_timeout:
            collision_failure_count += 1
            attempt_failures.append(
                f"LEFT fallback -> {_format_attempt_error(exc_left_timeout)}"
            )
            print(
                f"[fail] {bread_name}: LEFT timed out "
                f"({type(exc_left_timeout).__name__}: {exc_left_timeout})"
            )
        except Exception as exc_left:
            if _is_collision_like_failure(exc_left):
                collision_failure_count += 1
            attempt_failures.append(
                f"LEFT fallback -> {_format_attempt_error(exc_left)}"
            )
            print(
                f"[fail] {bread_name}: LEFT failed "
                f"({type(exc_left).__name__}: {exc_left})"
            )

        print(f"[retry] {bread_name}: rotate 180deg around Z and try again")
        perturbation_applied = True
        perturbation_type = "z_rotation_180"
        _rotate_bread_180deg_z(world, bread)

        print(f"[cut] {bread_name}: try RIGHT arm after rotation")
        try:
            attempt_count += 1
            _try_cut(context, bread, Arms.RIGHT, right_knife)
            success_rotated_right += 1
            successful_breads.add(bread)
            result_row = _record_bread_result(
                bread_results,
                bread_name,
                robot_name,
                "success",
                "RIGHT",
                _tool_name(right_knife),
                "after_rotation",
                attempt_failures,
                **common_result_kwargs,
                feasibility_reason="ok",
                robot_decision="rotate_object_and_retry",
                decision_reason="both_arms_failed_before_rotation",
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
                execution_time_s=time.perf_counter() - bread_start_time,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            print(f"[ok] {bread_name}: cut with RIGHT arm after rotation")
            continue
        except TimeoutError as exc_right_rot_timeout:
            collision_failure_count += 1
            attempt_failures.append(
                f"RIGHT after_rotation -> {_format_attempt_error(exc_right_rot_timeout)}"
            )
            print(
                f"[retry] {bread_name}: RIGHT after rotation timed out "
                f"({type(exc_right_rot_timeout).__name__}: {exc_right_rot_timeout})"
            )
        except Exception as exc_right_rot:
            if _is_collision_like_failure(exc_right_rot):
                collision_failure_count += 1
            attempt_failures.append(
                f"RIGHT after_rotation -> {_format_attempt_error(exc_right_rot)}"
            )
            print(
                f"[retry] {bread_name}: RIGHT after rotation failed "
                f"({type(exc_right_rot).__name__}: {exc_right_rot})"
            )

        print(f"[cut] {bread_name}: try LEFT arm after rotation")
        try:
            attempt_count += 1
            _try_cut(context, bread, Arms.LEFT, left_knife)
            success_rotated_left += 1
            successful_breads.add(bread)
            result_row = _record_bread_result(
                bread_results,
                bread_name,
                robot_name,
                "success",
                "LEFT",
                _tool_name(left_knife),
                "after_rotation",
                attempt_failures,
                **common_result_kwargs,
                feasibility_reason="ok",
                robot_decision="rotate_object_and_retry",
                decision_reason="both_arms_failed_before_rotation",
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
                execution_time_s=time.perf_counter() - bread_start_time,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            print(f"[ok] {bread_name}: cut with LEFT arm after rotation")
        except TimeoutError as exc_left_rot_timeout:
            failed += 1
            failed_breads.add(bread)
            collision_failure_count += 1
            attempt_failures.append(
                f"LEFT after_rotation -> {_format_attempt_error(exc_left_rot_timeout)}"
            )
            result_row = _record_bread_result(
                bread_results,
                bread_name,
                robot_name,
                "failed",
                "",
                _tool_name(left_knife),
                "after_rotation",
                attempt_failures,
                **common_result_kwargs,
                feasibility_reason=(
                    "prerequisite_requires_human_assistance"
                    if common_result_kwargs["required_prerequisite"]
                    else "collision_or_motion_failure"
                ),
                robot_decision=(
                    "request_human_help"
                    if common_result_kwargs["required_prerequisite"]
                    else "task_failed"
                ),
                decision_reason=(
                    "knowledge_base_prerequisite_detected"
                    if common_result_kwargs["required_prerequisite"]
                    else "all_cut_attempts_failed"
                ),
                assistance_requested=bool(
                    common_result_kwargs["required_prerequisite"]
                ),
                assistance_completed=False,
                task_blocked_by_prerequisite=bool(
                    common_result_kwargs["required_prerequisite"]
                ),
                task_resumed_after_assistance=False,
                final_success=False,
                total_attempts=attempt_count,
                retry_count=max(0, attempt_count - 1),
                collision_failure_count=collision_failure_count,
                recovery_used=True,
                recovery_success=False,
                perturbation_applied=perturbation_applied,
                perturbation_type=perturbation_type,
                execution_time_s=time.perf_counter() - bread_start_time,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            print(
                f"[fail] {bread_name}: LEFT after rotation timed out "
                f"({type(exc_left_rot_timeout).__name__}: {exc_left_rot_timeout})"
            )
        except Exception as exc_left_rot:
            failed += 1
            failed_breads.add(bread)
            if _is_collision_like_failure(exc_left_rot):
                collision_failure_count += 1
            attempt_failures.append(
                f"LEFT after_rotation -> {_format_attempt_error(exc_left_rot)}"
            )
            result_row = _record_bread_result(
                bread_results,
                bread_name,
                robot_name,
                "failed",
                "",
                _tool_name(left_knife),
                "after_rotation",
                attempt_failures,
                **common_result_kwargs,
                feasibility_reason=(
                    "prerequisite_requires_human_assistance"
                    if common_result_kwargs["required_prerequisite"]
                    else "collision_or_motion_failure"
                ),
                robot_decision=(
                    "request_human_help"
                    if common_result_kwargs["required_prerequisite"]
                    else "task_failed"
                ),
                decision_reason=(
                    "knowledge_base_prerequisite_detected"
                    if common_result_kwargs["required_prerequisite"]
                    else "all_cut_attempts_failed"
                ),
                assistance_requested=bool(
                    common_result_kwargs["required_prerequisite"]
                ),
                assistance_completed=False,
                task_blocked_by_prerequisite=bool(
                    common_result_kwargs["required_prerequisite"]
                ),
                task_resumed_after_assistance=False,
                final_success=False,
                total_attempts=attempt_count,
                retry_count=max(0, attempt_count - 1),
                collision_failure_count=collision_failure_count,
                recovery_used=True,
                recovery_success=False,
                perturbation_applied=perturbation_applied,
                perturbation_type=perturbation_type,
                execution_time_s=time.perf_counter() - bread_start_time,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            print(
                f"[fail] {bread_name}: LEFT after rotation failed "
                f"({type(exc_left_rot).__name__}: {exc_left_rot})"
            )

    highlight_current_target(
        world,
        breads,
        None,
        default_color=DEFAULT_BREAD_COLOR,
        active_color=ACTIVE_BREAD_COLOR,
        failed_color=FAILED_BREAD_COLOR,
        success_color=SUCCESS_BREAD_COLOR,
        failed_targets=failed_breads,
        successful_targets=successful_breads,
    )

    print("[summary]")
    print(f"  total breads: {len(breads)}")
    print(f"  success primary (RIGHT): {success_primary}")
    print(f"  success fallback (LEFT): {success_fallback}")
    print(f"  success after rotation (RIGHT): {success_rotated_right}")
    print(f"  success after rotation (LEFT): {success_rotated_left}")
    print(f"  failed both arms: {failed}")
    print(f"  results csv: {RESULTS_CSV_PATH}")

    shutdown_experiment_runtime(node)


# if __name__ == "__main__":
#     session = pycram_sessionmaker()()
#     # drop_database(session.bind)
#     Base.metadata.create_all(session.bind)
#     session.commit()
#
#     main_cutting()
