import os
import time

import numpy as np

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.locations.locations import CostmapLocation
from pycram.motion_executor import (
    simulated_robot_with_collision,
    simulated_robot_without_collision,
)
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.tool_based import SimplePouringAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup
from semantic_digital_twin.world_description.geometry import Color

from .demo_mix_all_bowls_retry import (
    _build_mixing_geometry_binding,
    _empty_mixing_progress,
    _timed,
    _update_costmap_debug_publishers,
)
from .spawn_random_bowls import _parse_stl, setup_random_bowl_world
from .tool_mounts import get_tool_mount_pose_kwargs
from .utils.demo_utils import (
    attach_available_tools,
    collect_named_targets,
    commit_plan_to_db,
    get_park_arms_argument,
    highlight_current_target,
    resolve_navigation_target_for_environment,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
)
from .utils.experiment_logging import (
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
from .world_setup import resolve_robot_name

DEFAULT_BOWL_COLOR = Color(R=0.78, G=0.80, B=0.86)
ACTIVE_BOWL_COLOR = Color(R=0.52, G=0.82, B=0.98)
FAILED_BOWL_COLOR = Color(R=0.95, G=0.20, B=0.20)
SUCCESS_BOWL_COLOR = Color(R=0.62, G=0.92, B=0.62)
RECORDS_DIR = os.path.join(os.path.dirname(__file__), "../records")
RESULTS_CSV_PATH = os.environ.get(
    "THESIS_POUR_RESULTS_CSV_PATH",
    os.path.join(RECORDS_DIR, "pour_all_bowls_results.csv"),
)
EXPERIMENT_CONDITION = "full_system"
BASELINE_NAME = "simple_pouring_action"
TASK_NAME = "pour"
DEBUG_PROFILE_POURING = True
session = None


def _results_csv_fieldnames():
    return ["bowl_name", *BASE_RESULT_FIELDNAMES]


def _empty_pouring_progress():
    progress = _empty_mixing_progress()
    progress.pop("mixing_phase_reached", None)
    progress.pop("mixing_planned_duration_s", None)
    progress.pop("mixing_estimated_completed_time_s", None)
    return progress


def _record_pour_result(
    results,
    bowl_name,
    robot_name,
    outcome,
    succeeded_arm,
    tool_name,
    phase,
    failures,
    *,
    execution_time_s,
    geometry_binding,
    motion_progress=None,
    **kwargs,
):
    geometry_binding = dict(geometry_binding)
    geometry_binding["technique_name"] = "simple_pour"
    row = build_base_result_row(
        results,
        robot_name,
        outcome,
        succeeded_arm,
        tool_name,
        phase,
        failures,
        bowl_name=bowl_name,
        **kwargs,
        **geometry_binding,
        **(motion_progress or _empty_pouring_progress()),
        execution_time_s=round(execution_time_s, 4),
    )
    return row


def _try_pour(context, bowl, pickup_pose, arm, cup_tool):
    if cup_tool is None:
        raise RuntimeError(f"No mounted pouring cup available for {arm.name}.")

    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(get_park_arms_argument(context.world)),
                MoveTorsoAction(TorsoState.HIGH),
                NavigateAction(pickup_pose, True, teleport=True),
            ],
            context,
        ).perform()

    current_plan = None
    try:
        with simulated_robot_with_collision:
            # The cup is attached to the selected hand before Context creation.
            # SimplePouringAction uses that mounted tool frame for the Cartesian tilt.
            current_plan = sequential(
                [
                    SimplePouringAction(
                        object_designator=bowl,
                        arm=arm,
                    ),
                    ParkArmsAction(get_park_arms_argument(context.world)),
                ],
                context,
            )
            current_plan.perform()
    finally:
        if current_plan is not None:
            commit_plan_to_db(session, current_plan)


def main_pouring(seed=None, robot_name=None, environment_name=None):
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
        seed=effective_seed,
        robot_name=robot_name,
        environment_name=environment_name,
    )

    print("[setup] starting ROS runtime and RViz publishers", flush=True)
    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_pour_all_bowls_retry",
    )

    try:
        print("[setup] building execution context", flush=True)
        context = Context.from_world(world)
        context.ros_node = node
        robot_label = _robot_name(context.robot)
        world_name = environment_name
        run_id = new_run_id()

        with simulated_robot_without_collision:
            sequential(
                [
                    ParkArmsAction(get_park_arms_argument(context.world)),

                ],
                context,
            ).perform()
        resolved_robot_name = resolve_robot_name(robot_name)
        print("[setup] attaching pouring cups", flush=True)
        mounted_cups = attach_available_tools(
            world,
            _parse_stl,
            mesh_parts=("objects", "jeroen_cup.stl"),
            right_name="jeroen_cup_right",
            left_name="jeroen_cup_left",
            right_pose_kwargs=get_tool_mount_pose_kwargs(
                "pour", resolved_robot_name, Arms.RIGHT
            ),
            left_pose_kwargs=get_tool_mount_pose_kwargs(
                "pour", resolved_robot_name, Arms.LEFT
            ),
            tool_cls=Cup,
        )

        print("[setup] collecting pouring targets", flush=True)
        bowls = collect_named_targets(world, "bowl_")


        print("[setup] closing grippers", flush=True)
        with simulated_robot_without_collision:
            sequential(
                [SetGripperAction(Arms.BOTH, GripperState.CLOSE)],
                context,
            ).perform()

        print("[setup] surface plan:")
        print(f"[setup] seed: {effective_seed}")
        for surface_name, area_m2, target_count, placed_count in surface_plan:
            print(
                f"  - {surface_name}: area={area_m2:.3f}m^2 "
                f"target={target_count} placed={placed_count}"
            )
        print(f"[setup] bowls to pour into: {len(bowls)}")

        initialize_csv(RESULTS_CSV_PATH, _results_csv_fieldnames())
        debug_costmap_publishers = {}
        successful_bowls = set()
        failed_bowls = set()
        pour_results = []
        success_count = 0

        for bowl_index, bowl in enumerate(bowls, start=1):
            bowl_name = _body_name(bowl)
            print(
                f"[Demo time] {success_count}/{len(bowls)} poured bowls "
                f"(next {bowl_index}/{len(bowls)}: {bowl_name})"
            )
            bowl_start_time = time.perf_counter()
            attempt_failures = []
            attempt_count = 0
            collision_failure_count = 0
            last_progress = _empty_pouring_progress()

            debug_costmap_publishers, _ = _timed(
                f"pour/{bowl_name}/costmap_preview",
                lambda: _update_costmap_debug_publishers(
                    node, context.robot, world, bowl, debug_costmap_publishers
                ),
            )
            pickup_loc = CostmapLocation(
                target=bowl.global_pose,
                reachable=True,
                reachable_arm=mounted_cups[0][0] if mounted_cups else None,
                validate_reachability=False,
                samples=1000,
                context=context,
            )
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
            common_result_kwargs = {
                "task_instance_id": bowl_name,
                "experiment_condition": EXPERIMENT_CONDITION,
                "baseline_name": BASELINE_NAME,
                "task_name": TASK_NAME,
                "seed": effective_seed,
                "world_name": world_name,
                "run_id": run_id,
                "knowledge_query_success": False,
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

            try:
                pickup_pose = resolve_navigation_target_for_environment(
                    pickup_loc,
                    description=f"pouring into {bowl.name}",
                    environment_name=environment_name,
                )[0]
            except Exception as exc:
                failed_bowls.add(bowl)
                attempt_failures.append(f"navigation setup -> {_format_attempt_error(exc)}")
                fallback_tool = mounted_cups[-1][1] if mounted_cups else None
                result_row = _record_pour_result(
                    pour_results,
                    bowl_name,
                    robot_label,
                    "failed",
                    "",
                    _tool_name(fallback_tool),
                    "pickup_setup",
                    attempt_failures,
                    **common_result_kwargs,
                    feasibility_reason="navigation_target_resolution_failed",
                    robot_decision="skip_object",
                    decision_reason="pickup_pose_unavailable",
                    assistance_requested=False,
                    assistance_completed=False,
                    task_blocked_by_prerequisite=False,
                    task_resumed_after_assistance=False,
                    final_success=False,
                    total_attempts=attempt_count,
                    retry_count=0,
                    collision_failure_count=collision_failure_count,
                    recovery_used=False,
                    recovery_success=False,
                    perturbation_applied=False,
                    perturbation_type="",
                    execution_time_s=time.perf_counter() - bowl_start_time,
                    geometry_binding=_build_mixing_geometry_binding(
                        bowl, robot=context.robot
                    ),
                    motion_progress=last_progress,
                )
                append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
                continue

            attempt_succeeded = False
            for attempt_index, (arm, tool) in enumerate(mounted_cups):
                decision = "pour" if attempt_index == 0 else "retry_with_left_arm"
                decision_reason = (
                    "primary_success" if attempt_index == 0 else "right_arm_failed"
                )
                print(f"[pour] {bowl_name}: try {arm.name} arm")
                try:
                    attempt_count += 1
                    _try_pour(context, bowl, pickup_pose, arm, tool)
                    success_count += 1
                    successful_bowls.add(bowl)
                    result_row = _record_pour_result(
                        pour_results,
                        bowl_name,
                        robot_label,
                        "success",
                        arm.name,
                        _tool_name(tool),
                        "primary",
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
                        recovery_used=attempt_count > 1,
                        recovery_success=attempt_count > 1,
                        perturbation_applied=False,
                        perturbation_type="",
                        execution_time_s=time.perf_counter() - bowl_start_time,
                        geometry_binding=_build_mixing_geometry_binding(
                            bowl, robot=context.robot
                        ),
                        motion_progress={
                            **last_progress,
                            "motion_approach_completed": True,
                            "motion_progress_note": "completed",
                        },
                    )
                    append_csv_row(
                        RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row
                    )
                    print(f"[ok] {bowl_name}: poured with {arm.name} arm")
                    attempt_succeeded = True
                    break
                except Exception as exc:
                    if _is_collision_like_failure(exc):
                        collision_failure_count += 1
                    attempt_failures.append(
                        f"{arm.name} primary -> {_format_attempt_error(exc)}"
                    )
                    print(
                        f"[retry] {bowl_name}: {arm.name} failed "
                        f"({type(exc).__name__}: {exc})"
                    )

            if attempt_succeeded:
                continue

            failed_bowls.add(bowl)
            last_tool = mounted_cups[-1][1] if mounted_cups else None
            result_row = _record_pour_result(
                pour_results,
                bowl_name,
                robot_label,
                "failed",
                "",
                _tool_name(last_tool),
                "primary",
                attempt_failures,
                **common_result_kwargs,
                feasibility_reason="collision_or_motion_failure",
                robot_decision="task_failed",
                decision_reason="all_pour_attempts_failed",
                assistance_requested=False,
                assistance_completed=False,
                task_blocked_by_prerequisite=False,
                task_resumed_after_assistance=False,
                final_success=False,
                total_attempts=attempt_count,
                retry_count=max(0, attempt_count - 1),
                collision_failure_count=collision_failure_count,
                recovery_used=attempt_count > 1,
                recovery_success=False,
                perturbation_applied=False,
                perturbation_type="",
                execution_time_s=time.perf_counter() - bowl_start_time,
                geometry_binding=_build_mixing_geometry_binding(
                    bowl, robot=context.robot
                ),
                motion_progress=last_progress,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)

        print(f"[summary] total bowls: {len(bowls)}")
        print(f"[summary] success: {success_count}")
        print(f"[summary] failed: {len(failed_bowls)}")
    finally:
        shutdown_experiment_runtime(node)


if __name__ == "__main__":
    main_pouring()
