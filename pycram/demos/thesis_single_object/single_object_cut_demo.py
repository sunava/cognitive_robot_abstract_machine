import io
import logging
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

import pycram.motion_executor as motion_executor_module
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.motion_executor import (
    simulated_robot_without_collision,
    simulated_robot_with_collision,
)
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.tool_based import CuttingAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import SetGripperAction
from pycram.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Knife
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

try:
    from ..thesis_new.demo_cut_all_breads_retry import (
        CUTTING_NUM_CUTS_X,
        CUTTING_POINTER_STRIDE,
        CUTTING_SLICE_THICKNESS_M,
        CUTTING_TECHNIQUE,
        _body_name,
        _cut_object_execution_config,
        _parse_stl,
    )
    from ..thesis_new.spawn_random_breads import (
        _pose_xyz,
        _set_uniform_scale,
        get_cut_object_config,
    )
    from ..thesis_new.tool_mounts import get_tool_mount_pose_kwargs
    from ..thesis_new.world_setup import (
        resolve_environment_name,
        resolve_robot_name,
        setup_thesis_world,
    )
except ImportError:
    from thesis_new.demo_cut_all_breads_retry import (
        CUTTING_NUM_CUTS_X,
        CUTTING_POINTER_STRIDE,
        CUTTING_SLICE_THICKNESS_M,
        CUTTING_TECHNIQUE,
        _body_name,
        _cut_object_execution_config,
        _parse_stl,
    )
    from thesis_new.spawn_random_breads import (
        _pose_xyz,
        _set_uniform_scale,
        get_cut_object_config,
    )
    from thesis_new.tool_mounts import get_tool_mount_pose_kwargs
    from thesis_new.world_setup import (
        resolve_environment_name,
        resolve_robot_name,
        setup_thesis_world,
    )
from pycram.robot_plans.actions.composite.thesis_math import body_local_aabb
from pycram.robot_plans.actions.composite.utils.demo_utils import (
    attach_available_tools,
    collect_named_targets,
    get_park_arms_argument,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
)
from semantic_digital_twin.world_description.geometry import Color

HANDPICKED_PICKUP_POSES = {
    "apartment": (1.70, 2.74, 0.0, 0.0),
    "apartment_without_walls": (1.95, 2.15, 0.0, 0.0),
    "kitchen": (2.10, 2.20, 0.0, 0.0),
    "isr": (1.40, 1.10, 0.0, 0.0),
}
CUTTING_BOARD_COLOR = Color(R=0.80, G=0.66, B=0.49)


def _suppress_demo_noise():
    logging.getLogger("pycram.motion_executor").setLevel(logging.ERROR)
    logging.getLogger("pycram.robot_plans.actions.base").setLevel(logging.WARNING)
    motion_executor_module.DEBUG_PROFILE_MOTION_EXECUTOR = False


def _try_cut(
    context,
    target,
    pickup_pose,
    arm,
    tool,
    *,
    cutting_technique,
    num_cuts_x,
):
    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(
                    get_park_arms_argument(context.world),
                    joint_state=StaticJointState.PARKTOOL,
                ),
            ],
            context,
        ).perform()

    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(
                    get_park_arms_argument(context.world),
                    joint_state=StaticJointState.PARKTOOL,
                ),
            ],
            context,
        ).perform()
        sequential([MoveTorsoAction(TorsoState.HIGH)], context).perform()
        sequential(
            [NavigateAction(pickup_pose, True, teleport=True)], context
        ).perform()

    with simulated_robot_with_collision:
        sequential(
            [
                CuttingAction(
                    container=target,
                    arm=arm,
                    tool=tool,
                    technique=cutting_technique,
                    pointer_stride=CUTTING_POINTER_STRIDE,
                    num_cuts_x=num_cuts_x,
                    slice_thickness=CUTTING_SLICE_THICKNESS_M,
                ),
                ParkArmsAction(get_park_arms_argument(context.world)),
            ],
            context,
        ).perform()


def _spawn_single_object(
    *,
    world,
    object_kind,
    spawn_position,
    spawn_yaw,
    spawn_scale,
):
    object_cfg = get_cut_object_config(object_kind)
    object_name = f"{object_cfg['object_name_prefix']}_0001"
    x_world, y_world, z_world = [float(value) for value in spawn_position]
    spawned = _parse_stl(*object_cfg["mesh_parts"])
    spawned.root.name.name = object_name
    _set_uniform_scale(
        spawned,
        (float(spawn_scale), float(spawn_scale), float(spawn_scale)),
        color=object_cfg["object_color"],
    )
    world_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=x_world,
        y=y_world,
        z=z_world,
        yaw=float(spawn_yaw),
        reference_frame=world.root,
    )
    world.merge_world_at_pose(spawned, world_pose)
    world.update_forward_kinematics()
    return object_name, _pose_xyz(world_pose)


def _resolve_pickup_pose(
    world, environment_name, pickup_position=None, pickup_yaw=None
):
    if pickup_position is not None:
        x, y, z = [float(value) for value in pickup_position]
        yaw = 0.0 if pickup_yaw is None else float(pickup_yaw)
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            reference_frame=world.root,
        )

    normalized_environment = resolve_environment_name(environment_name)
    if normalized_environment not in HANDPICKED_PICKUP_POSES:
        supported = ", ".join(sorted(HANDPICKED_PICKUP_POSES))
        raise ValueError(
            "No handpicked pickup pose configured for "
            f"'{normalized_environment}'. Supported: {supported}. "
            "Set PICKUP_POSITION and PICKUP_YAW in main.py for this environment."
        )

    x, y, z, yaw = HANDPICKED_PICKUP_POSES[normalized_environment]
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x=float(x),
        y=float(y),
        z=float(z),
        yaw=float(yaw),
        reference_frame=world.root,
    )


def _spawn_cutting_board_under_object(
    *,
    world,
    object_kind,
    spawn_position,
    spawn_yaw,
    spawn_scale,
    board_scale=1.0,
):
    object_cfg = get_cut_object_config(object_kind)
    object_preview = _parse_stl(*object_cfg["mesh_parts"])
    _set_uniform_scale(
        object_preview,
        (float(spawn_scale), float(spawn_scale), float(spawn_scale)),
        color=object_cfg["object_color"],
    )
    object_mins, _ = body_local_aabb(
        object_preview.root, use_visual=False, apply_shape_scale=True
    )

    board = _parse_stl("pycram_object_gap_demo", "board.stl")
    board.root.name.name = "cutting_board_0001"
    _set_uniform_scale(
        board,
        (float(board_scale), float(board_scale), float(board_scale)),
        color=CUTTING_BOARD_COLOR,
    )
    _, board_maxs = body_local_aabb(
        board.root, use_visual=False, apply_shape_scale=True
    )

    x_world, y_world, z_world = [float(value) for value in spawn_position]
    board_z = z_world + float(object_mins[2]) - float(board_maxs[2]) - 0.002
    board_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=x_world,
        y=y_world,
        z=board_z,
        yaw=float(spawn_yaw),
        reference_frame=world.root,
    )
    world.merge_world_at_pose(board, board_pose)
    world.update_forward_kinematics()
    return _pose_xyz(board_pose)


def run_single_object_cut_demo(
    *,
    robot_name=None,
    environment_name=None,
    object_kind="bread",
    spawn_position=(2.48, 2.11, 0.99),
    spawn_yaw=-np.pi / 2,
    spawn_scale=1.0,
    pickup_position=None,
    pickup_yaw=None,
    quiet_logs=True,
):
    if quiet_logs:
        _suppress_demo_noise()
    parse_output = io.StringIO()
    with redirect_stdout(parse_output), redirect_stderr(parse_output):
        world = setup_thesis_world(
            robot_name=robot_name,
            environment_name=environment_name,
        )
    board_xyz = _spawn_cutting_board_under_object(
        world=world,
        object_kind=object_kind,
        spawn_position=spawn_position,
        spawn_yaw=spawn_yaw,
        spawn_scale=spawn_scale,
    )
    object_name, world_xyz = _spawn_single_object(
        world=world,
        object_kind=object_kind,
        spawn_position=spawn_position,
        spawn_yaw=spawn_yaw,
        spawn_scale=spawn_scale,
    )

    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_single_object_cut_demo",
    )

    try:
        resolved_robot_name = resolve_robot_name(robot_name)
        arm_tools = attach_available_tools(
            world,
            _parse_stl,
            mesh_parts=("pycram_object_gap_demo", "big-knife.stl"),
            right_name="knife_right",
            left_name="knife_left",
            right_pose_kwargs=get_tool_mount_pose_kwargs(
                "cut", resolved_robot_name, Arms.RIGHT
            ),
            left_pose_kwargs=get_tool_mount_pose_kwargs(
                "cut", resolved_robot_name, Arms.LEFT
            ),
            tool_cls=Knife,
        )
        targets = collect_named_targets(
            world, f"{get_cut_object_config(object_kind)['object_name_prefix']}_"
        )
        if len(targets) != 1:
            raise RuntimeError(
                f"Expected exactly one spawned target, got {len(targets)}"
            )

        target = targets[0]
        cut_cfg = _cut_object_execution_config(object_kind)
        context = Context.from_world(world)
        context.ros_node = node

        with simulated_robot_without_collision:
            sequential(
                [SetGripperAction(Arms.BOTH, GripperState.CLOSE)],
                context,
            ).perform()

        print(f"[setup] robot={resolved_robot_name} environment={environment_name}")
        print(
            f"[setup] spawned {object_name} at "
            f"({world_xyz[0]:.3f}, {world_xyz[1]:.3f}, {world_xyz[2]:.3f})"
        )

        pickup_pose = _resolve_pickup_pose(
            world,
            environment_name,
            pickup_position=pickup_position,
            pickup_yaw=pickup_yaw,
        )
        pickup_xyz = _pose_xyz(pickup_pose)
        print(
            f"[setup] pickup pose at "
            f"({pickup_xyz[0]:.3f}, {pickup_xyz[1]:.3f}, {pickup_xyz[2]:.3f})"
        )

        last_error = None
        for arm, tool in arm_tools:
            print(f"[cut] trying {_body_name(target)} with {arm.name} arm")
            try:
                _try_cut(
                    context,
                    target,
                    pickup_pose,
                    arm,
                    tool,
                    cutting_technique=cut_cfg.get("technique", CUTTING_TECHNIQUE),
                    num_cuts_x=cut_cfg.get("num_cuts_x", CUTTING_NUM_CUTS_X),
                )
                print(f"[ok] {_body_name(target)} cut with {arm.name} arm")
                return
            except Exception as exc:
                last_error = exc
                print(f"[retry] {arm.name} arm failed: {type(exc).__name__}: {exc}")

        if last_error is not None:
            raise last_error
    finally:
        shutdown_experiment_runtime(node)
