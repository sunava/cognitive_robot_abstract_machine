"""Minimal cutting demo: the PR2 stands in the apartment and cuts a bread.

A stripped-down variant of ``single_object_cut_demo`` with everything but the
cutting path removed:

* robot and environment are hard-coded (PR2 in the apartment),
* no EQL knowledge resolution / semantic affordances (not needed by
  ``CuttingAction`` itself),
* no wipe/pour/mix branches, no retry loop, no quiet-logging wrappers.

Flow: build world -> spawn cutting board + bread -> attach a knife ->
close grippers -> navigate to the counter -> run ``CuttingAction``.
"""

import numpy as np

from demos.thesis_new.src.demo_cut_all_breads_retry import (
    CUTTING_POINTER_STRIDE,
    CUTTING_SLICE_THICKNESS_M,
    _parse_stl,
)
from demos.thesis_new.src.spawn_random_breads import (
    _set_uniform_scale,
    get_cut_object_config,
)
from demos.thesis_new.src.tool_mounts import get_tool_mount_pose_kwargs
from demos.thesis_new.src.utils.demo_utils import (
    attach_available_tools,
    collect_named_targets,
    get_park_arms_argument,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
)
from demos.thesis_new.src.world_setup import setup_thesis_world

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, CuttingTechnique
from pycram.motion_executor import (
    simulated_robot_with_collision,
    simulated_robot_without_collision,
)
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    body_local_aabb,
)
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Knife
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color

# --- fixed scenario --------------------------------------------------------
ROBOT = "pr2"
ENVIRONMENT = "apartment"
OBJECT_KIND = "bread"

# x, y, z, yaw (taken from the apartment tables in single_object_cut_demo).
SPAWN_POSE = (2.48, 2.11, 0.99, -np.pi / 2)  # where the bread/board sit
PICKUP_POSE = (1.5, 2.11, 0.99, 0.0)  # where the PR2 stands to cut

CUTTING_BOARD_COLOR = Color(R=0.50, G=0.36, B=0.22)
NUM_CUTS_X = 1


def _spawn_bread(world, spawn_xyz, spawn_yaw, scale=1.0):
    """Spawn the bread to cut and return its body name prefix match key."""
    cfg = get_cut_object_config(OBJECT_KIND)
    name = f"{cfg['object_name_prefix']}_0001"
    bread = _parse_stl(*cfg["mesh_parts"])
    bread.root.name.name = name
    _set_uniform_scale(bread, (scale, scale, scale), color=cfg["object_color"])
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=spawn_xyz[0],
        y=spawn_xyz[1],
        z=spawn_xyz[2],
        yaw=spawn_yaw,
        reference_frame=world.root,
    )
    world.merge_world_at_pose(bread, pose)
    world.update_forward_kinematics()
    return cfg["object_name_prefix"]


def _spawn_cutting_board(world, spawn_xyz, spawn_yaw, scale=1.0):
    """Spawn a cutting board sitting just below the bread."""
    cfg = get_cut_object_config(OBJECT_KIND)
    bread_preview = _parse_stl(*cfg["mesh_parts"])
    _set_uniform_scale(bread_preview, (scale, scale, scale), color=cfg["object_color"])
    bread_mins, _ = body_local_aabb(
        bread_preview.root, use_visual=False, apply_shape_scale=True
    )

    board = _parse_stl("pycram_object_gap_demo", "board.stl")
    board.root.name.name = "cutting_board_0001"
    _set_uniform_scale(board, (1.0, 1.0, 1.0), color=CUTTING_BOARD_COLOR)
    _, board_maxs = body_local_aabb(board.root, use_visual=False, apply_shape_scale=True)

    board_z = spawn_xyz[2] + float(bread_mins[2]) - float(board_maxs[2]) - 0.002
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=spawn_xyz[0],
        y=spawn_xyz[1],
        z=board_z,
        yaw=spawn_yaw,
        reference_frame=world.root,
    )
    world.merge_world_at_pose(board, pose)
    world.update_forward_kinematics()


def main():
    world = setup_thesis_world(robot_name=ROBOT, environment_name=ENVIRONMENT)

    spawn_xyz = SPAWN_POSE[:3]
    spawn_yaw = float(SPAWN_POSE[3])
    _spawn_cutting_board(world, spawn_xyz, spawn_yaw)
    bread_prefix = _spawn_bread(world, spawn_xyz, spawn_yaw)

    node = setup_experiment_runtime(world=world, node_name="pr2_cutting_demo")
    try:
        # Mount a knife on each available hand.
        arm_tools = attach_available_tools(
            world,
            _parse_stl,
            mesh_parts=("pycram_object_gap_demo", "big-knife.stl"),
            right_name="knife_right",
            left_name="knife_left",
            right_pose_kwargs=get_tool_mount_pose_kwargs("cut", ROBOT, Arms.RIGHT),
            left_pose_kwargs=get_tool_mount_pose_kwargs("cut", ROBOT, Arms.LEFT),
            tool_cls=Knife,
        )
        if not arm_tools:
            raise RuntimeError("No arm/tool available to cut with.")
        # Prefer the right hand for cutting.
        arm, tool = next(
            ((a, t) for a, t in arm_tools if a == Arms.RIGHT),
            arm_tools[0],
        )

        target = collect_named_targets(world, f"{bread_prefix}_")[0]

        context = Context.from_world(world)
        context.ros_node = node

        pickup_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=PICKUP_POSE[0],
            y=PICKUP_POSE[1],
            z=PICKUP_POSE[2],
            yaw=PICKUP_POSE[3],
            reference_frame=world.root,
        )

        # Close grippers, raise torso, drive to the counter.
        with simulated_robot_without_collision:
            sequential(
                [SetGripperAction(Arms.BOTH, GripperState.CLOSE)], context
            ).perform()
            sequential(
                [
                    ParkArmsAction(
                        get_park_arms_argument(world),
                        joint_state=StaticJointState.PARKTOOL,
                    )
                ],
                context,
            ).perform()
            sequential([MoveTorsoAction(TorsoState.HIGH)], context).perform()
            sequential(
                [NavigateAction(pickup_pose, True, teleport=True)], context
            ).perform()

        # Cut.
        with simulated_robot_with_collision:
            sequential(
                [
                    CuttingAction(
                        container=target,
                        arm=arm,
                        tool=tool,
                        technique=CuttingTechnique.SAW,
                        pointer_stride=CUTTING_POINTER_STRIDE,
                        num_cuts_x=NUM_CUTS_X,
                        slice_thickness=CUTTING_SLICE_THICKNESS_M,
                    ),
                    ParkArmsAction(get_park_arms_argument(world)),
                ],
                context,
            ).perform()
    finally:
        shutdown_experiment_runtime(node)


if __name__ == "__main__":
    main()
