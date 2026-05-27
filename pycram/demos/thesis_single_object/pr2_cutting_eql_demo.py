import os

import rclpy
import sys
from pathlib import Path

from krrood.entity_query_language.factories import variable
from pycram.language import SequentialNode
from semantic_digital_twin.adapters.mesh import STLParser

if __package__ in (None, ""):
    pycram_root = Path(__file__).resolve().parents[2]
    if str(pycram_root) not in sys.path:
        sys.path.insert(0, str(pycram_root))


from demos.thesis_new.src.spawn_random_breads import get_cut_object_config  # noqa: E402
from demos.thesis_new.src.tool_mounts import get_tool_mount_pose_kwargs  # noqa: E402
from demos.thesis_new.src.utils.demo_utils import (  # noqa: E402
    attach_available_tools,
    get_park_arms_argument,
)
from demos.thesis_new.src.world_setup import setup_thesis_world
from demos.thesis_single_object.single_object_cut_demo import (
    _resolve_spawn_pose,
    _spawn_cutting_board_under_object,
    _spawn_single_object,
    _set_uniform_scale,
)
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.motion_executor import (
    simulated_robot_with_collision,
    simulated_robot_without_collision,
)
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.tool_based import CuttingAction
from pycram.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bread,
    CuttingBoard,
    Knife,
    ToolAttachment,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)

from semantic_digital_twin.reasoning.predicates import on_supporting_surface

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "..", "pycram", "resources")
)


def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def attach_knives(world):
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

    return arm_tools


rclpy.init()
rclpy_node = rclpy.create_node("ros_node")

ROBOT = "pr2"
ENVIRONMENT = "apartment"
OBJECT_KIND = "bread"
SPAWN_SCALE = 1.0
EXTRA_BREAD_OFFSET = (0.45, 0.0, 0.0)


CUTTING_POINTER_STRIDE = 13


world = setup_thesis_world(robot_name=ROBOT, environment_name=ENVIRONMENT)
viz = VizMarkerPublisher(_world=world, node=rclpy_node)
viz.with_tf_publisher()


spawn_position, spawn_yaw = _resolve_spawn_pose(
    world,
    ROBOT,
    ENVIRONMENT,
)

_spawn_cutting_board_under_object(
    world=world,
    object_kind=OBJECT_KIND,
    spawn_position=spawn_position,
    spawn_yaw=spawn_yaw,
    spawn_scale=SPAWN_SCALE,
)
_spawn_single_object(
    world=world,
    object_kind=OBJECT_KIND,
    spawn_position=spawn_position,
    spawn_yaw=spawn_yaw,
    spawn_scale=SPAWN_SCALE,
)

with world.modify_world():
    bread_anno = Bread(root=world.get_body_by_name("bread_0001"))
    board_anno = CuttingBoard(root=world.get_body_by_name("cutting_board_0001"))
    world.add_semantic_annotation(bread_anno)
    world.add_semantic_annotation(board_anno)

arm_tools = attach_knives(world)

with world.modify_world():
    for arm, tool in arm_tools:
        world.add_semantic_annotation(tool)
        world.add_semantic_annotation(
            ToolAttachment(
                arm=arm,
                tool=tool,
            )
        )

print(f"Breads in domain: {len(world.get_semantic_annotations_by_type(Bread))}")
print(f"Boards in domain: {len(world.get_semantic_annotations_by_type(CuttingBoard))}")
print(f"Knives in domain: {len(world.get_semantic_annotations_by_type(Knife))}")
print(f"Arm tools: {arm_tools}")

knife_domain = world.get_semantic_annotations_by_type(Knife)
attachment_domain = world.get_semantic_annotations_by_type(ToolAttachment)

attachment_var = variable(ToolAttachment, domain=attachment_domain)

# Check on_supporting_surface directly
on_surface = on_supporting_surface(bread_anno, board_anno)
print(f"on_supporting_surface(bread, board): {on_surface}")


# Since the query returns the first entity in entity() call, but we need the others too
# We might need to change the query to set_of or access the assignment if possible.
# But for now, let's assume we want to use the underspecified action which handles this.


#
context = Context.from_world(world)

with simulated_robot_without_collision:
    sequential(
        [
            ParkArmsAction(
                get_park_arms_argument(context.world),
                joint_state=StaticJointState.PARKTOOL,
            ),
            SetGripperAction(Arms.BOTH, GripperState.CLOSE),
            MoveTorsoAction(TorsoState.HIGH),
        ],
        context,
    ).perform()


with simulated_robot_with_collision:
    sequential(
        [
            CuttingAction(
                container=...,
                arm=Arms.LEFT,
                tool=tool,
                technique="saw",
                slice_thickness=0.03,
            ),
            ParkArmsAction(get_park_arms_argument(context.world)),
        ],
        context,
    ).perform()
