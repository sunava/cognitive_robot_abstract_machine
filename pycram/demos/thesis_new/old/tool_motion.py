from pycram.robot_plans.actions.composite.tool_based import (
    GeneralizedActionPlan,
    ToolConfig,
    get_tool_config,
    tip_offset_from_body,
)


def make_tool_wrist_poses(points, world, wrist_to_tip, tool_cfg: ToolConfig):
    return GeneralizedActionPlan.make_tool_wrist_poses_for_world(
        points, world, wrist_to_tip, tool_cfg
    )
