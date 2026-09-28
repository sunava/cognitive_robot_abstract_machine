"""
Explicit driving is separate from manipulation at a stationary base.
"""

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import execute_single
from cramera.workshop_gaze import WorkshopAnchoredToolCenterPoint
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import Pose


# %% stationary manipulation
def test_mobile_workshop_tool_goal_holds_the_current_base(pr2_world_copy):
    """
    A free mobile drive receives an explicit fixed world pose during arm motion.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    motion = WorkshopAnchoredToolCenterPoint(
        Pose(reference_frame=world.root), Arms.LEFT
    )
    execute_single(motion, context=Context(world=world, robot=robot))
    hold = motion._motion_chart.nodes[-1]
    assert hold.tip_link is robot.root
    assert hold.root_link is world.root
    assert hold.goal_pose.reference_frame is robot.root
