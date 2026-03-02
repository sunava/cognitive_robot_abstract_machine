from pycram.motion_executor import simulated_robot
from demos.pycram_suturo_demos.helper_methods_and_useful_classes.pickup_helper_methods import (
    detach_object_from_hsrb,
)
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.motion_executor import real_robot
from pycram.robot_plans import (
    GiskardPlaceActionDescription,
    ParkArmsActionDescription,
)

from semantic_digital_twin.world import World


def place_demo(
    hsrb_world: World, context: Context, object_name: str, place_pose: PoseStamped
):
    object_to_place = hsrb_world.get_body_by_name(object_name)
    plan = SequentialPlan(
        context,
        GiskardPlaceActionDescription(
            object_designator=object_to_place,
            arm=Arms.LEFT,
            target_location=place_pose,
            simulated=True,
        ),
    )

    park_plan = SequentialPlan(context, ParkArmsActionDescription(Arms.BOTH))

    with simulated_robot:
        plan.perform()
        detach_object_from_hsrb(
            hsrb_world=hsrb_world,
            object_designator=object_to_place,
        )
        park_plan.perform()
