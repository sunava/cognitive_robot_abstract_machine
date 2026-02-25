from dataclasses import dataclass, field
from typing import Optional

from semantic_digital_twin.robots.abstract_robot import ParallelGripper
from semantic_digital_twin.world_description.world_entity import Body
from .base import BaseMotion
from giskardpy.motion_statechart.goals.pick_up import PickUp


@dataclass
class PickupMotion(BaseMotion):
    """
    Picks up an object with the manipulator
    """

    manipulator: ParallelGripper
    object_geometry: Body
    gripper_vertical: Optional[bool] = field(default=True, kw_only=True)

    def perform(self):
        return

    @property
    def _motion_chart(self):
        pickup = PickUp(
            manipulator=self.manipulator,
            object_geometry=self.object_geometry,
            gripper_vertical=self.gripper_vertical,
        )
        return pickup
