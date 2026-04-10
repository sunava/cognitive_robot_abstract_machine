from dataclasses import dataclass, field
from typing import Optional

from giskardpy.motion_statechart.goals.place import Place
from pycram.robot_plans import BaseMotion
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PlaceMotion(BaseMotion):
    """
    Motion for placing an object, i.e., moving the gripper to a certain pose
    It creates a _motion_chart that is used by the motion framework
    It directly calls the implemented PickUp of Giskard.
    """

    gripper: Manipulator = field(kw_only=True)
    """
    Name of the gripper that should be moved
    """

    object_designator: Body = field(kw_only=True)
    """
    Object designator_description describing the object that should be placed
    """
    goal_pose: HomogeneousTransformationMatrix = field(kw_only=True)
    """
    The goal_pose at which the object should be placed
    """

    simulated: bool = field(default=True, kw_only=True)
    """
    Parsing simulation argument
    """

    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper is allowed to collide with something
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        goal_pose = self.goal_pose

        return Place(
            manipulator=self.gripper,
            object_geometry=self.object_designator,
            goal=goal_pose,
            simulated=self.simulated,
        )
