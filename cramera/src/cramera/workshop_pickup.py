"""
Contact-sized approach to a loose screw among neighbouring fasteners.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.datastructures.enums import Arms
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.attachment_nodes import AttachNode
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from coraplex.view_manager import ViewManager
from cramera.pr2_gripper import Pr2GripperContact
from cramera.workshop_demo import PickUpScrewAction
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.world_description.world_entity import Body


# %% narrow bin pickup
@dataclass
class PickUpLooseScrewAction(PickUpScrewAction):
    """
    Preset finger opening above the bin and retain orientation during extraction.
    """

    approach_finger_position: float = 0.20
    """
    PR2 finger joint angle providing clearance around the authored screw head.
    """

    @property
    def _action_plan(self) -> PlanNode:
        """
        Approach through the free gap before closing against measured screw geometry.
        """
        before, target, after = self.grasp_description.grasp_pose_sequence(
            self.object_designator
        )
        hand = ViewManager.get_end_effector_view(self.arm, self.robot)
        contact = Pr2GripperContact(self.world, hand, self.object_designator)
        opening = JointState.from_mapping(
            {contact.connection: self.approach_finger_position}
        )
        return sequential(
            [
                MoveToolCenterPointMotion(
                    before,
                    self.arm,
                    max_linear_velocity=self.pre_approach_linear_velocity,
                ),
                MoveGripperMotion(GripperState.OPEN, self.arm, goal_state=opening),
                MoveToolCenterPointMotion(
                    target,
                    self.arm,
                    max_linear_velocity=self.final_approach_linear_velocity,
                ),
                self.create_grasp_motion(),
                AttachNode(body=self.object_designator, new_parent=hand.tool_frame),
                MoveToolCenterPointMotion(
                    after,
                    self.arm,
                    max_linear_velocity=self.lift_linear_velocity,
                    max_angular_velocity=0.3,
                ),
            ]
        )
