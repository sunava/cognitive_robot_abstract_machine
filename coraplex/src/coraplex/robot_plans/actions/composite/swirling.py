"""
Mix liquid by moving a held container around a stationary pivot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.datastructures.enums import Arms
from coraplex.datastructures.swirling import SwirlProfile
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.composite.tool_based import FullBodyControlledAction
from coraplex.robot_plans.motions.swirling import SwirlContainerMotion
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world_description.world_entity import Body

# %% Container mixing action


@dataclass(kw_only=True)
class SwirlingAction(FullBodyControlledAction):
    """
    Mix a held container by orbiting its bottom while preserving the current grasp.

    The container's positive local Z axis points toward its mouth. Start with the
    container upright and clear of surrounding objects; acquisition and placement are
    separate actions. Giskard constrains the tilt to a cone around that initial axis.
    """

    container: Body
    """
    Container already held by the selected arm.
    """

    arm: Arms
    """
    Arm whose existing grasp is preserved.
    """

    profile: SwirlProfile = field(default_factory=SwirlProfile)
    """
    Swirl duration, tilt, cycle count and entry/exit ramps.
    """

    pivot: Point3 | None = None
    """
    Stationary pivot; defaults to the center of the container's upper rim.
    """

    @property
    def _action_plan(self) -> PlanNode:
        """
        Expand into a continuous Giskard motion with pose and tilt constraints.
        """
        motion = SwirlContainerMotion(
            container=self.container,
            arm=self.arm,
            profile=self.profile,
            pivot=self.pivot,
        )
        plan = sequential([motion], self.context)
        motion.validate_grasp()
        return plan
