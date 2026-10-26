"""
A mobile robot drives around a box that blocks the straight way to its target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import execute_single
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from cramera_bridge.exceptions import TaskFailed
from cramera_bridge.geometry import NumericPose
from cramera_bridge.robot_task import RobotTaskDemo
from cramera_bridge.scenes.environment import SceneEnvironment
from cramera_protocol.bundle.acceptance import AcceptanceMetrics
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

BASE_TOLERANCE = 0.05
"""
How far, in metres, the robot's base may end up from its target.
"""

TARGET_DISTANCE = 3.0
"""
How far the target lies along the world's x axis, in metres.
"""

BARRIER_SIZE = (0.6, 0.8, 1.0)
"""
The barrier's extent along x, y and z, in metres.
"""

# %% environment


class NavigationBody(StrEnum):
    """
    The bodies of the navigation environment.
    """

    BARRIER = "navigation_barrier"
    """
    The box between the robot's start and its target.
    """


class NavigationMeasurement(StrEnum):
    """
    What the navigation demo measures.
    """

    BASE_ERROR = "baseError"
    """
    The distance in metres between the base and the target on the floor.
    """


@dataclass
class BarrierEnvironment(SceneEnvironment):
    """
    An empty floor with one box standing halfway to a target.
    """

    distance: float = TARGET_DISTANCE
    """
    How far the target lies along the world's x axis, in metres.
    """

    barrier_size: Scale = field(default_factory=lambda: Scale(*BARRIER_SIZE))
    """
    The box's size in metres.
    """

    def populate(self, world: World) -> None:
        BodySpecification.box(
            NavigationBody.BARRIER,
            self.barrier_size,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                self.distance / 2, 0, self.barrier_size.z / 2
            ),
        ).spawn(world)


# %% demo


@dataclass
class NavigationDemo(RobotTaskDemo):
    """
    Drives the robot to the far side of the barrier with CRAM's navigate action and
    measures where its base ends up.
    """

    target_x: float = TARGET_DISTANCE
    """
    The target's x coordinate in the world, in metres.
    """

    def build_plan(self, context: Context) -> Plan:
        target = Pose.from_xyz_rpy(self.target_x, reference_frame=context.world.root)
        return execute_single(NavigateAction(target), context=context).plan

    def measure(self, context: Context) -> AcceptanceMetrics:
        position = NumericPose.of_entity(context.world, context.robot.root).position
        error = math.dist(position[:2], (self.target_x, 0.0))
        if error > BASE_TOLERANCE:
            raise TaskFailed(f"the base ended {error:.3f} m from its target")
        return AcceptanceMetrics(
            measurements={NavigationMeasurement.BASE_ERROR.value: error}
        )
