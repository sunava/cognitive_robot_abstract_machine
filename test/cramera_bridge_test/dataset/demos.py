"""
A short task demo for the tests: raising a robot's torso takes a few seconds in
simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import execute_single
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from cramera_bridge.exceptions import TaskFailed
from cramera_protocol.bundle.acceptance import AcceptanceMetrics
from cramera_bridge.robot_task import RobotTaskDemo
from semantic_digital_twin.datastructures.definitions import TorsoState

TORSO_TOLERANCE = 0.01
"""
How far, in metres, the torso may stop from its target height.
"""

# %% demo


class TorsoMeasurement(StrEnum):
    """
    What the torso demo measures.
    """

    TORSO_ERROR = "torsoError"
    """
    The distance between the torso joint and its target value.
    """


@dataclass
class TorsoDemo(RobotTaskDemo):
    """
    Raises the robot's torso to the given state.
    """

    torso_state: TorsoState = TorsoState.HIGH
    """
    The state the torso is moved to.
    """

    def build_plan(self, context: Context) -> Plan:
        return execute_single(MoveTorsoAction(self.torso_state), context=context).plan

    def measure(self, context: Context) -> AcceptanceMetrics:
        joint_state = context.robot.get_torso().get_joint_state_by_type(
            self.torso_state
        )
        error = max(
            abs(connection.position - target)
            for connection, target in zip(
                joint_state.connections, joint_state.target_values
            )
        )
        if error > TORSO_TOLERANCE:
            raise TaskFailed(f"the torso stopped {error:.3f} m from its target")
        return AcceptanceMetrics(
            measurements={TorsoMeasurement.TORSO_ERROR.value: error}
        )
