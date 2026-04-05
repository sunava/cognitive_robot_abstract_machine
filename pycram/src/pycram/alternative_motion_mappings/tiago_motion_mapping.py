from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from pycram.datastructures.enums import ExecutionType
from pycram.robot_plans import MoveMotion
from pycram.robot_plans.motions.base import AlternativeMotion
from semantic_digital_twin.robots.tiago import Tiago


class TiagoMoveSim(MoveMotion, AlternativeMotion[Tiago]):
    """
    Uses a diff drive goal for the tiago base.
    """

    execution_type = ExecutionType.SIMULATED

    def perform(self):
        return

    @property
    def _motion_chart(self):
        if self.teleport:
            return SetOdometry(
                base_pose=self.target.to_homogeneous_matrix(),
                odom_connection=self.robot.drive,
            )

        return DifferentialDriveBaseGoal(
            goal_pose=self.target,
        )
