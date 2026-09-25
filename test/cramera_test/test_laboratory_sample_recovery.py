"""
Calibrated sample recovery keeps commands bounded at compliant joint stops.
"""

from enum import StrEnum
import math

import mujoco
import numpy as np

from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% measured stop deflection
class RecoveryJoint(StrEnum):
    """
    Articulation restored before the next single-sample pickup.
    """

    TORSO = "pr2/torso_lift_joint"
    """
    Vertical robot lift lowered during storage-drawer manipulation.
    """


def test_sample_recovery_bounds_commands_from_a_deflected_lower_stop(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Micrometre-scale physical stop deflection never becomes an invalid command.
    """
    physics = mobile_robot
    joint = physics.robot_joints[RecoveryJoint.TORSO]
    physics.set_joint_targets({RecoveryJoint.TORSO: joint.lower})
    deflection = 0.000002
    physics.data.qpos[joint.position_index] = joint.lower - deflection
    mujoco.mj_forward(physics.model, physics.data)
    motion = LaboratorySampleTransfer(
        physics=physics,
        object_key=LaboratoryBody.CLEAR_TUBE,
        destination=physics.pipetting.target_position,
        release_at_destination=False,
    )
    motion.start()
    assert motion.stage.joint_targets == {
        RecoveryJoint.TORSO: physics.initial_positions[RecoveryJoint.TORSO]
    }
    before = physics.data.qpos.copy()
    for _ in range(math.ceil(motion.parameters.control_period / physics.timestep)):
        motion.advance()
    np.testing.assert_array_equal(physics.data.qpos, before)
    assert physics.joint_targets[RecoveryJoint.TORSO] == joint.lower
    assert physics.data.ctrl[joint.actuator_index] == joint.lower
