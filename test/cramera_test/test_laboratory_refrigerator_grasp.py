"""
Refrigerator handle effort is isolated from fragile sample grasps.
"""

import numpy as np
import pytest

from cramera.laboratory_hinge_manipulation import (
    HingeInstrument,
    LaboratoryHingeManipulation,
)
from cramera.laboratory_physics_program import JointReference, ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_hinge_manipulation import assert_clear_environment_contacts
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% handle actuator restoration
def test_refrigerator_cancellation_restores_sample_gripper_profile(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    A cancelled firm handle grasp restores every original finger servo setting.
    """
    physics = mobile_robot
    finger = physics.model.actuator(JointReference.GRIPPER_JOINT).id
    gain = physics.model.actuator_gainprm.copy()
    bias = physics.model.actuator_biasprm.copy()
    torque = physics.model.actuator_forcerange.copy()
    action = LaboratoryHingeManipulation(physics, HingeInstrument.REFRIGERATOR, True)
    action.start()
    assert physics.model.actuator_gainprm[finger, 0] > gain[finger, 0]
    action.cancel()
    np.testing.assert_array_equal(physics.model.actuator_gainprm, gain)
    np.testing.assert_array_equal(physics.model.actuator_biasprm, bias)
    np.testing.assert_array_equal(physics.model.actuator_forcerange, torque)


# %% physical door operation
def test_refrigerator_cycle_clears_environment_and_restores_gripper(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Opening, stowing, closing and release preserve safe contact and sample effort.
    """
    physics = mobile_robot
    gain = physics.model.actuator_gainprm.copy()
    bias = physics.model.actuator_biasprm.copy()
    torque = physics.model.actuator_forcerange.copy()
    for opening in (True, False):
        action = LaboratoryHingeManipulation(
            physics, HingeInstrument.REFRIGERATOR, opening
        )
        action.start()
        for _ in range(int(180 / physics.timestep)):
            action.advance()
            physics.step()
            action.observe()
            assert_clear_environment_contacts(physics, action)
            if action.state != ProgramState.RUNNING:
                break
        assert action.state == ProgramState.SUCCEEDED, action.snapshot()
        assert action.grasp_verified
        assert action.maximum_motor_force == 0
        assert action.angle == pytest.approx(
            action.target_angle, abs=action.parameters.angle_tolerance
        )
        np.testing.assert_array_equal(physics.model.actuator_gainprm, gain)
        np.testing.assert_array_equal(physics.model.actuator_biasprm, bias)
        np.testing.assert_array_equal(physics.model.actuator_forcerange, torque)
