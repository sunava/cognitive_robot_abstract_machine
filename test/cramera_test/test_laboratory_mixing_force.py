"""
Mixing may hold a loaded glass firmly without changing other robot programs.
"""

import numpy as np

from cramera.laboratory_physics_program import JointReference

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_mixing_adapter import program_world


# %% controller-specific actuator bounds
def test_mixing_uses_its_finger_limit_and_transfer_restores_default(program_world):
    """
    A program switch restores the ordinary transfer's configured torque bound.
    """
    physics = program_world.physics
    actuator = physics.model.actuator(JointReference.GRIPPER_JOINT).id
    program_world.run_mixing()
    limit = program_world.program.parameters.finger_torque
    np.testing.assert_array_equal(
        physics.model.actuator_forcerange[actuator], [-limit, limit]
    )
    program_world.pause_program()
    program_world.run_program()
    original = physics.robot_parameters.finger_torque
    np.testing.assert_array_equal(
        physics.model.actuator_forcerange[actuator], [-original, original]
    )
