"""
Starting the robot preserves the physical mass of retained liquid contents.
"""

import numpy as np

from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mixing_adapter import program_world
from .test_laboratory_physics_robot import robot_physics


# %% retained liquid load
def test_transfer_start_preserves_contained_mass_and_inertia(program_world):
    """
    Restored contents contribute their load before the next integration step.
    """
    key = LaboratoryBody.CLEAR_TUBE
    program_world.fill_liquid(key, 5.0)
    physics = program_world.physics
    body = physics.objects[key].body_id
    mass = physics.model.body_mass[body]
    inertia = physics.model.body_inertia[body].copy()
    program_world.run_program()
    assert physics.model.body_mass[body] == mass
    np.testing.assert_array_equal(physics.model.body_inertia[body], inertia)
