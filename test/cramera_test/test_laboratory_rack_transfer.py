"""
The robot transports the loaded rack using physical carry-handle contacts.
"""

import numpy as np

from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_rack_transfer import LaboratoryRackTransfer
from cramera.laboratory_workflow import LaboratoryWorkflowProgram

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_rack_contacts import assert_rack_transfer_contacts


# %% loaded rack transport
def test_robot_carries_four_independent_samples_to_the_workbench(mobile_robot) -> None:
    """
    A verified grip lifts the carrier, transports it and releases it on the bench.
    """
    physics = mobile_robot
    physics.prepare_rack_in_drawer()
    physics.drawer.hold(physics.drawer.open_position)
    physics.step(int(18 / physics.timestep))
    action = LaboratoryRackTransfer(physics, LaboratoryWorkflowProgram.SAMPLE_KEYS)
    before = physics.data.qpos.copy()
    action.start()
    np.testing.assert_array_equal(physics.data.qpos, before)
    previous = None
    for _ in range(int(180 / physics.timestep)):
        before = physics.data.qpos.copy()
        action.advance()
        np.testing.assert_array_equal(physics.data.qpos, before)
        physics.step()
        before = physics.data.qpos.copy()
        action.observe()
        np.testing.assert_array_equal(physics.data.qpos, before)
        assert_rack_transfer_contacts(physics, action)
        if action.phase != previous:
            print(action.snapshot(), flush=True)
            previous = action.phase
        if action.state != ProgramState.RUNNING:
            break
    assert action.state == ProgramState.SUCCEEDED, action.snapshot()
    assert action.grasp_verified
    np.testing.assert_allclose(
        physics.data.xpos[physics.rack.body],
        action.destination,
        atol=action.parameters.position_tolerance,
    )
    assert physics.rack.supported_by("laboratory_room")
    for key in LaboratoryWorkflowProgram.SAMPLE_KEYS:
        assert physics.rack.occupants()[physics.rack.assignments[key]] == key
    assert not np.any(physics.data.xfrc_applied)
