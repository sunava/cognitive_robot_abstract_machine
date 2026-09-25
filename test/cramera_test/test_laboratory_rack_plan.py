"""
Rack retrieval precedes all liquid processing in the native demonstration.
"""

from cramera.laboratory_actions import LaboratoryOperation, LaboratoryInstrument
from cramera.laboratory_plan import LaboratoryPlan
from cramera.laboratory_workflow import LaboratoryWorkflowProgram
from cramera.laboratory_physics_program import LaboratoryProgramPhysics

import numpy as np

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_workflow_adapter import workflow_world


# %% preparation action ordering
def test_full_recipe_retrieves_loaded_rack_before_processing() -> None:
    """
    Opening, physical rack transport and closing are distinct native actions.
    """
    actions = LaboratoryPlan(LaboratoryWorkflowProgram.SAMPLE_KEYS).actions()
    requests = [action.request for action in actions]
    assert requests[0].operation == LaboratoryOperation.OPEN_INSTRUMENT
    assert requests[0].instrument == LaboratoryInstrument.DRAWER
    assert requests[1].operation == LaboratoryOperation.RETRIEVE_RACK
    assert requests[1].required_sample_keys == LaboratoryWorkflowProgram.SAMPLE_KEYS
    assert requests[2].operation == LaboratoryOperation.CLOSE_INSTRUMENT
    assert requests[2].instrument == LaboratoryInstrument.DRAWER
    assert requests[3].operation == LaboratoryOperation.TRANSFER


def test_explicit_preparation_stores_empty_samples_with_the_rack(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    A fresh full demo begins with all four empty vials inside the closed drawer.
    """
    workflow_world.run_workflow(prepare=True)
    physics = workflow_world.physics
    assert physics.drawer.closed
    assert (
        physics.data.xpos[physics.rack.body, 2]
        < physics.model.body_pos[physics.rack.body, 2]
    )
    for key in LaboratoryWorkflowProgram.SAMPLE_KEYS:
        assert physics.liquid.tubes[key].volume_ml == 0.0
        np.testing.assert_allclose(
            physics.data.xpos[physics.objects[key].body_id],
            physics.rack.slot_position(key),
            atol=1e-12,
        )
