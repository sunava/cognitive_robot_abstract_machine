"""
Individual laboratory actions operate on the existing physical scene.
"""

from __future__ import annotations

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import TaskStatus
from cramera.laboratory_actions import (
    SampleDestination,
    TransferSampleAction,
    WeighSampleAction,
)
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_workflow import (
    LaboratoryWorkflowProgram,
    WorkflowField,
    WorkflowStage,
)
from cramera.laboratory_world import LaboratoryBody
from semantic_digital_twin.robots.pr2 import PR2

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_scale import place_on_pan
from .test_laboratory_workflow_adapter import workflow_world


# %% independently reusable actions
def test_weigh_action_uses_the_existing_sample_without_preparing_the_full_recipe(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    A native weighing plan preserves loaded liquids and the measured pan placement.
    """
    physics = workflow_world.physics
    physics.fill_liquid(LaboratoryBody.CLEAR_TUBE, 3.2)
    place_on_pan(physics)
    physics.step(2500)
    assert physics.scale.stable
    before_positions = physics.data.qpos.copy()
    before_time = physics.data.time
    before_liquids = {
        key: sample.volume_ml for key, sample in physics.liquid.tubes.items()
    }
    measured_mass = physics.scale.gross_grams
    program = LaboratoryWorkflowProgram(physics)
    try:
        program.start_actions([WeighSampleAction(LaboratoryBody.CLEAR_TUBE)])
        np.testing.assert_array_equal(physics.data.qpos, before_positions)
        assert physics.data.time == before_time
        for _ in range(100):
            program.advance()
            physics.step()
            program.observe()
            if program.state != ProgramState.RUNNING:
                break

        assert program.state == ProgramState.SUCCEEDED
        sample = next(
            item for item in program.samples if item.key == LaboratoryBody.CLEAR_TUBE
        )
        assert sample.weight_grams == pytest.approx(measured_mass, abs=0.0001)
        assert {
            key: sample.volume_ml for key, sample in physics.liquid.tubes.items()
        } == before_liquids
        assert isinstance(program.execution.plan.context, Context)
        assert isinstance(program.execution.plan.robot, PR2)
        assert program.execution.plan.root.status == TaskStatus.SUCCEEDED
        assert program.execution.completed == 1
        assert len(program.execution.plan.actions) == 1
        completed = program.snapshot()[WorkflowField.WORKFLOW][WorkflowField.COMPLETED]
        assert WorkflowStage.STORE not in completed
        assert WorkflowStage.COMPLETE not in completed
    finally:
        program.reset()


def test_held_sample_uses_existing_contacts_without_a_previous_transfer_object(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Bilateral contact ownership remains measurable across independent actions.
    """
    physics = workflow_world.physics
    program = LaboratoryWorkflowProgram(physics)
    try:
        program.start_actions(
            [
                TransferSampleAction(
                    LaboratoryBody.CLEAR_TUBE,
                    SampleDestination.DISPENSER,
                    release_at_destination=False,
                )
            ]
        )
        for _ in range(150000):
            program.advance()
            physics.step()
            program.observe()
            if program.state != ProgramState.RUNNING:
                break

        assert program.state == ProgramState.SUCCEEDED, program.snapshot()
        assert program.motion.holding
        assert program._held_sample()
        before = physics.data.qpos.copy()
        program.motion = None

        assert program._held_sample()
        np.testing.assert_array_equal(physics.data.qpos, before)
    finally:
        program.reset()
