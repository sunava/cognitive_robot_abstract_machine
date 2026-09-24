"""
Fresh workflow preparation and paused physical state remain distinct.
"""

from enum import StrEnum

import numpy as np
import pytest

from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_centrifuge_command import CentrifugeAction
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics, RobotBaseJoint
from cramera.laboratory_workflow import LaboratoryWorkflowProgram
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_mobile_robot import mobile_robot


# %% explicit demo preparation
class CompetingOperation(StrEnum):
    """Operations that cannot take ownership of a paused workflow's samples."""

    TARGET = "target"
    """Manual sample movement."""
    LIQUID = "liquid"
    """Manual replacement of liquid contents."""
    SCALE = "scale"
    """Operator adjustment of the balance zero."""
    TRANSFER = "transfer"
    """The original single-sample transport demo."""
    MIXING = "mixing"
    """The original two-reagent mixing demo."""
    PIPETTING = "pipetting"
    """The original stationary dispenser demo."""


@pytest.fixture
def workflow_world(mobile_robot: LaboratoryRobotPhysics) -> LaboratoryProgramPhysics:
    """
    Keep real sample physics around the serialized program selection adapter.
    """
    return LaboratoryProgramPhysics(mobile_robot, mobile_robot.robot_directory)


def test_preparation_empties_four_samples_and_preserves_manual(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Fresh setup clears specimen contents without replacing accepted operating rules.
    """
    manual = workflow_world.physics.centrifuge.knowledge.active
    workflow_world.run_workflow(prepare=True)
    assert isinstance(workflow_world.program, LaboratoryWorkflowProgram)
    assert workflow_world.program.state == ProgramState.RUNNING
    assert workflow_world.physics.centrifuge.knowledge.active is manual
    assert all(
        workflow_world.physics.liquid.tubes[key].volume_ml == 0
        for key in LaboratoryWorkflowProgram.SAMPLE_KEYS
    )


def test_resume_without_prepared_workflow_is_rejected(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    A resume request never silently prepares or clears the laboratory.
    """
    before = workflow_world.physics.data.qpos.copy()
    liquid = workflow_world.physics.liquid.snapshot()
    with pytest.raises(InvalidPhysicsTarget):
        workflow_world.run_workflow(prepare=False)
    np.testing.assert_array_equal(workflow_world.physics.data.qpos, before)
    assert workflow_world.physics.liquid.snapshot() == liquid


def test_paused_resume_preserves_pose_and_liquid_ledger(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Resume reuses the same controller without resetting partially processed samples.
    """
    workflow_world.run_workflow(prepare=True)
    workflow_world.pause_program()
    physics = workflow_world.physics
    physics.fill_liquid(LaboratoryWorkflowProgram.SAMPLE_KEYS[0], 1)
    before = physics.data.qpos.copy()
    liquid = physics.liquid.snapshot()
    controller = workflow_world.program
    workflow_world.run_workflow(prepare=False)
    assert workflow_world.program is controller
    assert controller.state == ProgramState.RUNNING
    np.testing.assert_array_equal(physics.data.qpos, before)
    assert physics.liquid.snapshot() == liquid


@pytest.mark.parametrize("prepare", [True, False])
def test_running_request_cannot_restart_samples(
    workflow_world: LaboratoryProgramPhysics, prepare: bool
) -> None:
    """
    Active preparation is rejected while duplicate resume is state-preserving.
    """
    workflow_world.run_workflow(prepare=True)
    controller = workflow_world.program
    before = workflow_world.physics.data.qpos.copy()
    if prepare:
        with pytest.raises(InvalidPhysicsTarget):
            workflow_world.run_workflow(prepare=prepare)
    else:
        workflow_world.run_workflow(prepare=prepare)
    assert workflow_world.program is controller
    np.testing.assert_array_equal(workflow_world.physics.data.qpos, before)


@pytest.mark.parametrize("operation", list(CompetingOperation))
def test_paused_workflow_blocks_competing_operations(
    workflow_world: LaboratoryProgramPhysics, operation: CompetingOperation
) -> None:
    """
    Direct adapter callers cannot edit or replace a paused processing workflow.
    """
    workflow_world.run_workflow(prepare=True)
    workflow_world.pause_program()
    with pytest.raises(InvalidPhysicsTarget):
        if operation == CompetingOperation.TARGET:
            workflow_world.set_target(LaboratoryBody.CLEAR_TUBE, [0.2, 0, 1.1])
        elif operation == CompetingOperation.LIQUID:
            workflow_world.fill_liquid(LaboratoryBody.CLEAR_TUBE, 2)
        elif operation == CompetingOperation.SCALE:
            workflow_world.tare_scale()
        elif operation == CompetingOperation.TRANSFER:
            workflow_world.run_program()
        elif operation == CompetingOperation.MIXING:
            workflow_world.run_mixing()
        else:
            workflow_world.run_pipetting()
    assert workflow_world.program.state == ProgramState.PAUSED


def test_emergency_rotor_stop_pauses_workflow(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Stopping the rotor also holds the workflow's robot intentions.
    """
    workflow_world.run_workflow(prepare=True)
    workflow_world.centrifuge_command(CentrifugeAction.STOP)
    assert workflow_world.program.state == ProgramState.PAUSED


def test_stationary_pipetting_rejects_mobile_base_away_from_home(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """A stationary recipe cannot inherit the mobile workflow's remote dock."""
    physics = workflow_world.physics
    physics.set_joint_targets({RobotBaseJoint.HORIZONTAL: -0.4})
    physics.step(3000)
    before = physics.data.qpos.copy()
    with pytest.raises(InvalidPhysicsTarget):
        workflow_world.run_pipetting()
    np.testing.assert_array_equal(physics.data.qpos, before)
