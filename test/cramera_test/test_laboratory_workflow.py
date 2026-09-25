"""
Measured evidence gates the four-sample laboratory workflow.
"""

from __future__ import annotations

import math
from unittest.mock import Mock

import numpy as np
import pytest

from cramera.laboratory_centrifuge_command import CentrifugeAction
from cramera.laboratory_mixing_program import MixingField
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import ProgramField, ProgramState
from cramera.laboratory_pipetting import PipettingSource
from cramera.laboratory_refrigerator_command import RefrigeratorField
from cramera.laboratory_workflow import (
    LaboratoryWorkflowProgram,
    WorkflowField,
    WorkflowLocation,
    WorkflowSample,
    WorkflowStage,
    WorkflowStep,
    balanced_sample_order,
)
from cramera.laboratory_world import LaboratoryBody, LaboratoryField
from cramera.laboratory_sample_transfer import SampleGrasp

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_centrifuge import seat_tube
from .test_laboratory_physics import physics
from .test_laboratory_physics_robot import robot_physics


# %% evidence and preparation
def test_balance_pairs_the_closest_measured_sample_masses():
    """
    Opposing pair order comes from observed weights rather than sample colors.
    """
    samples = [
        WorkflowSample(key, weight_grams=weight)
        for key, weight in zip(
            LaboratoryWorkflowProgram.SAMPLE_KEYS, [12.0, 13.0, 12.1, 13.1]
        )
    ]
    ordered = balanced_sample_order(samples, 0.25)
    assert [sample.key for sample in ordered] == [
        samples[0].key,
        samples[2].key,
        samples[1].key,
        samples[3].key,
    ]


def test_unbalanced_measured_masses_cannot_form_a_recipe():
    """
    The rotor remains disabled when no two measured pairs satisfy its limit.
    """
    samples = [
        WorkflowSample(key, weight_grams=weight)
        for key, weight in zip(
            LaboratoryWorkflowProgram.SAMPLE_KEYS, [12.0, 12.1, 13.0, 14.0]
        )
    ]
    with pytest.raises(InvalidPhysicsTarget):
        balanced_sample_order(samples, 0.25)


def test_a_missing_weight_is_not_interpreted_as_zero():
    """
    An unweighed sample cannot be assigned to a rotor pocket.
    """
    samples = [WorkflowSample(key) for key in LaboratoryWorkflowProgram.SAMPLE_KEYS]
    with pytest.raises(InvalidPhysicsTarget):
        balanced_sample_order(samples, 0.25)


def test_workflow_rejects_nonempty_samples_without_changing_physics(robot_physics):
    """
    Preparation is explicit; constructing or starting never empties a vessel.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    positions = robot_physics.data.qpos.copy()
    contents = robot_physics.liquid.snapshot()
    with pytest.raises(InvalidPhysicsTarget):
        program.start()
    np.testing.assert_array_equal(robot_physics.data.qpos, positions)
    assert robot_physics.liquid.snapshot() == contents
    assert program.state == ProgramState.IDLE


def test_prepared_workflow_starts_without_moving_or_refilling_samples(robot_physics):
    """
    Empty rack samples transfer ownership without a hidden world reset.
    """
    for key in LaboratoryWorkflowProgram.SAMPLE_KEYS:
        robot_physics.fill_liquid(key, 0.0)
    program = LaboratoryWorkflowProgram(robot_physics)
    positions = robot_physics.data.qpos.copy()
    contents = robot_physics.liquid.snapshot()
    program.start()
    np.testing.assert_array_equal(robot_physics.data.qpos, positions)
    assert robot_physics.liquid.snapshot() == contents
    assert program.state == ProgramState.RUNNING
    assert program.snapshot()[ProgramField.SOURCE] == WorkflowField.SOURCE


def test_missing_reagent_is_rejected_before_motion(robot_physics):
    """
    Every sample must have enough inventory for both requested doses.
    """
    for key in LaboratoryWorkflowProgram.SAMPLE_KEYS:
        robot_physics.fill_liquid(key, 0.0)
    program = LaboratoryWorkflowProgram(robot_physics)
    source = next(iter(PipettingSource))
    robot_physics.pipetting.reservoirs[source].volume_ml = program.parameters.dose_ml
    with pytest.raises(InvalidPhysicsTarget):
        program.start()
    assert program.state == ProgramState.IDLE


def test_snapshot_does_not_report_unmeasured_weights_or_doses(robot_physics):
    """
    The progress panel distinguishes planned quantities from actual evidence.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    status = program.snapshot()[WorkflowField.WORKFLOW]
    assert status[WorkflowField.STAGE] == WorkflowStage.PREPARE
    for sample in status[WorkflowField.SAMPLES]:
        assert sample[WorkflowField.DOSE_A] is None
        assert sample[WorkflowField.DOSE_B] is None
        assert sample[WorkflowField.WEIGHT] is None


def test_unstable_scale_does_not_advance_or_record_a_weight(robot_physics):
    """
    Only a settled load from the intended sole sample becomes weighing evidence.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    program.state = ProgramState.RUNNING
    program.step_index = next(
        index
        for index, step in enumerate(program.steps)
        if step.operation == WorkflowStep.WEIGH
    )
    program.observe()
    assert program.step.operation == WorkflowStep.WEIGH
    assert program.samples[0].weight_grams is None


def test_paused_workflow_retains_partial_evidence(robot_physics):
    """
    Resuming does not require the initial empty-rack precondition again.
    """
    for key in LaboratoryWorkflowProgram.SAMPLE_KEYS:
        robot_physics.fill_liquid(key, 0.0)
    program = LaboratoryWorkflowProgram(robot_physics)
    program.start()
    program.samples[0].dose_a_ml = program.parameters.dose_ml
    robot_physics.fill_liquid(LaboratoryBody.CLEAR_TUBE, program.parameters.dose_ml)
    program.pause()
    positions = robot_physics.data.qpos.copy()
    contents = robot_physics.liquid.snapshot()
    program.start()
    np.testing.assert_array_equal(robot_physics.data.qpos, positions)
    assert robot_physics.liquid.snapshot() == contents
    assert program.samples[0].dose_a_ml == program.parameters.dose_ml


def test_stopped_rotor_without_completed_timer_cannot_report_processing(robot_physics):
    """
    An interrupted or never-started cycle cannot become successful spin evidence.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    program.state = ProgramState.RUNNING
    program.step_index = next(
        index
        for index, step in enumerate(program.steps)
        if step.operation == WorkflowStep.SPIN
    )
    program.issued = True
    program.observe()
    assert program.state == ProgramState.FAILED
    assert not program.spin_finished


def test_workflow_retains_instrument_ownership_while_paused(robot_physics):
    """
    Paused sample handling remains reserved until an explicit reset.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    program.state = ProgramState.PAUSED
    assert program.snapshot()[WorkflowField.WORKFLOW]["reserved"] is True


def test_unbalanced_weights_stop_workflow_without_escaping_observation(robot_physics):
    """
    A rejected measured load becomes visible failure rather than killing the worker.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    program.state = ProgramState.RUNNING
    program.step_index = next(
        index
        for index, step in enumerate(program.steps)
        if step.operation == WorkflowStep.BALANCE
    )
    for sample, weight in zip(program.samples, [12.0, 12.1, 13.0, 14.0]):
        sample.weight_grams = weight
    program.observe()
    assert program.state == ProgramState.FAILED
    assert not any(sample.centrifuge_slot for sample in program.samples)


def test_supported_storage_without_recipe_contents_cannot_finish(stored_workflow):
    """
    Final placement cannot hide a sample that lost its required liquid dose.
    """
    for sample in stored_workflow.samples:
        stored_workflow.physics.liquid.fill(sample.key, 0.0)
    stored_workflow._next()
    assert stored_workflow.state == ProgramState.FAILED


# %% measured instrument execution
def test_four_sample_spin_waits_for_timer_and_complete_braking(physics):
    """
    A real balanced timed cycle advances only after physical rotation ends.
    """
    program = LaboratoryWorkflowProgram(physics)
    for sample, slot in zip(program.samples, physics.centrifuge.slots.values()):
        physics.fill_liquid(
            sample.key, len(PipettingSource) * program.parameters.dose_ml
        )
        seat_tube(physics, sample.key, slot.identifier)
        physics.centrifuge_command("secure", slot=slot.identifier, key=sample.key)
    physics.centrifuge_command("close")
    physics.step(5000)
    program.state = ProgramState.RUNNING
    program.step_index = next(
        index
        for index, step in enumerate(program.steps)
        if step.operation == WorkflowStep.SPIN
    )
    initial_time = physics.data.time
    program.observe()
    while (
        program.state == ProgramState.RUNNING
        and program.step.operation == WorkflowStep.SPIN
    ):
        for _ in range(100):
            program.advance()
            physics.step()
            program.observe()
            if program.step.operation != WorkflowStep.SPIN:
                break
    assert program.state == ProgramState.RUNNING, program.snapshot()
    assert program.spin_finished
    assert physics.centrifuge.stopped
    assert physics.centrifuge.remaining_seconds == 0
    assert physics.data.time - initial_time >= program.parameters.spin_seconds
    assert program.step.operation == WorkflowStep.OPEN_CENTRIFUGE


@pytest.mark.parametrize("state", [ProgramState.RUNNING, ProgramState.PAUSED])
def test_interlock_timer_jump_never_counts_as_a_completed_cycle(robot_physics, state):
    """
    An interlock may clear remaining time without completing the requested run.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    program.state = state
    program.step_index = next(
        index
        for index, step in enumerate(program.steps)
        if step.operation == WorkflowStep.SPIN
    )
    program.issued = True
    program.spin_remaining = program.parameters.spin_seconds
    robot_physics.centrifuge.timer_started = True
    robot_physics.centrifuge.remaining_seconds = 0
    program.observe()
    assert program.state == ProgramState.FAILED
    assert not program.spin_finished


# %% measured placement before the next rack pickup
def test_each_sample_reaches_rotor_before_the_next_pickup(robot_physics):
    """
    Vacating front rack positions permits later front grasps in the rear row.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    for index in range(len(program.samples) - 1):
        secured = next(
            position
            for position, step in enumerate(program.steps)
            if step.sample_index == index and step.operation == WorkflowStep.SECURE
        )
        pickup = next(
            position
            for position, step in enumerate(program.steps)
            if step.sample_index == index + 1
            and step.operation == WorkflowStep.DISPENSER
        )
        assert secured < pickup


def test_next_rotor_slot_matches_an_actual_weighed_opposite(robot_physics):
    """
    The third load can match the first sample after a heavier second load.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    program.state = ProgramState.RUNNING
    identifiers = list(robot_physics.centrifuge.slots)
    first, second, third = program.samples[:3]
    first.weight_grams = 12.0
    second.weight_grams = 13.0
    third.weight_grams = 12.1
    first.centrifuge_slot = identifiers[0]
    second.centrifuge_slot = identifiers[1]
    robot_physics.centrifuge.slots[identifiers[0]].key = first.key
    robot_physics.centrifuge.slots[identifiers[1]].key = second.key
    program.step_index = next(
        index
        for index, step in enumerate(program.steps)
        if step.operation == WorkflowStep.ASSIGN and step.sample_index == 2
    )
    program.observe()
    assert (
        third.centrifuge_slot
        == robot_physics.centrifuge.slots[first.centrifuge_slot].opposite
    )


def test_batch_progress_requires_all_four_measured_samples(robot_physics):
    """
    The first sample's doses and weighing do not complete the whole batch.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    program.samples[0].dose_a_ml = program.parameters.dose_ml
    program.samples[0].dose_b_ml = program.parameters.dose_ml
    program.samples[0].weight_grams = 12.0
    completed = program.snapshot()[WorkflowField.WORKFLOW][WorkflowField.COMPLETED]
    assert WorkflowStage.FILL not in completed
    assert WorkflowStage.WEIGH not in completed
    assert WorkflowStage.LOAD not in completed


def test_processed_samples_are_regrasped_on_rack_before_cooler(robot_physics):
    """
    The rotor's top pickup changes to a front grasp on an intermediate support.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    for index in range(len(program.samples)):
        operations = [
            step.operation
            for step in program.steps
            if step.sample_index == index and step.stage == WorkflowStage.STORE
        ]
        assert operations == [
            WorkflowStep.RELEASE,
            WorkflowStep.UNLOAD,
            WorkflowStep.STORE,
            WorkflowStep.CONFIRM_STORAGE,
        ]


# %% supported grasp changes at the front of the rack
def test_weighed_samples_visit_regrasp_support_before_rotor(robot_physics):
    """
    Every measured sample changes grasp direction on the empty front support.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    for sample_index in range(len(program.samples)):
        operations = [
            step.operation
            for step in program.steps
            if step.sample_index == sample_index
        ]
        assert (
            operations.index(WorkflowStep.WEIGH)
            < operations.index(WorkflowStep.REGRASP)
            < operations.index(WorkflowStep.LOAD)
        )


@pytest.mark.parametrize(
    ("operation", "grasp"),
    [(WorkflowStep.REGRASP, SampleGrasp.FRONT), (WorkflowStep.UNLOAD, SampleGrasp.TOP)],
)
def test_grasp_changes_use_the_authored_front_rack_support(
    robot_physics, monkeypatch, operation, grasp
):
    """
    Both grasp changes release on the same named collision support.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    program.step_index = next(
        index for index, step in enumerate(program.steps) if step.operation == operation
    )
    factory = Mock(return_value=Mock())
    monkeypatch.setattr(
        "cramera.laboratory_sample_transfer.LaboratorySampleTransfer", factory
    )
    program._start_transfer()
    arguments = factory.call_args.kwargs
    rack = robot_physics.scene[LaboratoryField.LABORATORY][MixingField.RACK_SLOTS]
    expected = next(
        slot[LaboratoryField.POSE][:3]
        for slot in rack
        if slot[LaboratoryField.ID] == program.REGRASP_SLOT
    )
    assert arguments["destination"] == tuple(expected)
    assert arguments["support_body"] == LaboratoryBody.RACK
    assert arguments["grasp"] == grasp
    assert arguments["release_at_destination"]
    assert not arguments["from_held"]


def test_completed_regrasp_transfer_reports_the_actual_rack_slot(robot_physics):
    """
    The visible sample location distinguishes A1 from its original support.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    program.state = ProgramState.RUNNING
    program.step_index = next(
        index
        for index, step in enumerate(program.steps)
        if step.operation == WorkflowStep.REGRASP
    )
    program.motion = Mock(state=ProgramState.SUCCEEDED)
    sample = program.sample
    program.observe()
    assert sample.location == WorkflowLocation.REGRASP
    assert sample.snapshot()[WorkflowField.RACK_SLOT] == program.REGRASP_SLOT


# %% final measured instrument state
@pytest.fixture
def stored_workflow(robot_physics, monkeypatch):
    """
    Provide completed sample evidence before accepting the final instrument state.
    """
    program = LaboratoryWorkflowProgram(robot_physics)
    program.state = ProgramState.RUNNING
    program.step_index = len(program.steps) - 1
    program.spin_finished = True
    robot_physics.centrifuge_command(CentrifugeAction.CLOSE)
    robot_physics.data.qpos[
        robot_physics.model.jnt_qposadr[robot_physics.centrifuge.lid_joint]
    ] = 0.0
    for sample, slot in zip(
        program.samples, robot_physics.refrigerator.metadata[RefrigeratorField.SLOTS]
    ):
        sample.refrigerator_slot = slot[RefrigeratorField.ID]
        robot_physics.fill_liquid(
            sample.key, len(PipettingSource) * program.parameters.dose_ml
        )
    monkeypatch.setattr(
        robot_physics.refrigerator,
        "occupants",
        lambda: {sample.refrigerator_slot: sample.key for sample in program.samples},
    )
    assert robot_physics.refrigerator.door_closed
    assert robot_physics.centrifuge.lid_closed
    assert robot_physics.centrifuge.stopped
    return program


def test_final_cooler_door_is_checked_after_the_last_lid_operation(stored_workflow):
    """
    A cooler door that reopens after its accepted close step prevents success.
    """
    physics = stored_workflow.physics
    physics.data.qpos[physics.model.jnt_qposadr[physics.refrigerator.joint]] = (
        physics.refrigerator.open_angle
    )
    stored_workflow._next()
    assert stored_workflow.state == ProgramState.FAILED


def test_final_centrifuge_lid_must_be_closed(stored_workflow):
    """
    An observed open rotor cover cannot be part of a completed laboratory.
    """
    physics = stored_workflow.physics
    physics.data.qpos[physics.model.jnt_qposadr[physics.centrifuge.lid_joint]] = (
        physics.centrifuge.open_angle
    )
    stored_workflow._next()
    assert stored_workflow.state == ProgramState.FAILED


def test_final_rotor_must_be_stopped(stored_workflow):
    """
    Previously completed processing does not excuse current rotor motion.
    """
    physics = stored_workflow.physics
    physics.data.qvel[physics.model.jnt_dofadr[physics.centrifuge.rotor_joint]] = (
        2 * physics.centrifuge.parameters.stopped_rpm * math.tau / 60
    )
    stored_workflow._next()
    assert stored_workflow.state == ProgramState.FAILED


def test_final_closed_supported_storage_can_complete(stored_workflow):
    """
    Intact stored samples and closed stationary devices complete the recipe.
    """
    stored_workflow._next()
    assert stored_workflow.state == ProgramState.SUCCEEDED
