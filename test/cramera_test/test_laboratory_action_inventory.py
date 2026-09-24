"""
Independent actions resolve destinations and holders from measured inventory.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from coraplex.datastructures.enums import TaskStatus
from cramera.laboratory_actions import (
    LaboratoryActionFailed,
    ReleaseSampleAction,
    SampleDestination,
    TransferSampleAction,
    VerifyRotorBalanceAction,
)
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_manual import CentrifugeManual, ManualEdition
from cramera.laboratory_refrigerator_command import (
    RefrigeratorAction,
    RefrigeratorField,
)
from cramera.laboratory_workflow import LaboratoryWorkflowProgram
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_centrifuge import load_pair
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_workflow_adapter import workflow_world


# %% contact-supported inventory arrangements
def place_in_cooler(
    physics: LaboratoryRobotPhysics, key: LaboratoryBody, slot_index: int
) -> None:
    """
    Let an authored cooler shelf support a sample before inventory observation.
    """
    slot = physics.refrigerator.metadata[RefrigeratorField.SLOTS][slot_index]
    body = physics.objects[key].body_id
    joint = physics.model.body_jntadr[body]
    coordinate = physics.model.jnt_qposadr[joint]
    velocity = physics.model.jnt_dofadr[joint]
    physics.data.qpos[coordinate : coordinate + 3] = np.asarray(
        slot[RefrigeratorField.POSITION]
    ) + [0, 0, 0.002]
    physics.data.qpos[coordinate + 3 : coordinate + 7] = [1, 0, 0, 0]
    physics.data.qvel[velocity : velocity + 6] = 0
    mujoco.mj_forward(physics.model, physics.data)
    physics.step(1800)
    assert physics.refrigerator.occupants()[slot[RefrigeratorField.ID]] == key


# %% destinations independent of rotor preparation
def test_cooler_transfer_selects_a_measured_free_unreserved_slot(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    A standalone cooler transfer needs neither weighing nor rotor assignment.
    """
    physics = workflow_world.physics
    physics.refrigerator_command(RefrigeratorAction.OPEN)
    physics.step(6000)
    place_in_cooler(physics, LaboratoryBody.AMBER_TUBE, 0)
    slots = physics.refrigerator.metadata[RefrigeratorField.SLOTS]
    program = LaboratoryWorkflowProgram(physics)
    program.samples[2].refrigerator_slot = slots[1][RefrigeratorField.ID]
    try:
        program.start_actions(
            [
                TransferSampleAction(
                    LaboratoryBody.CLEAR_TUBE, SampleDestination.REFRIGERATOR
                )
            ]
        )
        before = physics.data.qpos.copy()
        program.advance()

        assert program.state == ProgramState.RUNNING, program.error
        assert program.sample.refrigerator_slot == slots[2][RefrigeratorField.ID]
        np.testing.assert_array_equal(
            program.motion.destination, slots[2][RefrigeratorField.POSITION]
        )
        assert program.sample.weight_grams is None
        assert program.sample.centrifuge_slot is None
        np.testing.assert_array_equal(physics.data.qpos, before)
    finally:
        program.reset()


def test_cooler_transfer_preserves_an_available_prior_reservation(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    The full recipe's already assigned storage destination remains stable.
    """
    physics = workflow_world.physics
    physics.refrigerator_command(RefrigeratorAction.OPEN)
    physics.step(6000)
    slot = physics.refrigerator.metadata[RefrigeratorField.SLOTS][-1]
    program = LaboratoryWorkflowProgram(physics)
    program.samples[0].refrigerator_slot = slot[RefrigeratorField.ID]
    try:
        program.start_actions(
            [
                TransferSampleAction(
                    LaboratoryBody.CLEAR_TUBE, SampleDestination.REFRIGERATOR
                )
            ]
        )
        program.advance()

        assert program.state == ProgramState.RUNNING, program.error
        assert program.sample.refrigerator_slot == slot[RefrigeratorField.ID]
        np.testing.assert_array_equal(
            program.motion.destination, slot[RefrigeratorField.POSITION]
        )
    finally:
        program.reset()


def test_no_available_cooler_slot_is_a_native_failure_without_pose_changes(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Occupied and reserved cooler positions cannot be silently reused.
    """
    physics = workflow_world.physics
    slots = physics.refrigerator.metadata[RefrigeratorField.SLOTS]
    program = LaboratoryWorkflowProgram(physics)
    other_samples = program.samples[1:]
    occupied_count = len(slots) - len(other_samples)
    assert occupied_count <= len(other_samples)
    for index, sample in enumerate(other_samples[:occupied_count]):
        place_in_cooler(physics, sample.key, index)
    for sample, slot in zip(other_samples, slots[occupied_count:]):
        sample.refrigerator_slot = slot[RefrigeratorField.ID]
    occupied = {
        identifier
        for identifier, key in physics.refrigerator.occupants().items()
        if key is not None
    }
    reserved = {sample.refrigerator_slot for sample in other_samples}
    assert occupied | reserved == {slot[RefrigeratorField.ID] for slot in slots}
    try:
        program.start_actions(
            [
                TransferSampleAction(
                    LaboratoryBody.CLEAR_TUBE, SampleDestination.REFRIGERATOR
                )
            ]
        )
        before = physics.data.qpos.copy()
        program.advance()
        program.execution.worker.join(timeout=2.0)

        assert program.state == ProgramState.FAILED
        assert program.execution.plan.root.status == TaskStatus.FAILED
        assert isinstance(
            program.execution.plan.actions[0].reason, LaboratoryActionFailed
        )
        assert not program.execution.worker.is_alive()
        assert program.motion is None
        np.testing.assert_array_equal(physics.data.qpos, before)
    finally:
        program.reset()


def test_rotor_transfer_requires_an_explicit_holder_assignment(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    An unassigned rotor destination fails its native action without leaving a worker.
    """
    program = LaboratoryWorkflowProgram(workflow_world.physics)
    try:
        program.start_actions(
            [
                TransferSampleAction(
                    LaboratoryBody.CLEAR_TUBE, SampleDestination.CENTRIFUGE
                )
            ]
        )
        program.advance()
        program.execution.worker.join(timeout=2.0)

        assert program.state == ProgramState.FAILED
        assert program.execution.plan.root.status == TaskStatus.FAILED
        assert not program.execution.worker.is_alive()
        assert program.motion is None
    finally:
        program.reset()


# %% operations on an existing secured load
def test_release_action_resolves_the_samples_actual_holder(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    A fresh native context can release a sample secured by an earlier operation.
    """
    physics = workflow_world.physics
    load_pair(physics)
    key = LaboratoryBody.CLEAR_TUBE
    holder = next(slot for slot in physics.centrifuge.slots.values() if slot.key == key)
    program = LaboratoryWorkflowProgram(physics)
    try:
        program.start_actions([ReleaseSampleAction(key)])
        before = physics.data.qpos.copy()
        program.advance()
        program.observe()

        assert program.state == ProgramState.SUCCEEDED, program.error
        assert holder.key is None
        assert program.sample.centrifuge_slot == holder.identifier
        assert program.execution.completed == 1
        np.testing.assert_array_equal(physics.data.qpos, before)
    finally:
        program.reset()


def test_standalone_balance_action_accepts_an_existing_balanced_pair(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Generic balance verification uses secured physical inventory and manual rules.
    """
    physics = workflow_world.physics
    load_pair(physics)
    assert physics.centrifuge.balanced
    program = LaboratoryWorkflowProgram(physics)
    try:
        program.start_actions([VerifyRotorBalanceAction()])
        program.advance()
        program.observe()

        assert program.state == ProgramState.SUCCEEDED, program.error
        assert program.execution.completed == 1
        assert not program.load_verified
        assert all(sample.weight_grams is None for sample in program.samples)
    finally:
        program.reset()


def test_standalone_balance_action_enforces_the_active_manual_load_requirement(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    A balanced pair is insufficient when the actual handbook requires four samples.
    """
    physics = workflow_world.physics
    load_pair(physics)
    physics.centrifuge.read_centrifuge_manual(
        CentrifugeManual.builtin(ManualEdition.SECOND).document
    )
    program = LaboratoryWorkflowProgram(physics)
    try:
        program.start_actions([VerifyRotorBalanceAction()])
        program.advance()
        program.observe()
        program.execution.worker.join(timeout=2.0)

        assert program.state == ProgramState.FAILED
        assert program.execution.plan.root.status == TaskStatus.FAILED
        assert not program.execution.worker.is_alive()
    finally:
        program.reset()


@pytest.mark.parametrize("missing_weight", [False, True])
def test_required_pair_verification_requires_its_own_weighing_evidence(
    workflow_world: LaboratoryProgramPhysics, missing_weight: bool
) -> None:
    """
    Explicit batch requirements include both physical inventory and measured weights.
    """
    physics = workflow_world.physics
    load_pair(physics)
    program = LaboratoryWorkflowProgram(physics)
    expected_keys = tuple(
        slot.key for slot in physics.centrifuge.slots.values() if slot.key is not None
    )
    for sample in program.samples:
        holder = next(
            (
                slot
                for slot in physics.centrifuge.slots.values()
                if slot.key == sample.key
            ),
            None,
        )
        if holder is not None:
            sample.centrifuge_slot = holder.identifier
            sample.weight_grams = physics.centrifuge.slot_mass(holder)
    if missing_weight:
        program.samples[0].weight_grams = None
    try:
        program.start_actions(
            [VerifyRotorBalanceAction(required_sample_keys=expected_keys)]
        )
        program.advance()
        program.observe()
        program.execution.worker.join(timeout=2.0)

        expected = ProgramState.FAILED if missing_weight else ProgramState.SUCCEEDED
        assert program.state == expected, program.error
        assert not program.load_verified
        assert not program.execution.worker.is_alive()
    finally:
        program.reset()


def test_required_full_batch_cannot_accept_only_a_balanced_pair(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    The full recipe's explicit inventory requirement cannot be satisfied by two samples.
    """
    physics = workflow_world.physics
    load_pair(physics)
    program = LaboratoryWorkflowProgram(physics)
    try:
        program.start_actions(
            [VerifyRotorBalanceAction(required_sample_keys=program.SAMPLE_KEYS)]
        )
        program.advance()
        program.observe()
        program.execution.worker.join(timeout=2.0)

        assert program.state == ProgramState.FAILED
        assert program.execution.completed == 0
        assert not program.execution.worker.is_alive()
        assert not program.load_verified
    finally:
        program.reset()
