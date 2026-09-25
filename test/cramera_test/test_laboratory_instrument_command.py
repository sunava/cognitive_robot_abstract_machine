"""
Robot instrument requests select bounded native actions without resetting samples.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing_extensions import Any

from cramera.laboratory_actions import LaboratoryInstrument, LaboratoryOperation
from cramera.laboratory_instrument_command import (
    InstrumentAction,
    InstrumentCommand,
    InstrumentField,
    InstrumentName,
    InvalidInstrumentCommand,
)
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import (
    LaboratoryProgramPhysics,
    ProgramField,
    ProgramState,
)
from cramera.laboratory_pipetting import PipettingState
from cramera.laboratory_workflow import LaboratoryWorkflowProgram

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_workflow_adapter import workflow_world


# %% strict public command schema
@pytest.mark.parametrize("instrument", list(InstrumentName))
@pytest.mark.parametrize("action", list(InstrumentAction))
def test_instrument_command_accepts_only_named_operations(
    instrument: InstrumentName, action: InstrumentAction
) -> None:
    """
    Validated fields retain the selected instrument and bounded operation.
    """
    command = InstrumentCommand.from_payload(
        {
            InstrumentField.INSTRUMENT: instrument.value,
            InstrumentField.ACTION: action.value,
        }
    )
    assert command == InstrumentCommand(instrument, action)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {InstrumentField.INSTRUMENT: InstrumentName.CENTRIFUGE},
        {InstrumentField.ACTION: InstrumentAction.OPEN},
        {
            InstrumentField.INSTRUMENT: "unknown",
            InstrumentField.ACTION: InstrumentAction.OPEN,
        },
        {
            InstrumentField.INSTRUMENT: InstrumentName.CENTRIFUGE,
            InstrumentField.ACTION: "unknown",
        },
        {
            InstrumentField.INSTRUMENT: True,
            InstrumentField.ACTION: InstrumentAction.OPEN,
        },
        {
            InstrumentField.INSTRUMENT: InstrumentName.CENTRIFUGE,
            InstrumentField.ACTION: [],
        },
        {
            InstrumentField.INSTRUMENT: InstrumentName.CENTRIFUGE,
            InstrumentField.ACTION: InstrumentAction.OPEN,
            ProgramField.PROGRESS: 1,
        },
    ],
)
def test_instrument_command_rejects_ambiguous_or_extra_fields(payload: Any) -> None:
    """
    Unknown operations and injected state changes never reach the physical adapter.
    """
    with pytest.raises(InvalidInstrumentCommand):
        InstrumentCommand.from_payload(payload)


# %% native subset admission and preservation
@pytest.mark.parametrize("instrument", list(InstrumentName))
@pytest.mark.parametrize(
    "action,operations",
    [
        (InstrumentAction.OPEN, [LaboratoryOperation.OPEN_INSTRUMENT]),
        (InstrumentAction.CLOSE, [LaboratoryOperation.CLOSE_INSTRUMENT]),
        (
            InstrumentAction.OPEN_CLOSE,
            [LaboratoryOperation.OPEN_INSTRUMENT, LaboratoryOperation.CLOSE_INSTRUMENT],
        ),
    ],
)
def test_instrument_selection_preserves_measured_samples_and_uses_native_actions(
    workflow_world: LaboratoryProgramPhysics,
    instrument: InstrumentName,
    action: InstrumentAction,
    operations: list[LaboratoryOperation],
) -> None:
    """
    Button requests retain every pose and liquid quantity while selecting only handle
    actions.
    """
    physics = workflow_world.physics
    physics.fill_liquid(LaboratoryWorkflowProgram.SAMPLE_KEYS[0], 1.25)
    before = physics.data.qpos.copy()
    velocities = physics.data.qvel.copy()
    masses = physics.model.body_mass.copy()
    contents = physics.liquid.snapshot()
    command = InstrumentCommand(instrument, action)

    workflow_world.run_instrument(command)

    program = workflow_world.program
    assert isinstance(program, LaboratoryWorkflowProgram)
    assert program.state == ProgramState.RUNNING
    assert [
        selected.request.operation for selected in program.selected_actions
    ] == operations
    assert all(
        selected.request.instrument == LaboratoryInstrument(instrument)
        for selected in program.selected_actions
    )
    np.testing.assert_array_equal(physics.data.qpos, before)
    np.testing.assert_array_equal(physics.data.qvel, velocities)
    np.testing.assert_array_equal(physics.model.body_mass, masses)
    assert physics.liquid.snapshot() == contents
    assert workflow_world.snapshot()[ProgramField.ROBOT][
        InstrumentField.INSTRUMENT
    ] == {
        InstrumentField.INSTRUMENT: instrument,
        InstrumentField.ACTION: action,
    }


def test_paused_instrument_resume_keeps_the_selected_actions(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    The ordinary resume endpoint retains the exact standalone plan and samples.
    """
    command = InstrumentCommand(
        InstrumentName.REFRIGERATOR, InstrumentAction.OPEN_CLOSE
    )
    workflow_world.run_instrument(command)
    controller = workflow_world.program
    selected = controller.selected_actions
    workflow_world.pause_program()
    before = workflow_world.physics.data.qpos.copy()
    contents = workflow_world.physics.liquid.snapshot()

    workflow_world.run_workflow(prepare=False)

    assert workflow_world.program is controller
    assert controller.selected_actions is selected
    assert controller.state == ProgramState.RUNNING
    np.testing.assert_array_equal(workflow_world.physics.data.qpos, before)
    assert workflow_world.physics.liquid.snapshot() == contents


@pytest.mark.parametrize("state", [ProgramState.RUNNING, ProgramState.PAUSED])
@pytest.mark.parametrize("native", [False, True])
def test_instrument_request_cannot_replace_owned_robot_work(
    workflow_world: LaboratoryProgramPhysics, state: ProgramState, native: bool
) -> None:
    """
    Running and paused controllers reject a new operation before changing anything.
    """
    if native:
        workflow_world.run_workflow(prepare=True)
    workflow_world.program.state = state
    controller = workflow_world.program
    before = workflow_world.physics.data.qpos.copy()
    contents = workflow_world.physics.liquid.snapshot()
    with pytest.raises(InvalidPhysicsTarget):
        workflow_world.run_instrument(
            InstrumentCommand(InstrumentName.CENTRIFUGE, InstrumentAction.OPEN)
        )
    assert workflow_world.program is controller
    assert controller.state == state
    np.testing.assert_array_equal(workflow_world.physics.data.qpos, before)
    assert workflow_world.physics.liquid.snapshot() == contents


def test_reset_removes_standalone_instrument_marker(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Reset does not leave a stale instrument task attached to the idle robot.
    """
    workflow_world.run_instrument(
        InstrumentCommand(InstrumentName.CENTRIFUGE, InstrumentAction.OPEN_CLOSE)
    )
    workflow_world.reset()
    assert (
        InstrumentField.INSTRUMENT not in workflow_world.snapshot()[ProgramField.ROBOT]
    )


@pytest.mark.parametrize("pumping", [False, True])
def test_active_instrument_rejects_handle_request_before_program_replacement(
    workflow_world: LaboratoryProgramPhysics, pumping: bool
) -> None:
    """
    A running pump or rotor must finish before its surroundings are manipulated.
    """
    physics = workflow_world.physics
    if pumping:
        physics.pipetting.state = PipettingState.DISPENSING
    else:
        physics.centrifuge.target_rpm = physics.centrifuge.parameters.stopped_rpm * 2
    controller = workflow_world.program
    before = physics.data.qpos.copy()
    contents = physics.liquid.snapshot()
    with pytest.raises(InvalidPhysicsTarget):
        workflow_world.run_instrument(
            InstrumentCommand(InstrumentName.CENTRIFUGE, InstrumentAction.OPEN)
        )
    assert workflow_world.program is controller
    np.testing.assert_array_equal(physics.data.qpos, before)
    assert physics.liquid.snapshot() == contents


@pytest.mark.parametrize(
    "state", [ProgramState.PAUSED, ProgramState.SUCCEEDED, ProgramState.FAILED]
)
def test_instrument_marker_survives_pause_and_terminal_results(
    workflow_world: LaboratoryProgramPhysics, state: ProgramState
) -> None:
    """
    The UI can identify which standalone operation paused, completed or failed.
    """
    command = InstrumentCommand(InstrumentName.REFRIGERATOR, InstrumentAction.CLOSE)
    workflow_world.run_instrument(command)
    workflow_world.program.state = state
    assert (
        workflow_world.snapshot()[ProgramField.ROBOT][InstrumentField.INSTRUMENT]
        == command.snapshot()
    )


def test_new_full_workflow_clears_previous_instrument_marker(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    A later prepared batch is presented as a full workflow rather than a handle test.
    """
    workflow_world.run_instrument(
        InstrumentCommand(InstrumentName.CENTRIFUGE, InstrumentAction.OPEN)
    )
    workflow_world.program.state = ProgramState.SUCCEEDED
    workflow_world.run_workflow(prepare=True)
    assert (
        InstrumentField.INSTRUMENT not in workflow_world.snapshot()[ProgramField.ROBOT]
    )
