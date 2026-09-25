"""Standalone instrument commands execute native actions through robot contacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from coraplex.datastructures.enums import TaskStatus
from cramera.laboratory_actions import (
    CloseInstrumentAction,
    LaboratoryContext,
    OpenInstrumentAction,
)
from cramera.laboratory_hinge_manipulation import LaboratoryHingeManipulation
from cramera.laboratory_instrument_command import (
    InstrumentAction,
    InstrumentCommand,
    InstrumentName,
)
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_workflow import LaboratoryWorkflowProgram
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_hinge_manipulation import assert_clear_environment_contacts
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_workflow_adapter import workflow_world


# %% bounded physical command acceptance
@dataclass(frozen=True)
class InstrumentExecutionAcceptance:
    """Physical observation bounds for independently selected handle actions."""

    action_seconds: float = 180.0
    """Maximum simulated duration of one complete grasp, hinge move and withdrawal."""
    settling_seconds: float = 1.0
    """Physical settling time before testing preservation of the existing scene."""
    sample_volume_ml: float = 3.2
    """Distinct loaded contents making an unintended refill or reset observable."""


@pytest.mark.parametrize("instrument", list(InstrumentName))
def test_selected_instrument_executes_contact_driven_native_actions(
    workflow_world: LaboratoryProgramPhysics, instrument: InstrumentName
) -> None:
    """Independent native plans operate each cover while preserving the laboratory."""
    physics = workflow_world.physics
    acceptance = InstrumentExecutionAcceptance()
    physics.fill_liquid(LaboratoryBody.CLEAR_TUBE, acceptance.sample_volume_ml)
    physics.step(int(acceptance.settling_seconds / physics.timestep))
    contents = {key: tube.volume_ml for key, tube in physics.liquid.tubes.items()}
    reservoirs = {
        key: reservoir.volume_ml
        for key, reservoir in physics.pipetting.reservoirs.items()
    }
    requests = []
    if instrument == InstrumentName.CENTRIFUGE:
        requests.append((InstrumentAction.CLOSE, (False,)))
    requests.append((InstrumentAction.OPEN_CLOSE, (True, False)))
    try:
        for action, directions in requests:
            before = physics.data.qpos.copy()
            before_time = physics.data.time
            workflow_world.run_instrument(InstrumentCommand(instrument, action))
            np.testing.assert_array_equal(physics.data.qpos, before)
            assert physics.data.time == before_time
            assert {
                key: tube.volume_ml for key, tube in physics.liquid.tubes.items()
            } == contents
            program = workflow_world.program
            assert isinstance(program, LaboratoryWorkflowProgram)
            completed = []
            previous_hinge = None
            for _ in range(
                int(acceptance.action_seconds * len(directions) / physics.timestep)
            ):
                before = physics.data.qpos.copy()
                program.advance()
                assert np.array_equal(physics.data.qpos, before)
                hinge = program.hinge
                physics.step()
                if hinge is not None:
                    assert_clear_environment_contacts(physics, hinge)
                before = physics.data.qpos.copy()
                program.observe()
                assert np.array_equal(physics.data.qpos, before)
                if (
                    hinge is not None
                    and hinge.state == ProgramState.SUCCEEDED
                    and hinge is not previous_hinge
                ):
                    assert_physical_completion(hinge)
                    completed.append(hinge.opening)
                    previous_hinge = hinge
                    print(
                        f"{instrument}: {'open' if hinge.opening else 'close'} "
                        f"completed at {physics.data.time:.3f}s",
                        flush=True,
                    )
                if program.state != ProgramState.RUNNING:
                    break
            assert program.state == ProgramState.SUCCEEDED, program.snapshot()
            assert tuple(completed) == directions
            assert program.execution.completed == len(directions)
            assert len(program.execution.plan.actions) == len(directions)
            assert [type(item.action) for item in program.execution.plan.actions] == [
                OpenInstrumentAction if opening else CloseInstrumentAction
                for opening in directions
            ]
            assert isinstance(program.execution.plan.context, LaboratoryContext)
            assert program.execution.plan.root.status == TaskStatus.SUCCEEDED
            assert {
                key: tube.volume_ml for key, tube in physics.liquid.tubes.items()
            } == contents
            assert {
                key: reservoir.volume_ml
                for key, reservoir in physics.pipetting.reservoirs.items()
            } == reservoirs
            assert not np.any(physics.data.xfrc_applied)
        if instrument == InstrumentName.CENTRIFUGE:
            assert physics.centrifuge.lid_closed
        else:
            assert physics.refrigerator.door_closed
    finally:
        workflow_world.stop_program()


def assert_physical_completion(hinge: LaboratoryHingeManipulation) -> None:
    """Require actual handle-driven travel, endpoint alignment and hand release."""
    assert hinge.moved
    assert hinge.grasp_verified
    assert hinge.maximum_handle_force > hinge.parameters.contact_force
    assert hinge.maximum_motor_force == 0
    assert hinge.angle == pytest.approx(
        hinge.target_angle, abs=hinge.parameters.angle_tolerance
    )
    assert hinge.handle_force < hinge.parameters.contact_force
