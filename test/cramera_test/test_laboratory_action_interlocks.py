"""
Instrument interlocks terminate native actions while physics remains live.
"""

from __future__ import annotations

import numpy as np

from coraplex.datastructures.enums import TaskStatus
from cramera.laboratory_actions import (
    CloseInstrumentAction,
    LaboratoryInstrument,
    OpenInstrumentAction,
    RunCentrifugeAction,
)
from cramera.laboratory_centrifuge_command import CentrifugeAction
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_workflow import LaboratoryWorkflowProgram

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_centrifuge import load_pair
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_workflow_adapter import workflow_world


# %% measured action failures
def test_rotating_lid_interlock_fails_the_native_action_without_stopping_physics(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    An unsafe opening request fails its action and prevents the next action.
    """
    physics = workflow_world.physics
    load_pair(physics)
    physics.centrifuge_command(CentrifugeAction.CLOSE)
    physics.step(5000)
    physics.centrifuge_command(CentrifugeAction.START, rpm=120)
    physics.step(500)
    assert not physics.centrifuge.stopped
    before = physics.data.qpos.copy()
    program = LaboratoryWorkflowProgram(physics)
    try:
        program.start_actions(
            [
                OpenInstrumentAction(LaboratoryInstrument.CENTRIFUGE),
                CloseInstrumentAction(LaboratoryInstrument.REFRIGERATOR),
            ]
        )
        program.advance()
        execution = program.execution
        execution.worker.join(timeout=2.0)

        assert program.state == ProgramState.FAILED
        assert execution.plan.root.status == TaskStatus.FAILED
        assert execution.plan.actions[0].status == TaskStatus.FAILED
        assert execution.completed == 0
        assert not execution.worker.is_alive()
        assert execution.active_request is None
        assert physics.centrifuge.target_rpm == 0
        np.testing.assert_array_equal(physics.data.qpos, before)
        before_time = physics.data.time
        physics.step()
        assert physics.data.time > before_time
    finally:
        program.reset()


def test_rejected_cycle_command_fails_the_native_action_during_observation(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    An invalid native cycle request cannot escape the physical observer thread.
    """
    physics = workflow_world.physics
    load_pair(physics)
    physics.centrifuge_command(CentrifugeAction.CLOSE)
    physics.step(5000)
    program = LaboratoryWorkflowProgram(physics)
    try:
        program.start_actions([RunCentrifugeAction(rpm=-1, duration=1)])
        program.advance()
        program.observe()
        execution = program.execution
        execution.worker.join(timeout=2.0)

        assert program.state == ProgramState.FAILED
        assert execution.plan.root.status == TaskStatus.FAILED
        assert execution.plan.actions[0].status == TaskStatus.FAILED
        assert execution.completed == 0
        assert not execution.worker.is_alive()
        assert physics.centrifuge.stopped
        before_time = physics.data.time
        physics.step()
        assert physics.data.time > before_time
    finally:
        program.reset()
