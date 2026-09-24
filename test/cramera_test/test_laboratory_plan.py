"""
Native plans own the observable physical laboratory action sequence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import TaskStatus
from coraplex.plans.executables import GiskardExecutable
from coraplex.language import SequentialNode
from coraplex.plans.plan import Plan
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_physics_robot import RobotHeadJoint
from cramera.laboratory_robot_session import LaboratoryRobotSession
from cramera.laboratory_workflow import WorkflowStep
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_workflow_adapter import workflow_world


# %% native sequence ownership
def test_live_workflow_executes_a_native_plan(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    The same live program exposes running native actions and their context.
    """
    workflow_world.run_workflow(prepare=True)
    program = workflow_world.program
    try:
        before = workflow_world.physics.data.qpos.copy()
        program.advance()
        np.testing.assert_array_equal(workflow_world.physics.data.qpos, before)
        assert isinstance(program.execution.plan, Plan)
        assert isinstance(program.execution.plan.context, Context)
        assert isinstance(program.execution.plan.root, SequentialNode)
        assert program.execution.plan.root.status == TaskStatus.RUNNING
        assert program.execution.active_request.object_key == program.samples[0].key
        assert program.step.operation == WorkflowStep.DISPENSER
    finally:
        program.reset()


def test_reset_cancels_the_pending_native_action(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Reset joins the waiting plan before abandoning its physical ownership.
    """
    workflow_world.run_workflow(prepare=True)
    program = workflow_world.program
    program.advance()
    execution = program.execution
    program.reset()
    assert not execution.worker.is_alive()
    assert execution.active_request is None
    assert program.state == ProgramState.IDLE
    assert program.execution is None


def test_pause_keeps_the_same_native_action_pending(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Pausing neither completes nor reconstructs the active action.
    """
    workflow_world.run_workflow(prepare=True)
    program = workflow_world.program
    try:
        program.advance()
        execution = program.execution
        request = execution.active_request
        program.pause()
        workflow_world.step(10)
        assert execution.active_request is request
        program.start()
        program.advance()
        assert program.execution is execution
        assert execution.active_request is request
    finally:
        program.reset()


def test_physical_failure_stops_the_native_plan(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    A measured failure prevents the plan from requesting the following dose.
    """
    workflow_world.run_workflow(prepare=True)
    program = workflow_world.program
    try:
        program.advance()
        execution = program.execution
        program._fail("Measured grasp was lost")
        execution.worker.join(timeout=2.0)
        assert not execution.worker.is_alive()
        assert execution.plan.root.status == TaskStatus.FAILED
        assert execution.completed == 0
        assert execution.active_request is None
        assert program.state == ProgramState.FAILED
    finally:
        program.reset()


def test_failed_action_records_the_measured_failure_configuration(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Retain actual failure-time articulation in the context and failed action record.
    """
    workflow_world.run_workflow(prepare=True)
    program = workflow_world.program
    try:
        program.advance()
        execution = program.execution
        workflow_world.physics.set_joint_targets({RobotHeadJoint.PAN: 0.3})
        workflow_world.physics.step(500)
        measured = workflow_world.physics.robot_state()[RobotHeadJoint.PAN]

        program._fail("Measured grasp was lost after the robot moved")
        execution.worker.join(timeout=2.0)

        assert not execution.worker.is_alive()
        prefix, local_name = RobotHeadJoint.PAN.split("/", 1)
        connection = execution.plan.world.get_connection_by_name(
            PrefixedName(local_name, prefix)
        )
        assert connection.position == measured
        [failed_action] = [
            action
            for action in execution.plan.actions
            if action.status == TaskStatus.FAILED
        ]
        recorded = failed_action.execution_data.execution_end_world_state
        np.testing.assert_array_equal(recorded, execution.plan.world.state._data)
        assert not np.shares_memory(recorded, execution.plan.world.state._data)
    finally:
        program.reset()


def test_stopping_the_session_joins_its_native_plan_without_resetting_physics(
    workflow_world: LaboratoryProgramPhysics,
    tmp_path: Path,
) -> None:
    """
    A stopped laboratory leaves no blocked action worker or active simulation scope.
    """
    previous_execution = GiskardExecutable.execution_type
    previous_avoidance = GiskardExecutable.collision_avoidance
    workflow_world.run_workflow(prepare=True)
    program = workflow_world.program
    try:
        program.advance()
        execution = program.execution
        before = workflow_world.physics.data.qpos.copy()
        session = LaboratoryRobotSession(data_directory=tmp_path)
        session.physics = workflow_world

        session._stop_physics()

        assert not execution.worker.is_alive()
        assert execution.active_request is None
        assert GiskardExecutable.execution_type == previous_execution
        assert GiskardExecutable.collision_avoidance == previous_avoidance
        np.testing.assert_array_equal(workflow_world.physics.data.qpos, before)
    finally:
        program.reset()
