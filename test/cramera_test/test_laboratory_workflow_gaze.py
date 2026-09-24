"""
The running laboratory recipe keeps the physical camera on its working gripper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_workflow import WorkflowField, WorkflowStep

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_workflow_adapter import workflow_world


# %% measured camera alignment
@dataclass(frozen=True)
class GazeObservation:
    """
    Measure the optical-axis error against the actual manipulation tool.
    """

    physics: LaboratoryRobotPhysics
    """
    Contact simulation containing measured camera and tool frames.
    """

    camera: str = "pr2/wide_stereo_optical_frame"
    """
    PR2 camera whose positive optical Z axis points forward.
    """

    tool: str = "pr2/l_gripper_tool_frame"
    """
    Tool used by the four-sample workflow.
    """

    settling_seconds: float = 8.0
    """
    Initial moving-arm interval allowing bounded head acceleration.
    """

    maximum_error: float = math.radians(5.0)
    """
    Accepted measured optical-axis error while following the moving gripper.
    """

    def angle(self) -> float:
        """
        Return the measured angle between camera gaze and the actual tool.
        """
        camera = self.physics.model.body(self.camera).id
        tool = self.physics.model.body(self.tool).id
        direction = self.physics.data.xpos[tool] - self.physics.data.xpos[camera]
        direction /= np.linalg.norm(direction)
        optical_axis = self.physics.data.xmat[camera].reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(optical_axis @ direction, -1.0, 1.0)))


def test_workflow_publishes_gaze_only_after_tracking_begins(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    The public workflow status distinguishes inactive gaze from measured tracking.
    """
    workflow_world.run_workflow(prepare=True)
    program = workflow_world.program
    assert program.snapshot()[WorkflowField.WORKFLOW][WorkflowField.GAZE] is None
    program.advance()
    assert (
        program.snapshot()[WorkflowField.WORKFLOW][WorkflowField.GAZE]
        == program.gaze.snapshot()
    )


def test_workflow_tracks_the_actual_gripper_without_pose_writes(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Starting the recipe produces physical head tracking beside arm motion.
    """
    workflow_world.run_workflow(prepare=True)
    observation = GazeObservation(workflow_world.physics)
    for _ in range(int(observation.settling_seconds / workflow_world.physics.timestep)):
        before = workflow_world.physics.data.qpos.copy()
        workflow_world.program.advance()
        np.testing.assert_array_equal(workflow_world.physics.data.qpos, before)
        workflow_world.physics.step()
        workflow_world.program.observe()
    assert workflow_world.program.state == ProgramState.RUNNING
    assert observation.angle() < observation.maximum_error


def test_pause_preserves_head_targets_and_resume_reuses_gaze(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    A paused recipe retains its gaze controller without issuing head motion.
    """
    workflow_world.run_workflow(prepare=True)
    workflow_world.step(100)
    program = workflow_world.program
    controller = program.gaze
    assert controller is not None
    program.pause()
    targets = workflow_world.physics.joint_targets.copy()
    workflow_world.step(100)
    assert workflow_world.physics.joint_targets == targets
    program.start()
    assert program.gaze is controller
    assert program.state == ProgramState.RUNNING
    program.advance()
    assert program.gaze is controller
    program.reset()
    assert program.gaze is None


def test_gaze_follows_through_real_dosing_and_supported_weighing(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Continuous gaze preserves a complete contact-driven fill and weigh sequence.
    """
    workflow_world.run_workflow(prepare=True)
    program = workflow_world.program
    observation = GazeObservation(workflow_world.physics)
    observed = set()
    instrument_steps = {WorkflowStep.DOSE_A, WorkflowStep.DOSE_B, WorkflowStep.WEIGH}
    allowed_seconds = (
        2 * program.parameters.transfer_timeout
        + len(instrument_steps) * program.parameters.operation_timeout
    )
    for _ in range(int(allowed_seconds / workflow_world.physics.timestep)):
        if program.step.operation in instrument_steps:
            assert observation.angle() < observation.maximum_error
            observed.add(program.step.operation)
        workflow_world.step()
        if (
            program.state != ProgramState.RUNNING
            or program.samples[0].weight_grams is not None
        ):
            break
    assert program.state == ProgramState.RUNNING, program.snapshot()
    assert observed == instrument_steps
    sample = program.samples[0]
    assert sample.dose_a_ml == pytest.approx(program.parameters.dose_ml)
    assert sample.dose_b_ml == pytest.approx(program.parameters.dose_ml)
    assert workflow_world.physics.liquid.tubes[sample.key].volume_ml == pytest.approx(
        sample.dose_a_ml + sample.dose_b_ml,
        abs=program.parameters.volume_tolerance_ml,
    )
    body = workflow_world.physics.objects[sample.key].body_id
    assert sample.weight_grams == pytest.approx(
        workflow_world.physics.model.body_mass[body] * 1000,
        abs=program.parameters.mass_tolerance_grams,
    )
