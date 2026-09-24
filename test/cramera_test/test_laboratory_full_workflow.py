"""
Exercise the complete four-sample recipe with authoritative contact dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_workflow import LaboratoryWorkflowProgram, WorkflowLocation

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_workflow_adapter import workflow_world


# %% physical recipe acceptance
@dataclass(frozen=True)
class WorkflowAcceptance:
    """
    Bound the complete physical recipe and final closed-door observation.
    """

    maximum_seconds: float = 2400.0
    """
    Simulation-time ceiling covering all instrument and manipulation waits.
    """

    settling_seconds: float = 5.0
    """
    Additional closed-door time used to detect an unstable final placement.
    """


def test_four_samples_are_dosed_weighed_spun_and_stored(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Every completed sample is physically carried through all four instruments.
    """
    acceptance = WorkflowAcceptance()
    workflow_world.run_workflow(prepare=True)
    program = workflow_world.program
    assert isinstance(program, LaboratoryWorkflowProgram)
    physics = workflow_world.physics
    previous_step = None
    for _ in range(int(acceptance.maximum_seconds / physics.timestep)):
        if program.step_index != previous_step:
            print(
                f"{physics.data.time:.3f}s: {program.step.operation} "
                f"[{program.sample.key if program.sample else 'instruments'}]",
                flush=True,
            )
            previous_step = program.step_index
        before_intention = physics.data.qpos.copy()
        program.advance()
        assert np.array_equal(physics.data.qpos, before_intention)
        physics.step()
        before_observation = physics.data.qpos.copy()
        program.observe()
        assert np.array_equal(physics.data.qpos, before_observation)
        if program.state in (ProgramState.SUCCEEDED, ProgramState.FAILED):
            break
    assert program.state == ProgramState.SUCCEEDED, program.snapshot()
    assert program.load_verified
    assert program.spin_finished
    workflow_world.step(int(acceptance.settling_seconds / physics.timestep))
    assert physics.centrifuge.stopped
    assert physics.centrifuge.lid_closed
    assert physics.refrigerator.door_closed
    assert all(slot.key is None for slot in physics.centrifuge.slots.values())
    occupants = physics.refrigerator.occupants()
    for sample in program.samples:
        assert sample.location == WorkflowLocation.REFRIGERATOR
        assert occupants[sample.refrigerator_slot] == sample.key
        assert sample.dose_a_ml == pytest.approx(program.parameters.dose_ml)
        assert sample.dose_b_ml == pytest.approx(program.parameters.dose_ml)
        assert physics.liquid.tubes[sample.key].volume_ml == pytest.approx(
            sample.dose_a_ml + sample.dose_b_ml,
            abs=program.parameters.volume_tolerance_ml,
        )
        body = physics.objects[sample.key].body_id
        assert sample.weight_grams == pytest.approx(
            physics.model.body_mass[body] * 1000,
            abs=program.parameters.mass_tolerance_grams,
        )
