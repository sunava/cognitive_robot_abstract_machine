"""
A close pour reaches its tilt before descending beside the receiver.
"""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cramera.laboratory_mixing_program import LaboratoryMixingProgram, MixingPhase

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics


# %% sequential alignment
def test_pour_alignment_rotates_before_lowering_lip(robot_physics) -> None:
    """
    The lower pouring target is approached with an already aligned glass axis.
    """
    program = LaboratoryMixingProgram(robot_physics)
    program.stage_index = next(
        index
        for index, stage in enumerate(program.stages)
        if stage.phase == MixingPhase.POUR
        and Rotation.from_quat(stage.rotation).apply([0, 0, 1])[2] < 0.2
    )
    previous = program.stages[program.stage_index - 1]
    program.start_position = np.asarray(previous.position)
    program.start_rotation = Rotation.from_quat(previous.rotation)
    interior = robot_physics.liquid.interior
    start_lip = program.start_position + program.start_rotation.apply(
        interior.lowest_lip(program.start_rotation.inv().apply([0, 0, 1]))
    )
    position, rotation = program._pour_approach(0.5)
    middle_lip = position + rotation.apply(
        interior.lowest_lip(rotation.inv().apply([0, 0, 1]))
    )
    assert middle_lip[2] == pytest.approx(start_lip[2])
    np.testing.assert_allclose(
        rotation.as_matrix(), Rotation.from_quat(program.stage.rotation).as_matrix()
    )
    position, rotation = program._pour_approach(1.0)
    np.testing.assert_allclose(position, program._stage_position())


def test_descent_progress_waits_for_measured_axis_and_pauses_on_new_error(
    robot_physics,
) -> None:
    """
    Elapsed recipe time alone cannot lower a glass whose tilt is still lagging.
    """
    program = LaboratoryMixingProgram(robot_physics)
    program.stage_index = next(
        index
        for index, stage in enumerate(program.stages)
        if stage.phase == MixingPhase.POUR
        and Rotation.from_quat(stage.rotation).apply([0, 0, 1])[2] < 0.2
    )
    previous = program.stages[program.stage_index - 1]
    program.start_position = np.asarray(previous.position)
    program.start_rotation = Rotation.from_quat(previous.rotation)
    duration = program.stage.duration / 4
    program.grasp_verified = program.holding = True
    assert program._pour_progress(1.0, duration) == 0.5
    body_id = robot_physics.objects[program.stage.object_key].body_id
    joint_id = robot_physics.model.body_jntadr[body_id]
    position_index = robot_physics.model.jnt_qposadr[joint_id]
    robot_physics.data.qpos[position_index + 2] += 2 * program.parameters.lift_height
    original = robot_physics.data.qpos[position_index + 3 : position_index + 7].copy()
    robot_physics.data.qpos[position_index + 3 : position_index + 7] = (
        Rotation.from_quat(program.stage.rotation).as_quat(scalar_first=True)
    )
    mujoco.mj_forward(robot_physics.model, robot_physics.data)
    assert program._pour_progress(1.0, duration) == 0.75
    robot_physics.data.qpos[position_index + 3 : position_index + 7] = original
    mujoco.mj_forward(robot_physics.model, robot_physics.data)
    assert program._pour_progress(1.0, duration) == 0.75
