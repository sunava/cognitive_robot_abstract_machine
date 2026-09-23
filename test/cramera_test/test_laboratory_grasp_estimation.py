"""
A cylindrical tube's axial spin does not redefine the hand approach.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cramera.laboratory_mixing_program import LaboratoryMixingProgram, MixingPhase
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics


# %% cylindrical grasp orientation
def test_axial_tube_rotation_does_not_turn_estimated_front_grasp(
    robot_physics: LaboratoryRobotPhysics,
) -> None:
    """
    Relocalization tracks the glass axis without enforcing its arbitrary yaw.
    """
    program = LaboratoryMixingProgram(robot_physics)
    program.stage_index = next(
        index
        for index, stage in enumerate(program.stages)
        if stage.phase == MixingPhase.LIFT
    )
    program.stage_time = program.stage.duration
    program.grasp_verified = program.holding = True
    tool_id = robot_physics.model.body(program.TOOL_FRAME).id
    tool_rotation = Rotation.from_matrix(robot_physics.data.xmat[tool_id].reshape(3, 3))
    position, rotation = program._pose()
    program.grasp_rotation = tool_rotation.inv() * rotation
    program.grasp_position = tool_rotation.inv().apply(
        position - robot_physics.data.xpos[tool_id]
    )
    before = program.grasp_rotation.as_matrix()
    body_id = robot_physics.objects[program.stage.object_key].body_id
    joint_id = robot_physics.model.body_jntadr[body_id]
    position_index = robot_physics.model.jnt_qposadr[joint_id]
    rotated = rotation * Rotation.from_euler("z", 0.5)
    robot_physics.data.qpos[position_index + 3 : position_index + 7] = rotated.as_quat(
        scalar_first=True
    )
    mujoco.mj_forward(robot_physics.model, robot_physics.data)
    program._estimate_grasp(0.1)
    np.testing.assert_allclose(program.grasp_rotation.as_matrix(), before, atol=1e-12)


def test_lift_capture_preserves_front_heading_when_cylinder_spins(
    robot_physics: LaboratoryRobotPhysics,
) -> None:
    """
    The same tube axis produces the same calibrated hand orientation.
    """
    program = LaboratoryMixingProgram(robot_physics)
    program.stage_index = next(
        index
        for index, stage in enumerate(program.stages)
        if stage.phase == MixingPhase.LIFT
    )
    program.grasp_verified = program.holding = True
    program._enter_stage()
    before = program.grasp_rotation.as_matrix()
    body_id = robot_physics.objects[program.stage.object_key].body_id
    joint_id = robot_physics.model.body_jntadr[body_id]
    position_index = robot_physics.model.jnt_qposadr[joint_id]
    rotation = program._pose()[1] * Rotation.from_euler("z", 0.5)
    robot_physics.data.qpos[position_index + 3 : position_index + 7] = rotation.as_quat(
        scalar_first=True
    )
    mujoco.mj_forward(robot_physics.model, robot_physics.data)
    program._enter_stage()
    np.testing.assert_allclose(program.grasp_rotation.as_matrix(), before, atol=1e-12)
