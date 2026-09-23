"""
The pouring demonstration reaches its glass with a horizontal front grasp.
"""

import numpy as np

from cramera.laboratory_mixing_program import LaboratoryMixingProgram, MixingPhase
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics


# %% measured front approach
def test_grasp_reaches_the_upright_glass_from_the_front(
    robot_physics: LaboratoryRobotPhysics,
) -> None:
    """
    The actual palm approaches horizontally with fingers on opposite sides.
    """
    program = LaboratoryMixingProgram(robot_physics)
    program.start()
    for _ in range(50000):
        program.advance()
        robot_physics.step()
        program.observe()
        if program.stage.phase == MixingPhase.GRASP:
            break
        if program.state != ProgramState.RUNNING:
            break

    assert program.stage.phase == MixingPhase.GRASP, program.snapshot()
    tool = robot_physics.model.body(program.TOOL_FRAME).id
    orientation = robot_physics.data.xmat[tool].reshape(3, 3)
    angular_tolerance = np.sin(np.deg2rad(10))
    assert abs(orientation[2, 0]) < angular_tolerance
    assert abs(orientation[2, 1]) < angular_tolerance


def test_pouring_keeps_the_upper_arm_clear_of_the_bench(
    robot_physics: LaboratoryRobotPhysics,
) -> None:
    """
    The pouring controller preserves an arm posture above the laboratory bench.
    """
    program = LaboratoryMixingProgram(robot_physics)
    program.start()
    upper_arm = robot_physics.model.body("pr2/l_upper_arm_link").id
    room = robot_physics.model.body("laboratory_room").id
    for _ in range(80000):
        program.advance()
        robot_physics.step()
        program.observe()
        for index in range(robot_physics.data.ncon):
            contact = robot_physics.data.contact[index]
            bodies = {
                int(robot_physics.model.geom_bodyid[contact.geom1]),
                int(robot_physics.model.geom_bodyid[contact.geom2]),
            }
            assert bodies != {upper_arm, room}, program.snapshot()
        if program.stage.phase == MixingPhase.RETURN:
            break
        if program.state != ProgramState.RUNNING:
            break
    assert program.stage.phase == MixingPhase.RETURN, program.snapshot()
