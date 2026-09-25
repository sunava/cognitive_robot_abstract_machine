"""
Swept base safety includes every independent body carried in a sample rack.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from cramera.laboratory_physics_robot import LaboratoryRobotPhysics, RobotBaseJoint
from cramera.laboratory_rack import RackGeometry
from cramera.laboratory_robot_base import LaboratoryBasePath, UnsafeBasePath
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% independent carried bodies
@pytest.fixture
def carried_rack(mobile_robot: LaboratoryRobotPhysics) -> LaboratoryRobotPhysics:
    """
    Place separate contacting samples and their carrier in an unobstructed corridor.
    """
    physics = mobile_robot
    position = np.array([1.4, -1.7, 1.2])
    rack = physics.model.body(RackGeometry.BODY).id
    shift = position - physics.data.xpos[rack]
    for name in (RackGeometry.BODY, *physics.liquid.tubes):
        body = physics.model.body(name).id
        coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[body]]
        physics.data.qpos[coordinate : coordinate + 3] += shift
    physics.data.qpos[physics.robot_joints[RobotBaseJoint.DEPTH].position_index] = -0.5
    mujoco.mj_forward(physics.model, physics.data)
    return physics


def test_carried_sample_above_rack_blocks_the_swept_group_path(
    carried_rack: LaboratoryRobotPhysics,
) -> None:
    """
    An obstacle touching only a sample still blocks the loaded carrier's travel.
    """
    physics = carried_rack
    sample = physics.model.body(LaboratoryBody.CLEAR_TUBE).id
    stopper = physics.model.body(LaboratoryBody.STOPPER).id
    coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[stopper]]
    physics.data.qpos[coordinate : coordinate + 3] = physics.data.xpos[sample] + [
        -0.2,
        0.0,
        0.11,
    ]
    mujoco.mj_forward(physics.model, physics.data)
    target = {RobotBaseJoint.HORIZONTAL: -0.4}
    LaboratoryBasePath(physics).validate(target)
    before = physics.data.qpos.copy()
    path = LaboratoryBasePath(
        physics, held_objects=(RackGeometry.BODY, *physics.liquid.tubes)
    )
    with pytest.raises(UnsafeBasePath) as obstruction:
        path.validate(target)
    assert obstruction.value.robot_geometry.startswith(LaboratoryBody.CLEAR_TUBE)
    assert obstruction.value.obstacle_geometry.startswith(LaboratoryBody.STOPPER)
    np.testing.assert_array_equal(physics.data.qpos, before)


def test_carried_group_contacts_do_not_block_clear_travel(
    carried_rack: LaboratoryRobotPhysics,
) -> None:
    """
    Rack-to-sample support contacts remain internal to a safely swept assembly.
    """
    physics = carried_rack
    carried = (RackGeometry.BODY, *physics.liquid.tubes)
    before = physics.data.qpos.copy()
    path = LaboratoryBasePath(physics, held_objects=carried)
    path.validate({RobotBaseJoint.HORIZONTAL: -0.3, RobotBaseJoint.DEPTH: -0.7})
    for name in carried:
        body = physics.model.body(name).id
        np.testing.assert_allclose(
            path.scratch.xpos[body] - physics.data.xpos[body], [-0.3, -0.2, 0.0]
        )
    np.testing.assert_array_equal(physics.data.qpos, before)
