"""
Physical shelf transport after extracting a sample from the rotor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from cramera.laboratory_physics_robot import LaboratoryRobotPhysics, RobotBaseJoint
from cramera.laboratory_refrigerator_command import RefrigeratorField
from cramera.laboratory_robot_base import LaboratoryBasePath
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% measured carry geometry
@dataclass(frozen=True)
class CabinetCarryPosture:
    """
    Measured front-grasp configuration after a rotor-to-rack handoff.
    """

    joint_positions: dict[str, float]
    """
    Observed arm, torso and finger coordinates before lateral travel.
    """

    sample_pose: list[float]
    """
    Observed free-body pose of the held vessel in MuJoCo coordinate order.
    """

    def apply(self, physics: LaboratoryRobotPhysics, key: str) -> None:
        """
        Initialize collision geometry to the recorded carrying configuration.
        """
        for name, position in self.joint_positions.items():
            physics.data.qpos[physics.robot_joints[name].position_index] = position
        body = physics.objects[key].body_id
        coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[body]]
        physics.data.qpos[coordinate : coordinate + 7] = self.sample_pose
        mujoco.mj_forward(physics.model, physics.data)


def test_cabinet_route_clears_measured_upper_arm_posture(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Generated cabinet travel clears the chamber for the observed handoff posture.
    """
    physics = mobile_robot
    key = LaboratoryBody.CLEAR_TUBE
    slot = physics.refrigerator.metadata[RefrigeratorField.SLOTS][1]
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=key,
        destination=tuple(slot[RefrigeratorField.POSITION]),
        support_body=physics.refrigerator.metadata[RefrigeratorField.BODY],
    )
    fixture = Path(__file__).parent / "dataset/laboratory_cabinet_carry_posture.json"
    CabinetCarryPosture(**json.loads(fixture.read_text())).apply(physics, key)
    dock = transfer._horizontal_base_target(np.asarray(transfer.destination))
    reached_dock = False
    for stage in transfer.stages:
        if not stage.closed or stage.base_targets is None:
            continue
        path = LaboratoryBasePath(physics, key)
        assert path.obstruction(stage.base_targets) is None
        state = physics.robot_state()
        displacement = np.array(
            [
                stage.base_targets[RobotBaseJoint.HORIZONTAL]
                - state[RobotBaseJoint.HORIZONTAL],
                stage.base_targets[RobotBaseJoint.DEPTH] - state[RobotBaseJoint.DEPTH],
                0.0,
            ]
        )
        for name, target in stage.base_targets.items():
            physics.data.qpos[physics.robot_joints[name].position_index] = target
        body = physics.objects[key].body_id
        coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[body]]
        physics.data.qpos[coordinate : coordinate + 3] += displacement
        mujoco.mj_forward(physics.model, physics.data)
        if stage.base_targets[RobotBaseJoint.HORIZONTAL] == dock:
            reached_dock = True
            break
    assert reached_dock
