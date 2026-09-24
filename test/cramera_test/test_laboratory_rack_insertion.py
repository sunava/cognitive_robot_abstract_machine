"""
A measured upper grip remains stable while seating through a narrow rack hole.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cramera.laboratory_gaze import LaboratoryGaze
from cramera.laboratory_mixing_program import MixingPhase
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer, SampleGrasp
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_centrifuge import seat_tube
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% measured insertion setup
@dataclass(frozen=True)
class InsertionJoint:
    """
    Measured scalar joint state at the beginning of rack descent.
    """

    position: float
    """
    Joint coordinate observed after transport from the centrifuge.
    """

    velocity: float
    """
    Measured coordinate velocity at the same instant.
    """

    target: float | None
    """
    Active position demand, absent for passive mimic coordinates.
    """


@dataclass(frozen=True)
class RackInsertionState:
    """
    Compact physical sample-and-hand state recorded above the destination.
    """

    joints: dict[str, dict[str, float | None]]
    """
    Named joint states, independent of the generated MuJoCo index order.
    """

    sample_pose: list[float]
    """
    Measured free-body coordinates of the held vessel.
    """

    sample_velocity: list[float]
    """
    Measured linear and angular velocity of the vessel.
    """

    grasp_position: list[float]
    """
    Estimated vessel-root position in the tool frame.
    """

    grasp_rotation: list[float]
    """
    Estimated vessel orientation in the tool frame, in XYZW order.
    """

    def apply(self, physics: LaboratoryRobotPhysics, key: str) -> None:
        """
        Initialize a fresh fixture to a measured, force-held insertion pose.
        """
        for name, values in self.joints.items():
            state = InsertionJoint(**values)
            joint = physics.model.joint(name).id
            physics.data.qpos[physics.model.jnt_qposadr[joint]] = state.position
            physics.data.qvel[physics.model.jnt_dofadr[joint]] = state.velocity
            if state.target is not None:
                physics.set_joint_targets({name: state.target})
        body = physics.objects[key].body_id
        joint = physics.model.body_jntadr[body]
        coordinate = physics.model.jnt_qposadr[joint]
        velocity = physics.model.jnt_dofadr[joint]
        physics.data.qpos[coordinate : coordinate + 7] = self.sample_pose
        physics.data.qvel[velocity : velocity + 6] = self.sample_velocity
        mujoco.mj_forward(physics.model, physics.data)


# %% physical supported insertion
def test_upper_grip_seats_sample_without_catching_the_rack(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Real actuator motion retains the filled sample until the rack supports it.
    """
    physics = mobile_robot
    key = LaboratoryBody.TEAL_TUBE
    seat_tube(physics, LaboratoryBody.CLEAR_TUBE, "C1")
    physics.fill_liquid(key, 2.0)
    fixture = Path(__file__).parent / "dataset/laboratory_rack_insertion.json"
    measured = RackInsertionState(**json.loads(fixture.read_text()))
    measured.apply(physics, key)
    rack = next(
        slot["pose"][:3]
        for slot in physics.scene["laboratory"]["rackSlots"]
        if slot["id"] == "A1"
    )
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=key,
        destination=tuple(rack),
        support_body=LaboratoryBody.RACK,
        grasp=SampleGrasp.TOP,
    )
    transfer.stage_index = next(
        index
        for index, stage in enumerate(transfer.stages[:-1])
        if stage.phase == MixingPhase.PLACE
        and transfer.stages[index + 1].phase == MixingPhase.RELEASE
    )
    transfer.grasp_position = np.asarray(measured.grasp_position)
    transfer.grasp_rotation = Rotation.from_quat(measured.grasp_rotation)
    transfer.grasp_verified = True
    transfer.grasped_objects.add(key)
    transfer._enter_stage()
    transfer.start()
    gaze = LaboratoryGaze(physics)
    for _ in range(int(50 / physics.timestep)):
        before = physics.data.qpos.copy()
        transfer.advance()
        gaze.advance()
        np.testing.assert_array_equal(physics.data.qpos, before)
        physics.step()
        transfer.observe()
        if transfer.state != ProgramState.RUNNING:
            break
    assert transfer.state == ProgramState.SUCCEEDED, transfer.snapshot()
    assert transfer.released
    assert transfer._supported_destination(
        physics.objects[key].body_id, transfer.parameters.destination_tolerance
    )
    assert physics.liquid.tubes[key].volume_ml == pytest.approx(2.0)
