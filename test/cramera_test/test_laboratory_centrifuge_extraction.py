"""A processed sample leaves the rotor before returning to its supported rack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import mujoco
import numpy as np
import pytest

from cramera.laboratory_centrifuge_command import CentrifugeAction
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_sample_transfer import LaboratorySampleTransfer, SampleGrasp
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% supported post-cycle state
class ExtractionFixture(StrEnum):
    """Recorded physical state after two samples have reached the cooler."""

    THIRD_UNLOAD = "laboratory_third_unload.json"
    """Human-readable measured joints and supported samples before extraction."""


@dataclass
class RecordedJoint:
    """Measured generalized coordinates of one physical joint."""

    position: list[float]
    """Scalar hinge coordinate or complete floating-body pose."""
    velocity: list[float]
    """Measured scalar velocity or floating-body spatial velocity."""


@dataclass
class SupportedExtractionState:
    """Initialize a regression fixture from a genuine supported batch checkpoint."""

    joints: dict[str, RecordedJoint]
    """Measured joint coordinates indexed by their compiled model names."""
    joint_targets: dict[str, float]
    """Existing bounded robot actuator references at the checkpoint."""
    sample_volumes: dict[str, float]
    """Actual liquid quantities retained by the four physical samples."""
    secured_slots: dict[str, str]
    """Other samples still retained by their rotor holder clamps."""
    centrifuge_lid_target: float
    """Physical lid actuator target during supported extraction."""
    refrigerator_target: float
    """Cabinet door's requested hinge angle."""
    refrigerator_reference: float
    """Current rate-limited door actuator reference."""

    @classmethod
    def read(cls) -> SupportedExtractionState:
        """Load the named, readable physical checkpoint."""
        path = Path(__file__).parent / "dataset" / ExtractionFixture.THIRD_UNLOAD
        state = cls(**json.loads(path.read_text()))
        state.joints = {
            name: RecordedJoint(**joint) for name, joint in state.joints.items()
        }
        return state

    def initialize(self, physics: LaboratoryRobotPhysics) -> None:
        """Place the initial fixture before actuator-only simulation begins."""
        for name, recorded in self.joints.items():
            joint = physics.model.joint(name).id
            position = int(physics.model.jnt_qposadr[joint])
            velocity = int(physics.model.jnt_dofadr[joint])
            physics.data.qpos[position : position + len(recorded.position)] = (
                recorded.position
            )
            physics.data.qvel[velocity : velocity + len(recorded.velocity)] = (
                recorded.velocity
            )
        physics.set_joint_targets(self.joint_targets)
        physics.centrifuge.lid_target = self.centrifuge_lid_target
        physics.data.ctrl[physics.centrifuge.lid_actuator] = self.centrifuge_lid_target
        physics.refrigerator.target_angle = self.refrigerator_target
        physics.refrigerator.reference_angle = self.refrigerator_reference
        physics.data.ctrl[physics.refrigerator.actuator] = self.refrigerator_reference
        mujoco.mj_forward(physics.model, physics.data)
        for key, volume in self.sample_volumes.items():
            physics.fill_liquid(key, volume)
        for slot, key in self.secured_slots.items():
            physics.centrifuge_command(CentrifugeAction.SECURE, slot=slot, key=key)


# %% measured clearance and supported release
def test_processed_sample_returns_to_rack_with_retained_contents(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """Real fingertip transport clears the rotor and dispenser after a cycle."""
    physics = mobile_robot
    state = SupportedExtractionState.read()
    state.initialize(physics)
    key = LaboratoryBody.TEAL_TUBE
    destination = tuple(physics.scene["laboratory"]["rackSlots"][0]["pose"][:3])
    transfer = LaboratorySampleTransfer(
        physics=physics,
        object_key=key,
        destination=destination,
        support_body=LaboratoryBody.RACK,
        grasp=SampleGrasp.TOP,
    )
    transfer.start()
    maximum_seconds = sum(stage.duration for stage in transfer.stages) + (
        len(transfer.stages) * transfer.parameters.stage_timeout
    )
    for _ in range(int(maximum_seconds / physics.timestep)):
        before = physics.data.qpos.copy()
        transfer.advance()
        np.testing.assert_array_equal(physics.data.qpos, before)
        physics.step()
        before = physics.data.qpos.copy()
        transfer.observe()
        np.testing.assert_array_equal(physics.data.qpos, before)
        if transfer.state != ProgramState.RUNNING:
            break
    assert transfer.state == ProgramState.SUCCEEDED, transfer.snapshot()
    assert transfer.released
    assert physics.liquid.tubes[key].volume_ml == pytest.approx(
        state.sample_volumes[key]
    )
    assert physics.target is None
    np.testing.assert_array_equal(physics.data.xfrc_applied, 0)
