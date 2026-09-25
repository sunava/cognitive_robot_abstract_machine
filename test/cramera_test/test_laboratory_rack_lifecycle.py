"""
Rack transport preserves physical state across ownership and lifecycle changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import pytest

from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import JointReference, ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_rack_transfer import LaboratoryRackTransfer, RackTransferPhase
from cramera.laboratory_workflow import LaboratoryWorkflowProgram
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% physical lifecycle fixtures
@dataclass(frozen=True)
class FingerProfile:
    """
    Detached actuator settings that must survive temporary carrier ownership.
    """

    force_range: np.ndarray
    """
    Original lower and upper finger effort bounds.
    """

    gain: np.ndarray
    """
    Original position and velocity actuator gains.
    """

    bias: np.ndarray
    """
    Original position and velocity actuator bias terms.
    """

    @classmethod
    def capture(cls, physics: LaboratoryRobotPhysics) -> FingerProfile:
        """
        Copy the actual finger profile without retaining mutable model arrays.
        """
        finger = physics.model.actuator(JointReference.GRIPPER_JOINT).id
        return cls(
            physics.model.actuator_forcerange[finger].copy(),
            physics.model.actuator_gainprm[finger].copy(),
            physics.model.actuator_biasprm[finger].copy(),
        )

    def assert_matches(self, physics: LaboratoryRobotPhysics) -> None:
        """
        Require exact restoration of every captured actuator coefficient.
        """
        actual = self.capture(physics)
        np.testing.assert_array_equal(actual.force_range, self.force_range)
        np.testing.assert_array_equal(actual.gain, self.gain)
        np.testing.assert_array_equal(actual.bias, self.bias)


@pytest.fixture
def open_loaded_drawer(mobile_robot: LaboratoryRobotPhysics) -> LaboratoryRobotPhysics:
    """
    Initialize the open drawer with separate samples before testing controller
    ownership.
    """
    physics = mobile_robot
    drawer = physics.drawer
    physics.data.qpos[physics.model.jnt_qposadr[drawer.joint]] = drawer.open_position
    drawer.restore_hold(drawer.open_position)
    mujoco.mj_forward(physics.model, physics.data)
    physics.prepare_rack_in_drawer()
    return physics


# %% pause and resume
def test_pause_and_resume_keep_the_same_rack_operation(
    open_loaded_drawer: LaboratoryRobotPhysics,
) -> None:
    """
    Pausing holds references while resuming retains phase, progress and actuator
    ownership.
    """
    physics = open_loaded_drawer
    before = physics.data.qpos.copy()
    action = LaboratoryRackTransfer(physics, LaboratoryWorkflowProgram.SAMPLE_KEYS)
    action.start()
    assert action.state == ProgramState.RUNNING
    np.testing.assert_array_equal(physics.data.qpos, before)
    finger = physics.model.actuator(JointReference.GRIPPER_JOINT).id
    np.testing.assert_array_equal(
        physics.model.actuator_forcerange[finger],
        [-action.parameters.finger_torque, action.parameters.finger_torque],
    )
    for _ in range(int(2 * action.parameters.period / physics.timestep)):
        action.advance()
    np.testing.assert_array_equal(physics.data.qpos, before)
    elapsed = action.elapsed
    phase_time = action.phase_time
    phase = action.phase
    initial_joints = action.start_joints.copy()
    controls = physics.data.ctrl.copy()
    profile = FingerProfile.capture(physics)
    action.pause()
    assert action.state == ProgramState.PAUSED
    action.advance()
    action.observe()
    assert action.elapsed == elapsed
    assert action.phase_time == phase_time
    assert action.phase == phase
    np.testing.assert_array_equal(physics.data.ctrl, controls)
    profile.assert_matches(physics)
    action.start()
    assert action.state == ProgramState.RUNNING
    assert action.start_joints == initial_joints
    assert action.elapsed == elapsed
    assert action.phase_time == phase_time
    assert action.phase == phase
    np.testing.assert_array_equal(physics.data.qpos, before)
    np.testing.assert_array_equal(physics.data.ctrl, controls)
    profile.assert_matches(physics)
    action.cancel()


# %% termination restores the original actuator profile
@pytest.mark.parametrize("state", [ProgramState.RUNNING, ProgramState.PAUSED])
def test_cancellation_restores_original_finger_limits_gain_and_bias(
    open_loaded_drawer: LaboratoryRobotPhysics, state: ProgramState
) -> None:
    """
    Cancellation restores the exact original profile without resetting carried-body
    coordinates.
    """
    physics = open_loaded_drawer
    original = FingerProfile.capture(physics)
    before = physics.data.qpos.copy()
    action = LaboratoryRackTransfer(physics, LaboratoryWorkflowProgram.SAMPLE_KEYS)
    action.start()
    if state == ProgramState.PAUSED:
        action.pause()
    assert action.state == state
    action.cancel()
    assert action.state == ProgramState.FAILED
    original.assert_matches(physics)
    np.testing.assert_array_equal(physics.data.qpos, before)
    controls = physics.data.ctrl.copy()
    action.start()
    action.advance()
    action.observe()
    action.pause()
    assert action.state == ProgramState.FAILED
    original.assert_matches(physics)
    np.testing.assert_array_equal(physics.data.qpos, before)
    np.testing.assert_array_equal(physics.data.ctrl, controls)


def test_grasp_timeout_restores_original_finger_profile(
    open_loaded_drawer: LaboratoryRobotPhysics,
) -> None:
    """
    A failed contact wait releases temporary actuator settings without a physical reset.
    """
    physics = open_loaded_drawer
    original = FingerProfile.capture(physics)
    before = physics.data.qpos.copy()
    action = LaboratoryRackTransfer(physics, LaboratoryWorkflowProgram.SAMPLE_KEYS)
    action.start()
    action.phase = RackTransferPhase.GRASP
    action.phase_time = action.parameters.phase_timeout + physics.timestep
    action.observe()
    assert action.state == ProgramState.FAILED
    assert not action.grasp_verified
    original.assert_matches(physics)
    np.testing.assert_array_equal(physics.data.qpos, before)


# %% initial physical validation
def test_closed_drawer_rejects_retrieval_without_changing_any_actuator(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    A closed drawer cannot begin loaded retrieval or install a stronger finger profile.
    """
    physics = mobile_robot
    physics.prepare_rack_in_drawer()
    original = FingerProfile.capture(physics)
    before = physics.data.qpos.copy()
    controls = physics.data.ctrl.copy()
    action = LaboratoryRackTransfer(physics, LaboratoryWorkflowProgram.SAMPLE_KEYS)
    with pytest.raises(InvalidPhysicsTarget):
        action.start()
    assert action.state == ProgramState.IDLE
    original.assert_matches(physics)
    np.testing.assert_array_equal(physics.data.qpos, before)
    np.testing.assert_array_equal(physics.data.ctrl, controls)


def test_non_sample_target_rejects_retrieval_without_changing_physics(
    open_loaded_drawer: LaboratoryRobotPhysics,
) -> None:
    """
    A loose stopper cannot be claimed as one of the rack's required sample vessels.
    """
    physics = open_loaded_drawer
    original = FingerProfile.capture(physics)
    before = physics.data.qpos.copy()
    controls = physics.data.ctrl.copy()
    action = LaboratoryRackTransfer(physics, (LaboratoryBody.STOPPER,))
    with pytest.raises(InvalidPhysicsTarget):
        action.start()
    assert action.state == ProgramState.IDLE
    original.assert_matches(physics)
    np.testing.assert_array_equal(physics.data.qpos, before)
    np.testing.assert_array_equal(physics.data.ctrl, controls)


def test_already_supported_rack_finishes_without_installing_transport_gains(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    An already placed loaded rack completes as a measured no-op.
    """
    physics = mobile_robot
    physics.step(500)
    original = FingerProfile.capture(physics)
    before = physics.data.qpos.copy()
    controls = physics.data.ctrl.copy()
    action = LaboratoryRackTransfer(physics, LaboratoryWorkflowProgram.SAMPLE_KEYS)
    action.start()
    assert action.state == ProgramState.SUCCEEDED
    assert action.phase == RackTransferPhase.COMPLETE
    original.assert_matches(physics)
    np.testing.assert_array_equal(physics.data.qpos, before)
    np.testing.assert_array_equal(physics.data.ctrl, controls)
