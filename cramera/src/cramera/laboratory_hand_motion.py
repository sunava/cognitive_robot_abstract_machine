"""
Bounded Cartesian PR2 hand intentions evaluated against physical joints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cramera.laboratory_mixing_program import LaboratoryMixingProgram
from cramera.laboratory_physics_program import JointReference
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics


# %% measured Cartesian servo
@dataclass
class LaboratoryHandMotion:
    """
    Move an empty or contact-held left hand through bounded joint actuators.
    """

    physics: LaboratoryRobotPhysics
    """Authoritative robot joints and contacts."""
    maximum_speed: float = 0.5
    """
    Maximum arm joint reference speed in radians per second.
    """

    finger_speed: float = 0.2
    """
    Maximum finger closure speed in radians per second.
    """

    gain: float = 4.0
    """
    Cartesian convergence rate per second.
    """

    regularization: float = 0.0001
    """
    Damped least-squares inverse regularization.
    """

    shoulder_posture: float | None = None
    """
    Redundant shoulder preference preserving a raised-elbow reach.
    """

    tool: int = field(init=False)
    """
    Physical tool frame body.
    """

    names: list[str] = field(init=False)
    """
    Independently actuated left arm joints.
    """

    indices: list[int] = field(init=False)
    """
    Velocity coordinates of the controlled joints.
    """

    def __post_init__(self) -> None:
        """
        Resolve the physical tool and independent left arm chain.
        """
        self.tool = self.physics.model.body(LaboratoryMixingProgram.TOOL_FRAME).id
        self.names = [
            name
            for name in self.physics.joint_targets
            if name.startswith("pr2/l_") and "gripper" not in name
        ]
        self.indices = [
            int(self.physics.model.jnt_dofadr[self.physics.model.joint(name).id])
            for name in self.names
        ]

    @property
    def position(self) -> np.ndarray:
        """
        Measured tool origin in world coordinates.
        """
        return self.physics.data.xpos[self.tool].copy()

    @property
    def rotation(self) -> Rotation:
        """
        Measured tool orientation in world coordinates.
        """
        return Rotation.from_matrix(self.physics.data.xmat[self.tool].reshape(3, 3))

    def advance(
        self, position: np.ndarray, rotation: Rotation, finger: float, elapsed: float
    ) -> None:
        """
        Project a Cartesian pose error onto limited physical joint references.
        """
        translation = np.zeros((3, self.physics.model.nv))
        angular = np.zeros_like(translation)
        mujoco.mj_jacBody(
            self.physics.model, self.physics.data, translation, angular, self.tool
        )
        jacobian = np.vstack((translation[:, self.indices], angular[:, self.indices]))
        error = np.concatenate(
            (position - self.position, (rotation * self.rotation.inv()).as_rotvec())
        )
        active = list(range(len(self.names)))
        speeds = np.zeros(len(self.names))
        for _ in self.names:
            selected = jacobian[:, active]
            proposal = selected.T @ np.linalg.solve(
                selected @ selected.T + self.regularization * np.eye(6),
                self.gain * error,
            )
            shoulder = self.names.index("pr2/l_shoulder_lift_joint")
            if self.shoulder_posture is not None and shoulder in active:
                column = active.index(shoulder)
                nullspace = np.eye(len(active)) - np.linalg.pinv(selected) @ selected
                direction = nullspace[:, column]
                if direction[column] > 0.001:
                    joint = self.physics.robot_joints[self.names[shoulder]]
                    desired = 2.0 * (
                        self.shoulder_posture
                        - self.physics.data.qpos[joint.position_index]
                    )
                    proposal += (
                        direction * (desired - proposal[column]) / direction[column]
                    )
            blocked = []
            for index, speed in zip(active, proposal):
                joint = self.physics.robot_joints[self.names[index]]
                target = self.physics.joint_targets[self.names[index]]
                if (speed < 0 and target <= joint.lower + 0.002) or (
                    speed > 0 and target >= joint.upper - 0.002
                ):
                    blocked.append(index)
                else:
                    speeds[index] = speed
            if not blocked:
                break
            for index in blocked:
                active.remove(index)
                speeds[index] = 0.0
        speeds /= max(1.0, np.max(np.abs(speeds)) / self.maximum_speed)
        targets = {
            name: float(
                np.clip(
                    self.physics.joint_targets[name] + elapsed * speed,
                    self.physics.robot_joints[name].lower,
                    self.physics.robot_joints[name].upper,
                )
            )
            for name, speed in zip(self.names, speeds)
        }
        previous = self.physics.joint_targets[JointReference.GRIPPER_JOINT]
        targets[JointReference.GRIPPER_JOINT] = float(
            np.clip(
                finger,
                previous - self.finger_speed * elapsed,
                previous + self.finger_speed * elapsed,
            )
        )
        self.physics.set_joint_targets(targets)
