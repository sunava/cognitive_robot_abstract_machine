"""Keep the PR2's optical camera pointed at its measured manipulation frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from cramera.laboratory_physics_robot import LaboratoryRobotPhysics, RobotHeadJoint
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pointing import Pointing
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body


# %% observed frames and tracking bounds
class GazeFrame(StrEnum):
    """Measured PR2 frames whose relationship defines the viewing direction."""

    CAMERA = "pr2/wide_stereo_optical_frame"
    """Optical camera frame whose positive Z axis is the viewing direction."""
    TOOL = "pr2/l_gripper_tool_frame"
    """Manipulation frame midway between the active left gripper's fingers."""
    TARGET = "manipulation_target"
    """Symbolic coordinates updated from the measured tool each control cycle."""


class GazeField(StrEnum):
    """Measured gaze evidence published in the laboratory snapshot."""

    CONSTRAINT = "constraint"
    """Native Giskard task enforcing the viewing direction."""
    TARGET_FRAME = "targetFrame"
    """Measured manipulation frame followed by the head."""
    CAMERA_FRAME = "cameraFrame"
    """Optical frame whose viewing direction is controlled."""
    ANGLE_ERROR = "angleErrorRadians"
    """Current angle between the physical viewing and target directions."""


@dataclass(frozen=True)
class GazeParameters:
    """Conservative pointing speed and fixed controller cadence."""

    frequency: float = 20.0
    """Pointing updates per simulated second, independent of renderer timing."""
    maximum_velocity: float = 0.5
    """Head command speed in radians per second, below both PR2 URDF limits."""
    threshold: float = 0.025
    """Angular alignment tolerance in radians for the pointing observation."""


# %% persistent head-only pointing
@dataclass
class LaboratoryGaze:
    """Track the measured tool using a persistent head-only Giskard constraint.

    The controller's two-joint world receives physical measurements. Its predicted
    positions become bounded servo targets; only the physics engine moves the robot.
    """

    physics: LaboratoryRobotPhysics
    """Authoritative articulation and bounded actuator command interface."""
    parameters: GazeParameters = field(default_factory=GazeParameters)
    """Pointing cadence, speed and observation tolerance."""
    world: World = field(init=False)
    """Minimal head kinematics, expressed relative to the measured head support."""
    executor: Executor = field(init=False)
    """Persistent native Giskard pointing task and quadratic-program controller."""
    target: Point3 = field(init=False)
    """Live auxiliary coordinates of the measured tool in the head support frame."""
    connections: dict[RobotHeadJoint, RevoluteConnection] = field(
        init=False, default_factory=dict
    )
    """Head axes available to the quadratic program; arms and base are absent."""
    support_body: int = field(init=False)
    """MuJoCo body immediately above the head pan joint."""
    next_update: float = field(init=False, default=0.0)
    """Earliest simulation time at which another fixed-size command is computed."""

    def __post_init__(self) -> None:
        """Compile one continuing pointing constraint from the physical head chain."""
        self.world = World()
        self._build_head()
        context = MotionStatechartContext(
            world=self.world,
            qp_controller_config=QPControllerConfig(
                target_frequency=self.parameters.frequency, verbose=False
            ),
        )
        self.target = Point3.create_with_variables(GazeFrame.TARGET)
        self.target.reference_frame = self.world.root
        context.float_variable_data.register_expression(self.target)
        self._update_target(context)
        camera = self.world.get_body_by_name(GazeFrame.CAMERA)
        chart = MotionStatechart()
        chart.add_node(
            Pointing(
                root_link=self.world.root,
                tip_link=camera,
                goal_point=self.target,
                pointing_axis=Vector3.Z(reference_frame=camera),
                max_velocity=self.parameters.maximum_velocity,
                threshold=self.parameters.threshold,
            )
        )
        self.executor = Executor(context)
        self._observe_joints()
        self.executor.compile(chart)

    def _build_head(self) -> None:
        """Retain physical fixed transforms and only the pan and tilt coordinates."""
        model = self.physics.model
        pan_body = int(model.jnt_bodyid[model.joint(RobotHeadJoint.PAN).id])
        self.support_body = int(model.body_parentid[pan_body])
        path = []
        current = model.body(GazeFrame.CAMERA).id
        while current != self.support_body:
            path.append(current)
            current = int(model.body_parentid[current])
        head_bodies = {
            int(model.jnt_bodyid[model.joint(name).id]): name for name in RobotHeadJoint
        }
        parent = Body(name=PrefixedName(model.body(self.support_body).name))
        with self.world.modify_world():
            for body_id in reversed(path):
                child = Body(name=PrefixedName(model.body(body_id).name))
                quaternion = model.body_quat[body_id]
                transform = HomogeneousTransformationMatrix.from_xyz_quaternion(
                    *model.body_pos[body_id], *quaternion[1:], quaternion[0]
                )
                name = head_bodies.get(body_id)
                if name is None:
                    connection = FixedConnection(
                        parent=parent,
                        child=child,
                        parent_T_connection_expression=transform,
                    )
                else:
                    joint_id = model.joint(name).id
                    anchor = model.jnt_pos[joint_id]
                    limits = self.physics.robot_joints[name]
                    connection = RevoluteConnection.create_with_dofs(
                        world=self.world,
                        parent=parent,
                        child=child,
                        name=PrefixedName(name),
                        axis=Vector3.from_iterable(model.jnt_axis[joint_id]),
                        parent_T_connection_expression=transform
                        @ HomogeneousTransformationMatrix.from_xyz_rpy(*anchor),
                        connection_T_child_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                            *-anchor
                        ),
                        dof_limits=DegreeOfFreedomLimits(
                            lower=DerivativeMap(
                                position=limits.lower,
                                velocity=-self.parameters.maximum_velocity,
                            ),
                            upper=DerivativeMap(
                                position=limits.upper,
                                velocity=self.parameters.maximum_velocity,
                            ),
                        ),
                    )
                    self.connections[name] = connection
                self.world.add_connection(connection)
                parent = child

    def _observe_joints(self) -> None:
        """Seed the control model from current physical positions and derivatives."""
        for name, connection in self.connections.items():
            joint = self.physics.model.joint(name)
            state = self.world.state[connection.raw_dof.id]
            state.position = float(self.physics.data.qpos[joint.qposadr[0]])
            state.velocity = float(self.physics.data.qvel[joint.dofadr[0]])
            state.acceleration = float(self.physics.data.qacc[joint.dofadr[0]])
            state.jerk = 0.0
        self.world.notify_state_change()

    def _update_target(self, context: MotionStatechartContext) -> None:
        """Express the current manipulation position in the moving head support."""
        support = self.physics.data.body(self.support_body)
        rotation = support.xmat.reshape(3, 3)
        displacement = self.physics.data.body(GazeFrame.TOOL).xpos - support.xpos
        context.float_variable_data.set_value(self.target, rotation.T @ displacement)

    def advance(self) -> None:
        """Submit one bounded head command when the fixed simulation cadence is due."""
        now = float(self.physics.data.time)
        if now + 1e-9 < self.next_update:
            return
        interval = 1.0 / self.parameters.frequency
        self.next_update = now + interval
        self._observe_joints()
        self._update_target(self.executor.context)
        self.executor.tick()
        targets = {}
        maximum_step = self.parameters.maximum_velocity * interval
        for name, connection in self.connections.items():
            joint = self.physics.robot_joints[name]
            current = float(self.physics.data.qpos[joint.position_index])
            prediction = self.world.state[connection.raw_dof.id]
            stiffness = self.physics.model.actuator_gainprm[joint.actuator_index, 0]
            damping = -self.physics.model.actuator_biasprm[joint.actuator_index, 2]
            # Compensate the servo's velocity damping while following the QP motion.
            target = prediction.position + damping / stiffness * prediction.velocity
            targets[name] = float(
                np.clip(
                    target,
                    max(joint.lower, current - maximum_step),
                    min(joint.upper, current + maximum_step),
                )
            )
        self.physics.set_joint_targets(targets)

    @property
    def angle_error(self) -> float:
        """Measure the optical-axis angle to the current physical gripper position."""
        camera = self.physics.data.body(GazeFrame.CAMERA)
        direction = self.physics.data.body(GazeFrame.TOOL).xpos - camera.xpos
        direction = direction / np.linalg.norm(direction)
        axis = camera.xmat.reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(np.dot(axis, direction), -1.0, 1.0)))

    def snapshot(self) -> dict[GazeField, str | float]:
        """Publish the active constraint and its actual physical pointing error."""
        return {
            GazeField.CONSTRAINT: Pointing.__name__,
            GazeField.TARGET_FRAME: GazeFrame.TOOL,
            GazeField.CAMERA_FRAME: GazeFrame.CAMERA,
            GazeField.ANGLE_ERROR: self.angle_error,
        }
