"""Contact-driven PR2 operation of articulated laboratory covers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cramera.laboratory_centrifuge import CentrifugeGeometry, InvalidCentrifugeOperation
from cramera.laboratory_hand_motion import LaboratoryHandMotion
from cramera.laboratory_mixing_program import LaboratoryMixingProgram
from cramera.laboratory_physics_program import JointReference, ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics, RobotBaseJoint
from cramera.laboratory_refrigerator import RefrigeratorGeometry
from cramera.laboratory_robot_base import LaboratoryBasePath


# %% instrument and execution contract
class HingeInstrument(StrEnum):
    """Articulated instruments with a graspable exterior handle."""

    CENTRIFUGE = "centrifuge"
    """Rear-hinged protective rotor cover."""
    REFRIGERATOR = "refrigerator"
    """Side-hinged glazed storage door."""


class HingePhase(StrEnum):
    """Measured stages of one hand-operated hinge movement."""

    STOW = "stow"
    """Raise the empty hand into its transport posture."""
    TRAVEL = "travel"
    """Move to the instrument through the free frontal corridor."""
    APPROACH = "approach"
    """Align open fingertips with the measured handle."""
    CLEARANCE = "clearance"
    """Reach above the handle after raising the empty hand."""
    RAISE = "raise"
    """Raise the empty hand ahead of the instrument's sweep."""
    GRASP = "grasp"
    """Require opposing handle contacts before unlocking the hinge."""
    TURN = "turn"
    """Move the held handle along its physical hinge arc."""
    RELEASE = "release"
    """Secure the measured endpoint and open the fingers."""
    CLEAR_HANDLE = "clear_handle"
    """Lift released fingers clear before changing the wrist pitch."""
    WITHDRAW = "withdraw"
    """Clear the instrument after releasing the handle."""
    COMPLETE = "complete"
    """The measured hinge destination has been reached and released."""


class HingeField(StrEnum):
    """Observable hand and instrument execution feedback."""

    STATE = "state"
    """Execution lifecycle."""
    PHASE = "phase"
    """Current hand movement."""
    INSTRUMENT = "instrument"
    """Selected physical instrument."""
    OPENING = "opening"
    """Whether this action opens or closes the cover."""
    ANGLE = "angle"
    """Measured hinge position."""
    TARGET = "targetAngle"
    """Requested hinge endpoint."""
    FORCE = "handleForce"
    """Measured sum of fingertip normal forces."""
    ERROR = "error"
    """Physical failure explanation."""


@dataclass(frozen=True)
class HingeParameters:
    """Conservative physical motion and measured acceptance bounds."""

    period: float = 0.01
    """Cartesian control interval in simulated seconds."""
    angular_speed: float = 0.12
    """Maximum handle arc reference speed in radians per second."""
    base_speed: float = 0.08
    """Maximum base reference speed in metres per second."""
    finger_torque: float = 0.5
    """Maximum finger torque while manipulating robust instrument handles."""
    refrigerator_finger_torque: float = 1.5
    """Finger torque limit for the vertical refrigerator handle in Newton-metres."""
    refrigerator_finger_stiffness: float = 20.0
    """Refrigerator finger position stiffness in Newton-metres per radian."""
    refrigerator_angular_speed: float = 0.06
    """Refrigerator hinge reference speed in radians per second."""
    refrigerator_clearance_height: float = 0.85
    """Tool height above the open refrigerator door before exterior descent."""
    refrigerator_grasp_pitch: float = -np.pi / 3
    """Downward front-grasp pitch preserving wrist travel throughout the door arc."""
    open_finger: float = 0.5
    """Opening target providing clearance around the wider lid handle."""
    closed_finger: float = 0.0
    """Contact-limited finger closure target."""
    contact_force: float = 0.02
    """Required opposing fingertip normal force in Newtons."""
    contact_time: float = 0.15
    """Continuous opposing handle contact required to unlock."""
    centering_speed: float = 0.005
    """Maximum force-balancing hand adjustment in metres per second."""
    centering_limit: float = 0.003
    """Maximum lateral correction from the measured grasp in metres."""
    grasp_estimation_rate: float = 0.5
    """Rate of measured grip-displacement convergence per second."""
    maximum_grasp_correction: float = 0.003
    """Maximum compensation for contact settling in metres per second."""
    lost_contact_time: float = 0.5
    """Maximum interruption of a confirmed handle grasp in seconds."""
    position_tolerance: float = 0.006
    """Cartesian alignment tolerance in metres."""
    handle_clearance_height: float = 0.01
    """Initial vertical lift before withdrawing from the opened lid in metres."""
    handle_clearance_tolerance: float = 0.002
    """Measured completion tolerance for the initial finger-clearance lift."""
    lid_withdrawal: tuple[float, float, float] = (0.0, 0.02, 0.04)
    """World displacement clearing the opened cover after the initial finger lift."""
    lid_withdrawal_pitch: float = np.pi / 18
    """Small wrist rotation preserving reach while withdrawing above the lid."""
    angle_tolerance: float = 0.02
    """Measured hinge endpoint tolerance in radians."""
    phase_timeout: float = 40.0
    """Maximum settling time for one physical manipulation stage."""


# %% physical hinge manipulation
@dataclass
class LaboratoryHingeManipulation:
    """Open or close a physical hinge through measured gripper-handle contacts."""

    physics: LaboratoryRobotPhysics
    """Authoritative articulated simulation."""
    instrument: HingeInstrument
    """Instrument whose cover is manipulated."""
    opening: bool
    """Whether to open the cover or return it to its latch."""
    parameters: HingeParameters = field(default_factory=HingeParameters)
    """Bounded references and physical completion thresholds."""
    state: ProgramState = field(init=False, default=ProgramState.IDLE)
    """Execution lifecycle independent of physics stepping."""
    phase: HingePhase = field(init=False, default=HingePhase.STOW)
    """Current measured manipulation stage."""
    error: str | None = field(init=False, default=None)
    """Physical reason for an unsuccessful action."""
    hand: LaboratoryHandMotion = field(init=False)
    """Measured Cartesian hand servo."""
    joint: int = field(init=False)
    """Physical hinge joint index."""
    actuator: int = field(init=False)
    """Instrument motor disabled while the hand drives the hinge."""
    handle: int = field(init=False)
    """Collision geometry of the visible handle."""
    target_angle: float = field(init=False)
    """Requested mechanical endpoint in radians."""
    reference_angle: float = field(init=False)
    """Bounded angular reference for the held handle."""
    rotation: Rotation = field(init=False)
    """World gripper orientation during the handle movement."""
    motor_range: np.ndarray = field(init=False)
    """Original instrument holding torque bounds."""
    finger_range: np.ndarray = field(init=False)
    """Original finger torque bounds restored after manipulation."""
    finger_gain: np.ndarray = field(init=False)
    """Original finger actuator gain parameters restored after manipulation."""
    finger_bias: np.ndarray = field(init=False)
    """Original finger actuator bias parameters restored after manipulation."""
    phase_time: float = field(init=False, default=0.0)
    """Active simulated seconds in the current phase."""
    elapsed: float = field(init=False, default=0.0)
    """Total active simulated seconds."""
    control_time: float = field(init=False, default=0.0)
    """Accumulated Cartesian control interval."""
    contact_duration: float = field(init=False, default=0.0)
    """Uninterrupted opposing handle contact duration."""
    lost_contact_duration: float = field(init=False, default=0.0)
    """Duration of an interrupted verified grasp."""
    grasp_verified: bool = field(init=False, default=False)
    """Whether sustained opposing handle contacts were observed."""
    moved: bool = field(init=False, default=False)
    """Whether physical hinge motion was needed."""
    handle_force: float = field(init=False, default=0.0)
    """Current total opposing fingertip force on the handle."""
    finger_forces: dict[str, float] = field(init=False, default_factory=dict)
    """Independent measured handle loads indexed by fingertip body name."""
    grip_adjustment: float = field(init=False, default=0.0)
    """Force-balancing displacement along the captured gripper jaw axis."""
    maximum_handle_force: float = field(init=False, default=0.0)
    """Largest measured handle force."""
    maximum_motor_force: float = field(init=False, default=0.0)
    """Largest instrument motor force during hand-driven arc motion."""
    grasp_angle: float = field(init=False, default=0.0)
    """Measured hinge angle when the handle grasp is accepted."""
    grasp_rotation: Rotation = field(init=False, default_factory=Rotation.identity)
    """Measured tool orientation at the accepted handle grasp."""
    grip_offset: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    """Tool-minus-handle displacement measured at grasp confirmation."""
    release_position: np.ndarray = field(
        init=False, default_factory=lambda: np.zeros(3)
    )
    """Measured hand position held while the fingers open."""
    release_rotation: Rotation = field(init=False, default_factory=Rotation.identity)
    """Measured hand orientation held while releasing the handle."""
    withdraw_position: np.ndarray = field(
        init=False, default_factory=lambda: np.zeros(3)
    )
    """Clearance target after releasing the handle."""
    dock: np.ndarray = field(init=False)
    """Planar base displacement at the instrument approach."""
    start_joints: dict[str, float] = field(init=False, default_factory=dict)
    """Measured joints before the initial empty-hand transport posture."""
    transit_position: np.ndarray = field(
        init=False, default_factory=lambda: np.zeros(3)
    )
    """Measured front-side Cartesian waypoint before base travel."""

    def __post_init__(self) -> None:
        """Resolve authored handle geometry and the instrument's physical hinge."""
        self.hand = LaboratoryHandMotion(self.physics)
        if self.instrument == HingeInstrument.CENTRIFUGE:
            self.joint = self.physics.model.joint(CentrifugeGeometry.LID_JOINT).id
            self.actuator = self.physics.model.actuator(CentrifugeGeometry.LID_MOTOR).id
            self.handle = self.physics.model.geom(
                "laboratory_centrifuge_lid_lid_handle"
            ).id
            self.dock = np.array([-0.6, 0.0])
        else:
            self.joint = self.physics.model.joint(RefrigeratorGeometry.JOINT).id
            self.actuator = self.physics.model.actuator(RefrigeratorGeometry.MOTOR).id
            self.handle = self.physics.model.geom(
                "laboratory_refrigerator_door_refrigerator_handle"
            ).id
            self.dock = np.array([-1.25, -0.3])
            self.hand.shoulder_posture = 0.3
        self.target_angle = (
            float(self.physics.model.jnt_range[self.joint, 1]) if self.opening else 0.0
        )
        self.reference_angle = self.angle
        self.motor_range = self.physics.model.actuator_forcerange[self.actuator].copy()
        finger = self.physics.model.actuator(JointReference.GRIPPER_JOINT).id
        self.finger_range = self.physics.model.actuator_forcerange[finger].copy()
        self.finger_gain = self.physics.model.actuator_gainprm[finger].copy()
        self.finger_bias = self.physics.model.actuator_biasprm[finger].copy()
        self.rotation = Rotation.from_matrix(
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        )

        if self.instrument == HingeInstrument.REFRIGERATOR:
            self.rotation = self.rotation * Rotation.from_euler(
                "y", self.parameters.refrigerator_grasp_pitch
            )

    @property
    def angular_speed(self) -> float:
        """Instrument-specific bounded angular reference rate."""
        return (
            self.parameters.refrigerator_angular_speed
            if self.instrument == HingeInstrument.REFRIGERATOR
            else self.parameters.angular_speed
        )

    @property
    def angle(self) -> float:
        """Measured hinge position in radians."""
        return float(self.physics.data.qpos[self.physics.model.jnt_qposadr[self.joint]])

    @property
    def handle_position(self) -> np.ndarray:
        """Measured collision-handle centre in world coordinates."""
        return self.physics.data.geom_xpos[self.handle].copy()

    @property
    def approach_position(self) -> np.ndarray:
        """Tool centre exposing the handle between the opposing fingertip pads."""
        offset = np.zeros(3)
        if self.instrument == HingeInstrument.CENTRIFUGE:
            body = self.physics.model.geom_bodyid[self.handle]
            offset = self.physics.data.xmat[body].reshape(3, 3)[:, 2] * 0.01
        return self.handle_position + offset

    def start(self) -> None:
        """Begin or resume a manipulation without assigning physical positions."""
        if self.state not in (ProgramState.IDLE, ProgramState.PAUSED):
            return
        if (
            self.instrument == HingeInstrument.CENTRIFUGE
            and not self.physics.centrifuge.stopped
        ):
            raise InvalidCentrifugeOperation("Stop the rotor before grasping the lid")
        if self.state == ProgramState.IDLE:
            if abs(self.angle - self.target_angle) <= self.parameters.angle_tolerance:
                self.state = ProgramState.SUCCEEDED
                self.phase = HingePhase.COMPLETE
                return
            self.start_joints = self.physics.robot_state()
            self.moved = True
            finger = self.physics.model.actuator(JointReference.GRIPPER_JOINT).id
            torque = self.parameters.finger_torque
            if self.instrument == HingeInstrument.REFRIGERATOR:
                torque = self.parameters.refrigerator_finger_torque
                stiffness = self.parameters.refrigerator_finger_stiffness
                self.physics.model.actuator_gainprm[finger, 0] = stiffness
                self.physics.model.actuator_biasprm[finger, 1] = -stiffness
            self.physics.model.actuator_forcerange[finger] = (-torque, torque)
            self._hold(self.angle)
        self.state = ProgramState.RUNNING

    def pause(self) -> None:
        """Hold current arm references while retaining a physically held handle."""
        if self.state == ProgramState.RUNNING:
            self.state = ProgramState.PAUSED

    def cancel(self) -> None:
        """End manipulation and restore physical actuator limits before reset."""
        self._restore_actuators()
        if self.state in (ProgramState.RUNNING, ProgramState.PAUSED):
            self.state = ProgramState.FAILED
            self.error = "Handle manipulation was cancelled"

    def _restore_actuators(self, target_angle: float | None = None) -> None:
        """Hold the measured hinge and return the fingers to their original effort."""
        self._hold(self.angle if target_angle is None else target_angle)
        finger = self.physics.model.actuator(JointReference.GRIPPER_JOINT).id
        self.physics.model.actuator_forcerange[finger] = self.finger_range
        self.physics.model.actuator_gainprm[finger] = self.finger_gain
        self.physics.model.actuator_biasprm[finger] = self.finger_bias

    def _hold(self, angle: float) -> None:
        """Hold the already measured hinge endpoint after hand motion."""
        self.physics.model.actuator_forcerange[self.actuator] = self.motor_range
        self.physics.data.ctrl[self.actuator] = angle
        if self.instrument == HingeInstrument.CENTRIFUGE:
            self.physics.centrifuge.lid_target = angle
        else:
            self.physics.refrigerator.reference_angle = angle
            self.physics.refrigerator.target_angle = angle

    def _enter(self, phase: HingePhase) -> None:
        """Capture the physical origin of the next execution stage."""
        self.phase = phase
        self.phase_time = self.contact_duration = self.lost_contact_duration = 0.0
        if phase == HingePhase.TURN:
            self.reference_angle = self.angle
            self.grip_offset = self.hand.position - self.handle_position
            self.grasp_angle = self.angle
            self.grasp_rotation = self.hand.rotation
            self.grip_adjustment = 0.0
            self.physics.model.actuator_forcerange[self.actuator] = (0.0, 0.0)
        elif phase == HingePhase.RELEASE:
            self.release_position = self.hand.position
            self.release_rotation = self.hand.rotation
            self._hold(self.angle)
        elif phase == HingePhase.CLEAR_HANDLE:
            self.withdraw_position = self.release_position + np.array(
                [0.0, 0.0, self.parameters.handle_clearance_height]
            )
        elif phase == HingePhase.WITHDRAW:
            withdrawal = np.array([0.0, -0.12, 0.08])
            if self.instrument == HingeInstrument.CENTRIFUGE:
                withdrawal = np.array([0.0, 0.0, 0.08])
            if self.instrument == HingeInstrument.REFRIGERATOR:
                withdrawal = -self.hand.rotation.apply([0.12, 0.0, 0.0]) + np.array(
                    [0.0, 0.0, 0.08]
                )
            self.withdraw_position = self.hand.position + withdrawal
            if self.instrument == HingeInstrument.CENTRIFUGE and self.opening:
                self.withdraw_position = self.release_position + np.array(
                    self.parameters.lid_withdrawal
                )
        elif phase == HingePhase.CLEARANCE:
            self.transit_position = self.approach_position.copy()
            self.transit_position[2] = 1.47
            if self.instrument == HingeInstrument.REFRIGERATOR:
                rotation = (
                    Rotation.from_rotvec(
                        self.physics.data.xaxis[self.joint] * self.angle
                    )
                    * self.rotation
                )
                self.transit_position = self.approach_position - rotation.apply(
                    [0.08, 0.0, 0.0]
                )
                self.transit_position[2] = self.parameters.refrigerator_clearance_height
        elif phase == HingePhase.RAISE:
            self.transit_position = self.hand.position.copy()
            self.transit_position[2] = 1.47
        elif phase == HingePhase.TRAVEL:
            obstruction = LaboratoryBasePath(self.physics).obstruction(
                {
                    RobotBaseJoint.HORIZONTAL: float(self.dock[0]),
                    RobotBaseJoint.DEPTH: float(self.dock[1]),
                }
            )
            if obstruction is not None:
                self._fail(str(obstruction))

    def advance(self) -> None:
        """Issue bounded robot intentions while the physical hinge remains free."""
        if self.state != ProgramState.RUNNING:
            return
        self.phase_time += self.physics.timestep
        self.elapsed += self.physics.timestep
        self.control_time += self.physics.timestep
        if self.control_time < self.parameters.period:
            return
        elapsed = self.control_time
        self.control_time = 0.0
        if self.phase == HingePhase.STOW:
            fraction = min(1.0, self.phase_time / 8.0)
            fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            targets = {
                name: float(
                    np.clip(
                        self.start_joints[name]
                        + fraction * (target - self.start_joints[name]),
                        self.physics.robot_joints[name].lower,
                        self.physics.robot_joints[name].upper,
                    )
                )
                for name in self.hand.names
                for target in (
                    float(
                        np.clip(
                            self.physics.initial_positions[name],
                            self.physics.robot_joints[name].lower,
                            self.physics.robot_joints[name].upper,
                        )
                    ),
                )
            }
            torso = "pr2/torso_lift_joint"
            height = self.physics.initial_positions[torso]
            targets[torso] = float(
                np.clip(
                    self.start_joints[torso]
                    + fraction * (height - self.start_joints[torso]),
                    self.physics.robot_joints[torso].lower,
                    self.physics.robot_joints[torso].upper,
                )
            )
            targets[JointReference.GRIPPER_JOINT] = self.parameters.open_finger
            self.physics.set_joint_targets(targets)
            return
        if self.phase == HingePhase.TRAVEL:
            self._travel(elapsed)
            return
        if (
            self.instrument == HingeInstrument.REFRIGERATOR
            and self.phase != HingePhase.CLEARANCE
        ):
            torso = "pr2/torso_lift_joint"
            current = self.physics.joint_targets[torso]
            self.physics.set_joint_targets(
                {
                    torso: max(
                        self.physics.robot_joints[torso].lower, current - 0.03 * elapsed
                    )
                }
            )
        rotation = self.rotation
        if self.instrument == HingeInstrument.CENTRIFUGE and self.phase not in (
            HingePhase.CLEARANCE,
            HingePhase.RAISE,
            HingePhase.WITHDRAW,
        ):
            fraction = (
                min(1.0, self.phase_time / 6.0)
                if self.phase == HingePhase.APPROACH
                else 1.0
            )
            rotation = Rotation.from_euler("x", -np.pi / 4 * fraction) * rotation
        if self.instrument == HingeInstrument.REFRIGERATOR:
            rotation = (
                Rotation.from_rotvec(self.physics.data.xaxis[self.joint] * self.angle)
                * rotation
            )
        target = self.handle_position.copy()
        if self.phase in (HingePhase.APPROACH, HingePhase.GRASP):
            target = self.approach_position
        if self.phase in (HingePhase.CLEARANCE, HingePhase.RAISE):
            target = self.transit_position
        if self.phase == HingePhase.APPROACH and self.phase_time < 6.0:
            withdrawal = np.array([0.0, -0.10, 0.0])
            if self.instrument == HingeInstrument.CENTRIFUGE:
                withdrawal = np.array([0.0, 0.0, 0.10])
            if self.instrument == HingeInstrument.REFRIGERATOR:
                withdrawal = -rotation.apply([0.10, 0.0, 0.0])
            target += withdrawal * (1.0 - self.phase_time / 6.0)
        if self.phase == HingePhase.TURN:
            if self.instrument == HingeInstrument.CENTRIFUGE:
                self._balance_handle(elapsed)
                observed = (
                    self.hand.position
                    - self.handle_position
                    - self.grasp_rotation.apply([0.0, self.grip_adjustment, 0.0])
                )
                correction = (observed - self.grip_offset) * min(
                    1.0, self.parameters.grasp_estimation_rate * elapsed
                )
                length = float(np.linalg.norm(correction))
                maximum = self.parameters.maximum_grasp_correction * elapsed
                if length > maximum:
                    correction *= maximum / length
                self.grip_offset += correction
            if (
                self.instrument != HingeInstrument.REFRIGERATOR
                or self.contact_duration > 0.0
            ):
                self.reference_angle += float(
                    np.clip(
                        self.target_angle
                        - (
                            self.angle
                            if self.instrument == HingeInstrument.REFRIGERATOR
                            else self.reference_angle
                        ),
                        -self.angular_speed * elapsed,
                        self.angular_speed * elapsed,
                    )
                )
            reference = np.clip(
                self.reference_angle, self.angle - 0.08, self.angle + 0.08
            )
            hinge = self.physics.data.xanchor[self.joint]
            axis = self.physics.data.xaxis[self.joint]
            turn = Rotation.from_rotvec(axis * (reference - self.angle))
            target = hinge + turn.apply(target - hinge) + self.grip_offset
            if self.instrument == HingeInstrument.CENTRIFUGE:
                target = (
                    hinge
                    + turn.apply(self.handle_position - hinge)
                    + self.balanced_grip_offset
                )
            if self.instrument == HingeInstrument.REFRIGERATOR:
                self._balance_handle(elapsed)
                grip_turn = Rotation.from_rotvec(axis * (reference - self.grasp_angle))
                target = (
                    hinge
                    + turn.apply(self.handle_position - hinge)
                    + grip_turn.apply(self.balanced_grip_offset)
                )
                rotation = grip_turn * self.grasp_rotation
        if self.phase == HingePhase.RELEASE:
            target = self.release_position
            rotation = self.release_rotation
        if self.phase == HingePhase.CLEAR_HANDLE:
            target = self.withdraw_position
            rotation = self.release_rotation
        if self.phase == HingePhase.WITHDRAW:
            target = self.withdraw_position
            if self.instrument == HingeInstrument.CENTRIFUGE and self.opening:
                rotation = (
                    Rotation.from_euler("x", self.parameters.lid_withdrawal_pitch)
                    * self.release_rotation
                )
        finger = (
            self.parameters.closed_finger
            if self.phase in (HingePhase.GRASP, HingePhase.TURN)
            else self.parameters.open_finger
        )
        self.hand.advance(target, rotation, finger, elapsed)

    @property
    def balanced_grip_offset(self) -> np.ndarray:
        """Preserve the captured grasp while allowing bounded jaw-force centering."""
        return self.grip_offset + self.grasp_rotation.apply(
            [0.0, self.grip_adjustment, 0.0]
        )

    def _balance_handle(self, elapsed: float) -> None:
        """Restore opposing pressure through a bounded lateral hand intention."""
        left, right = (
            self.finger_forces.get(name, 0.0)
            for name in LaboratoryMixingProgram.FINGERTIPS
        )
        if left + right <= self.parameters.contact_force:
            return
        adjustment = (
            self.parameters.centering_speed * elapsed * (left - right) / (left + right)
        )
        self.grip_adjustment = float(
            np.clip(
                self.grip_adjustment + adjustment,
                -self.parameters.centering_limit,
                self.parameters.centering_limit,
            )
        )

    def _travel(self, elapsed: float) -> None:
        """Move the retracted robot through the frontal instrument corridor."""
        targets = {}
        for name, destination in zip(
            (RobotBaseJoint.HORIZONTAL, RobotBaseJoint.DEPTH), self.dock
        ):
            current = self.physics.joint_targets[name]
            targets[name] = float(
                np.clip(
                    destination,
                    current - elapsed * self.parameters.base_speed,
                    current + elapsed * self.parameters.base_speed,
                )
            )
        self.physics.set_joint_targets(targets)

    def _contacts(self) -> bool:
        """Measure independent fingertip forces against the authored handle."""
        fingertips = {
            self.physics.model.body(name).id: 0.0
            for name in LaboratoryMixingProgram.FINGERTIPS
        }
        for index, contact in enumerate(self.physics.data.contact):
            if self.handle not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.handle else contact.geom1
            body = int(self.physics.model.geom_bodyid[other])
            if body in fingertips:
                force = np.zeros(6)
                mujoco.mj_contactForce(
                    self.physics.model, self.physics.data, index, force
                )
                fingertips[body] += max(0.0, float(force[0]))
        self.handle_force = sum(fingertips.values())
        self.finger_forces = {
            self.physics.model.body(body).name: force
            for body, force in fingertips.items()
        }
        self.maximum_handle_force = max(self.maximum_handle_force, self.handle_force)
        return all(
            value >= self.parameters.contact_force for value in fingertips.values()
        )

    def _hand_clear(self) -> bool:
        """Require a released hand to clear every moving instrument surface."""
        model = self.physics.model
        moving = int(model.geom_bodyid[self.handle])
        hand_geometry = [
            index
            for index in range(model.ngeom)
            if model.body(model.geom_bodyid[index]).name.startswith("pr2/l_gripper")
        ]
        instrument_geometry = np.flatnonzero(model.geom_bodyid == moving)
        return all(
            mujoco.mj_geomDistance(
                model, self.physics.data, hand, int(surface), 0.015, None
            )
            >= 0.015
            for hand in hand_geometry
            for surface in instrument_geometry
        )

    def observe(self) -> None:
        """Accept progress only from measured hand contacts and hinge positions."""
        if self.state != ProgramState.RUNNING:
            return
        contact = self._contacts()
        self.contact_duration = (
            self.contact_duration + self.physics.timestep if contact else 0.0
        )
        if self.phase == HingePhase.STOW and self.phase_time >= 9.0:
            self._enter(HingePhase.TRAVEL)
        elif self.phase == HingePhase.TRAVEL:
            measured = self.physics.robot_state()
            base = np.array(
                [measured[RobotBaseJoint.HORIZONTAL], measured[RobotBaseJoint.DEPTH]]
            )
            if np.linalg.norm(base - self.dock) < self.parameters.position_tolerance:
                self._enter(
                    HingePhase.RAISE
                    if self.instrument == HingeInstrument.CENTRIFUGE
                    else (
                        HingePhase.CLEARANCE
                        if not self.opening
                        else HingePhase.APPROACH
                    )
                )
        elif self.phase == HingePhase.RAISE:
            if (
                np.linalg.norm(self.hand.position - self.transit_position)
                < self.parameters.position_tolerance
            ):
                self._enter(HingePhase.CLEARANCE)
        elif self.phase == HingePhase.CLEARANCE:
            if (
                np.linalg.norm(self.hand.position - self.transit_position)
                < self.parameters.position_tolerance
            ):
                self._enter(HingePhase.APPROACH)
        elif self.phase == HingePhase.APPROACH:
            if (
                self.phase_time > 6.0
                and np.linalg.norm(self.hand.position - self.approach_position)
                < self.parameters.position_tolerance
            ):
                self._enter(HingePhase.GRASP)
        elif self.phase == HingePhase.GRASP:
            if self.contact_duration >= self.parameters.contact_time:
                self.grasp_verified = True
                self._enter(HingePhase.TURN)
        elif self.phase == HingePhase.TURN:
            self.maximum_motor_force = max(
                self.maximum_motor_force,
                abs(float(self.physics.data.actuator_force[self.actuator])),
            )
            self.lost_contact_duration = (
                0.0 if contact else self.lost_contact_duration + self.physics.timestep
            )
            if self.lost_contact_duration > self.parameters.lost_contact_time:
                self._fail("The hand lost opposing handle contact")
            elif abs(self.angle - self.target_angle) <= self.parameters.angle_tolerance:
                self._enter(HingePhase.RELEASE)
        elif self.phase == HingePhase.RELEASE:
            if (
                self.phase_time > 3.0
                and self.handle_force < self.parameters.contact_force
            ):
                self._enter(
                    HingePhase.CLEAR_HANDLE
                    if self.instrument == HingeInstrument.CENTRIFUGE and self.opening
                    else HingePhase.WITHDRAW
                )
        elif self.phase == HingePhase.CLEAR_HANDLE:
            if (
                self.phase_time > 1.0
                and np.linalg.norm(self.hand.position - self.withdraw_position)
                < self.parameters.handle_clearance_tolerance
            ):
                self._enter(HingePhase.WITHDRAW)
        elif self.phase == HingePhase.WITHDRAW:
            if self.phase_time > 1.0 and self._hand_clear():
                self.phase = HingePhase.COMPLETE
                self.state = ProgramState.SUCCEEDED
                self._restore_actuators(self.target_angle)
        if self.phase_time > self.parameters.phase_timeout:
            self._fail(f"The robot did not complete the {self.phase.value} movement")

    def _fail(self, reason: str) -> None:
        """Hold the observed instrument position after a failed manipulation."""
        self._restore_actuators()
        self.error = reason
        self.state = ProgramState.FAILED

    def snapshot(self) -> dict[str, str | float | bool | None]:
        """Publish physical cover and hand progress."""
        return {
            HingeField.STATE: self.state,
            HingeField.PHASE: self.phase,
            HingeField.INSTRUMENT: self.instrument,
            HingeField.OPENING: self.opening,
            HingeField.ANGLE: self.angle,
            HingeField.TARGET: self.target_angle,
            HingeField.FORCE: self.handle_force,
            HingeField.ERROR: self.error,
        }
