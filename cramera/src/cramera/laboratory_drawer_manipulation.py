"""Contact-driven PR2 operation of the sample-storage drawer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cramera.laboratory_drawer import DrawerGeometry
from cramera.laboratory_hand_motion import LaboratoryHandMotion
from cramera.laboratory_mixing_program import LaboratoryMixingProgram
from cramera.laboratory_physics_program import JointReference, ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics, RobotBaseJoint
from cramera.laboratory_robot_base import LaboratoryBasePath


# %% drawer manipulation contract
class DrawerPhase(StrEnum):
    """Measured stages of one robot-operated drawer movement."""

    STOW = "stow"
    """Retract the empty hand before base travel."""
    TRAVEL = "travel"
    """Move through the checked frontal corridor."""
    CLEARANCE = "clearance"
    """Lower the hand in front of the drawer face."""
    APPROACH = "approach"
    """Align open fingers with the measured pull bar."""
    GRASP = "grasp"
    """Require sustained opposing handle contacts."""
    SLIDE = "slide"
    """Pull or push the drawer through the held handle."""
    RELEASE = "release"
    """Hold the measured endpoint while opening the fingers."""
    WITHDRAW = "withdraw"
    """Clear the drawer before ending the action."""
    COMPLETE = "complete"
    """The drawer reached its endpoint and the hand is clear."""


class DrawerManipulationField(StrEnum):
    """Measured robot drawer-action feedback."""

    STATE = "state"
    """Execution lifecycle."""
    PHASE = "phase"
    """Current manipulation stage."""
    OPENING = "opening"
    """Whether the robot is opening the drawer."""
    POSITION = "position"
    """Measured prismatic displacement in metres."""
    TARGET = "targetPosition"
    """Requested drawer endpoint in metres."""
    FORCE = "handleForce"
    """Sum of measured fingertip normal forces in Newtons."""
    ERROR = "error"
    """Physical failure explanation."""


@dataclass(frozen=True)
class DrawerManipulationParameters:
    """Bounded physical references and measured acceptance thresholds."""

    period: float = 0.01
    """Cartesian control interval in simulated seconds."""
    slide_speed: float = 0.025
    """Maximum handle translation reference speed in metres per second."""
    base_speed: float = 0.08
    """Maximum base reference speed in metres per second."""
    finger_torque: float = 1.5
    """Force-limited finger torque during the robust handle grasp."""
    finger_stiffness: float = 20.0
    """Finger position gain for the rigid pull bar, in Newton-metres per radian."""
    open_finger: float = 0.5
    """Finger opening demand providing clearance around the pull bar."""
    closed_finger: float = 0.0
    """Contact-limited closure demand."""
    contact_force: float = 0.02
    """Minimum opposing fingertip normal load in Newtons."""
    contact_time: float = 0.15
    """Continuous opposing contact required before motor release, in seconds."""
    lost_contact_time: float = 0.5
    """Maximum interruption of the confirmed grasp, in seconds."""
    position_tolerance: float = 0.004
    """Required drawer endpoint accuracy in metres."""
    release_tolerance: float = 0.002
    """Endpoint accuracy before finger release, leaving margin for contact settling."""
    hand_tolerance: float = 0.006
    """Required hand waypoint accuracy in metres."""
    reference_lead: float = 0.02
    """Maximum hand reference lead over measured drawer travel, in metres."""
    phase_timeout: float = 45.0
    """Maximum physical settling time for one stage, in seconds."""
    stow_time: float = 8.0
    """Duration of the initial bounded arm retraction, in seconds."""
    approach_time: float = 6.0
    """Duration of the final frontal approach, in seconds."""
    approach_distance: float = 0.12
    """Front-side waypoint distance from the handle, in metres."""
    centering_speed: float = 0.004
    """Maximum jaw-force centering displacement rate, in metres per second."""
    centering_limit: float = 0.003
    """Maximum centering displacement from the captured grasp, in metres."""
    dock: tuple[float, float] = (-0.32, -0.37)
    """Checked planar base displacement for the full drawer travel."""


@dataclass
class LaboratoryDrawerManipulation:
    """Open or close the drawer through measured front-grasp friction contacts."""

    physics: LaboratoryRobotPhysics
    """Authoritative articulated laboratory physics."""
    opening: bool
    """Whether to pull the drawer open or push it closed."""
    parameters: DrawerManipulationParameters = field(
        default_factory=DrawerManipulationParameters
    )
    """Conservative physical references and acceptance bounds."""
    state: ProgramState = field(init=False, default=ProgramState.IDLE)
    """Execution lifecycle independent of physics stepping."""
    phase: DrawerPhase = field(init=False, default=DrawerPhase.STOW)
    """Current measured manipulation stage."""
    error: str | None = field(init=False, default=None)
    """Physical reason for unsuccessful completion."""
    hand: LaboratoryHandMotion = field(init=False)
    """Measured Cartesian hand servo."""
    handle: int = field(init=False)
    """Collision geometry of the drawer pull bar."""
    target_position: float = field(init=False)
    """Requested prismatic endpoint in metres."""
    reference_position: float = field(init=False)
    """Bounded hand-driven drawer displacement reference."""
    rotation: Rotation = field(init=False)
    """Frontal approach orientation with vertically opposing jaws."""
    finger_range: np.ndarray = field(init=False)
    """Original glass-safe finger torque limits."""
    finger_gain: np.ndarray = field(init=False)
    """Original glass-safe finger actuator gain coefficients."""
    finger_bias: np.ndarray = field(init=False)
    """Original glass-safe finger actuator bias coefficients."""
    start_joints: dict[str, float] = field(init=False, default_factory=dict)
    """Measured arm configuration before retraction."""
    phase_time: float = field(init=False, default=0.0)
    """Active simulated seconds in the current stage."""
    elapsed: float = field(init=False, default=0.0)
    """Total active simulated duration."""
    control_time: float = field(init=False, default=0.0)
    """Accumulated Cartesian control interval."""
    contact_duration: float = field(init=False, default=0.0)
    """Continuous opposing handle contact duration."""
    lost_contact_duration: float = field(init=False, default=0.0)
    """Duration of an interrupted confirmed grasp."""
    grasp_verified: bool = field(init=False, default=False)
    """Whether sustained opposing handle contacts were measured."""
    moved: bool = field(init=False, default=False)
    """Whether the drawer required physical movement."""
    handle_force: float = field(init=False, default=0.0)
    """Current combined fingertip normal load on the handle."""
    maximum_handle_force: float = field(init=False, default=0.0)
    """Largest measured combined fingertip load."""
    maximum_motor_force: float = field(init=False, default=0.0)
    """Largest measured drawer motor force during hand-driven travel."""
    finger_forces: dict[str, float] = field(init=False, default_factory=dict)
    """Measured normal loads by fingertip body."""
    grip_offset: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    """Measured tool-minus-handle displacement at grasp confirmation."""
    grip_adjustment: float = field(init=False, default=0.0)
    """Bounded jaw-force centering correction."""
    waypoint: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    """Fixed Cartesian destination of the current clearance or release stage."""

    def __post_init__(self) -> None:
        """Resolve the physical pull bar and the drawer's mechanical range."""
        self.hand = LaboratoryHandMotion(self.physics, shoulder_posture=0.3)
        self.handle = self.physics.model.geom(DrawerGeometry.HANDLE).id
        self.target_position = (
            self.physics.drawer.open_position if self.opening else 0.0
        )
        self.reference_position = self.position
        self.rotation = Rotation.from_matrix(
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        finger = self.physics.model.actuator(JointReference.GRIPPER_JOINT).id
        self.finger_range = self.physics.model.actuator_forcerange[finger].copy()
        self.finger_gain = self.physics.model.actuator_gainprm[finger].copy()
        self.finger_bias = self.physics.model.actuator_biasprm[finger].copy()

    @property
    def position(self) -> float:
        """Measured drawer displacement in metres."""
        return self.physics.drawer.position

    @property
    def handle_position(self) -> np.ndarray:
        """Measured world centre of the authored handle collision bar."""
        return self.physics.data.geom_xpos[self.handle].copy()

    def start(self) -> None:
        """Begin or resume measured drawer manipulation without changing poses."""
        if self.state not in (ProgramState.IDLE, ProgramState.PAUSED):
            return
        if self.state == ProgramState.IDLE:
            if (
                abs(self.position - self.target_position)
                <= self.parameters.position_tolerance
            ):
                self.state = ProgramState.SUCCEEDED
                self.phase = DrawerPhase.COMPLETE
                return
            self.moved = True
            self.start_joints = self.physics.robot_state()
            finger = self.physics.model.actuator(JointReference.GRIPPER_JOINT).id
            self.physics.model.actuator_forcerange[finger] = (
                -self.parameters.finger_torque,
                self.parameters.finger_torque,
            )
            self.physics.model.actuator_gainprm[finger, 0] = (
                self.parameters.finger_stiffness
            )
            self.physics.model.actuator_biasprm[finger, 1] = (
                -self.parameters.finger_stiffness
            )
            self.physics.drawer.restore_hold(self.position)
        elif self.phase == DrawerPhase.SLIDE:
            self.physics.drawer.release_to_hand()
        self.state = ProgramState.RUNNING

    def pause(self) -> None:
        """Hold the measured drawer and robot references while execution is paused."""
        if self.state == ProgramState.RUNNING:
            self.physics.drawer.restore_hold(self.position)
            self.state = ProgramState.PAUSED

    def cancel(self) -> None:
        """Restore bounded holding and the original finger torque after cancellation."""
        self._restore_actuators()
        if self.state in (ProgramState.RUNNING, ProgramState.PAUSED):
            self.state = ProgramState.FAILED
            self.error = "Drawer manipulation was cancelled"

    def _restore_actuators(self) -> None:
        """Hold the measured drawer and restore glass-safe finger torque limits."""
        self.physics.drawer.restore_hold(self.position)
        finger = self.physics.model.actuator(JointReference.GRIPPER_JOINT).id
        self.physics.model.actuator_forcerange[finger] = self.finger_range
        self.physics.model.actuator_gainprm[finger] = self.finger_gain
        self.physics.model.actuator_biasprm[finger] = self.finger_bias

    def _enter(self, phase: DrawerPhase) -> None:
        """Capture measured references for the next manipulation stage."""
        self.phase = phase
        self.phase_time = self.contact_duration = self.lost_contact_duration = 0.0
        if phase == DrawerPhase.TRAVEL:
            targets = dict(
                zip(
                    (RobotBaseJoint.HORIZONTAL, RobotBaseJoint.DEPTH),
                    self.parameters.dock,
                )
            )
            obstruction = LaboratoryBasePath(self.physics).obstruction(targets)
            if obstruction is not None:
                self._fail(str(obstruction))
        elif phase == DrawerPhase.CLEARANCE:
            self.waypoint = self.handle_position - np.array(
                [0.0, self.parameters.approach_distance, 0.0]
            )
        elif phase == DrawerPhase.SLIDE:
            self.reference_position = self.position
            self.grip_offset = self.hand.position - self.handle_position
            self.rotation = self.hand.rotation
            self.physics.drawer.release_to_hand()
        elif phase == DrawerPhase.RELEASE:
            self.waypoint = self.hand.position
            self.physics.drawer.restore_hold(self.position)
        elif phase == DrawerPhase.WITHDRAW:
            self.waypoint = self.hand.position + np.array([0.0, -0.12, 0.04])

    def advance(self) -> None:
        """Issue bounded robot actuator targets while physical contacts move the drawer."""
        if self.state != ProgramState.RUNNING:
            return
        self.phase_time += self.physics.timestep
        self.elapsed += self.physics.timestep
        self.control_time += self.physics.timestep
        if self.control_time < self.parameters.period:
            return
        elapsed = self.control_time
        self.control_time = 0.0
        if self.phase == DrawerPhase.STOW:
            fraction = min(1.0, self.phase_time / self.parameters.stow_time)
            fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            targets = {
                name: float(
                    np.clip(
                        self.start_joints[name]
                        + fraction
                        * (
                            self.physics.initial_positions[name]
                            - self.start_joints[name]
                        ),
                        self.physics.robot_joints[name].lower,
                        self.physics.robot_joints[name].upper,
                    )
                )
                for name in self.hand.names
            }
            targets[JointReference.GRIPPER_JOINT] = self.parameters.open_finger
            self.physics.set_joint_targets(targets)
            return
        if self.phase == DrawerPhase.TRAVEL:
            targets = {
                name: float(
                    np.clip(
                        destination,
                        self.physics.joint_targets[name]
                        - elapsed * self.parameters.base_speed,
                        self.physics.joint_targets[name]
                        + elapsed * self.parameters.base_speed,
                    )
                )
                for name, destination in zip(
                    (RobotBaseJoint.HORIZONTAL, RobotBaseJoint.DEPTH),
                    self.parameters.dock,
                )
            }
            self.physics.set_joint_targets(targets)
            return
        torso = "pr2/torso_lift_joint"
        self.physics.set_joint_targets(
            {
                torso: max(
                    self.physics.robot_joints[torso].lower,
                    self.physics.joint_targets[torso] - 0.03 * elapsed,
                )
            }
        )
        target = self.handle_position
        if self.phase in (
            DrawerPhase.CLEARANCE,
            DrawerPhase.RELEASE,
            DrawerPhase.WITHDRAW,
        ):
            target = self.waypoint
        elif self.phase == DrawerPhase.APPROACH:
            fraction = max(0.0, 1 - self.phase_time / self.parameters.approach_time)
            target -= np.array([0.0, self.parameters.approach_distance * fraction, 0.0])
        elif self.phase == DrawerPhase.SLIDE:
            self.reference_position += float(
                np.clip(
                    self.target_position - self.reference_position,
                    -self.parameters.slide_speed * elapsed,
                    self.parameters.slide_speed * elapsed,
                )
            )
            reference = float(
                np.clip(
                    self.reference_position,
                    self.position - self.parameters.reference_lead,
                    self.position + self.parameters.reference_lead,
                )
            )
            left, right = (
                self.finger_forces.get(name, 0.0)
                for name in LaboratoryMixingProgram.FINGERTIPS
            )
            if left + right > self.parameters.contact_force:
                self.grip_adjustment = float(
                    np.clip(
                        self.grip_adjustment
                        + self.parameters.centering_speed
                        * elapsed
                        * (left - right)
                        / (left + right),
                        -self.parameters.centering_limit,
                        self.parameters.centering_limit,
                    )
                )
            axis = self.physics.data.xaxis[self.physics.drawer.joint]
            target += (
                axis * (reference - self.position)
                + self.grip_offset
                + self.rotation.apply([0.0, self.grip_adjustment, 0.0])
            )
        finger = (
            self.parameters.closed_finger
            if self.phase in (DrawerPhase.GRASP, DrawerPhase.SLIDE)
            else self.parameters.open_finger
        )
        self.hand.advance(target, self.rotation, finger, elapsed)

    def _contacts(self) -> bool:
        """Measure the independent fingertip normal loads on the drawer pull."""
        forces = {
            self.physics.model.body(name).id: 0.0
            for name in LaboratoryMixingProgram.FINGERTIPS
        }
        for index, contact in enumerate(self.physics.data.contact):
            if self.handle not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.handle else contact.geom1
            body = int(self.physics.model.geom_bodyid[other])
            if body in forces:
                force = np.zeros(6)
                mujoco.mj_contactForce(
                    self.physics.model, self.physics.data, index, force
                )
                forces[body] += max(0.0, float(force[0]))
        self.handle_force = sum(forces.values())
        self.maximum_handle_force = max(self.maximum_handle_force, self.handle_force)
        self.finger_forces = {
            self.physics.model.body(body).name: force for body, force in forces.items()
        }
        return all(force >= self.parameters.contact_force for force in forces.values())

    def _hand_clear(self) -> bool:
        """Require all gripper geometry to clear the released drawer."""
        model = self.physics.model
        body = int(model.geom_bodyid[self.handle])
        hand = [
            index
            for index in range(model.ngeom)
            if model.body(model.geom_bodyid[index]).name.startswith("pr2/l_gripper")
        ]
        surfaces = np.flatnonzero(model.geom_bodyid == body)
        return all(
            mujoco.mj_geomDistance(
                model, self.physics.data, finger, int(surface), 0.015, None
            )
            >= 0.015
            for finger in hand
            for surface in surfaces
        )

    def observe(self) -> None:
        """Advance only after measured grasp, slide and clearance conditions hold."""
        if self.state != ProgramState.RUNNING:
            return
        contact = self._contacts()
        self.contact_duration = (
            self.contact_duration + self.physics.timestep if contact else 0.0
        )
        if (
            self.phase == DrawerPhase.STOW
            and self.phase_time >= self.parameters.stow_time + 1.0
        ):
            self._enter(DrawerPhase.TRAVEL)
        elif self.phase == DrawerPhase.TRAVEL:
            measured = self.physics.robot_state()
            base = np.array(
                [measured[RobotBaseJoint.HORIZONTAL], measured[RobotBaseJoint.DEPTH]]
            )
            if (
                np.linalg.norm(base - self.parameters.dock)
                < self.parameters.hand_tolerance
            ):
                self._enter(DrawerPhase.CLEARANCE)
        elif self.phase == DrawerPhase.CLEARANCE:
            if (
                np.linalg.norm(self.hand.position - self.waypoint)
                < self.parameters.hand_tolerance
            ):
                self._enter(DrawerPhase.APPROACH)
        elif self.phase == DrawerPhase.APPROACH:
            if (
                self.phase_time >= self.parameters.approach_time
                and np.linalg.norm(self.hand.position - self.handle_position)
                < self.parameters.hand_tolerance
            ):
                self._enter(DrawerPhase.GRASP)
        elif self.phase == DrawerPhase.GRASP:
            if self.contact_duration >= self.parameters.contact_time:
                self.grasp_verified = True
                self._enter(DrawerPhase.SLIDE)
        elif self.phase == DrawerPhase.SLIDE:
            self.maximum_motor_force = max(
                self.maximum_motor_force,
                abs(
                    float(
                        self.physics.data.actuator_force[self.physics.drawer.actuator]
                    )
                ),
            )
            self.lost_contact_duration = (
                0.0 if contact else self.lost_contact_duration + self.physics.timestep
            )
            if self.lost_contact_duration > self.parameters.lost_contact_time:
                self._fail("The hand lost opposing drawer-handle contact")
            elif (
                abs(self.position - self.target_position)
                <= self.parameters.release_tolerance
            ):
                self._enter(DrawerPhase.RELEASE)
        elif self.phase == DrawerPhase.RELEASE:
            if (
                self.phase_time > 3.0
                and self.handle_force < self.parameters.contact_force
            ):
                self._enter(DrawerPhase.WITHDRAW)
        elif self.phase == DrawerPhase.WITHDRAW:
            if (
                np.linalg.norm(self.hand.position - self.waypoint)
                < self.parameters.hand_tolerance
                and self._hand_clear()
            ):
                if abs(self.position - self.target_position) > (
                    self.parameters.position_tolerance
                ):
                    self._fail("The drawer moved away from its endpoint after release")
                    return
                self.state = ProgramState.SUCCEEDED
                self.phase = DrawerPhase.COMPLETE
                self._restore_actuators()
        if self.phase_time > self.parameters.phase_timeout:
            self._fail(
                f"The robot did not complete the drawer {self.phase.value} movement"
            )

    def _fail(self, reason: str) -> None:
        """Restore bounded holding after a physical acceptance failure."""
        self._restore_actuators()
        self.error = reason
        self.state = ProgramState.FAILED

    def snapshot(self) -> dict[str, str | bool | float | None]:
        """Describe the current drawer operation using measured physical state."""
        return {
            DrawerManipulationField.STATE: self.state,
            DrawerManipulationField.PHASE: self.phase,
            DrawerManipulationField.OPENING: self.opening,
            DrawerManipulationField.POSITION: self.position,
            DrawerManipulationField.TARGET: self.target_position,
            DrawerManipulationField.FORCE: self.handle_force,
            DrawerManipulationField.ERROR: self.error,
        }
