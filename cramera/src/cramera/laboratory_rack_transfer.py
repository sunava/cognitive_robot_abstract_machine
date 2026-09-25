"""Contact-driven transport of a loaded sample rack from drawer to workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cramera.laboratory_hand_motion import LaboratoryHandMotion
from cramera.laboratory_mixing_program import LaboratoryMixingProgram
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import JointReference, ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics, RobotBaseJoint
from cramera.laboratory_rack import RackGeometry
from cramera.laboratory_robot_base import LaboratoryBasePath
from cramera.laboratory_world import LaboratoryBody


# %% transport contract
class RackTransferPhase(StrEnum):
    """Physical stages of retrieving a rack with independently supported samples."""

    STOW = "stow"
    """Retract the empty arm before changing the base position."""
    DOCK = "dock"
    """Reach the open drawer through a checked base path."""
    CLEARANCE = "clearance"
    """Align the open hand above the carry bar."""
    APPROACH = "approach"
    """Descend onto the handle with horizontally opposing jaws."""
    GRASP = "grasp"
    """Require opposing physical contacts before lifting."""
    LIFT = "lift"
    """Raise the upright rack clear of the drawer and workbench."""
    TRAVEL = "travel"
    """Translate the loaded carrier through the free frontal corridor."""
    DOCK_BENCH = "dock_bench"
    """Advance toward the bench with the carried rack above its surface."""
    ALIGN = "align"
    """Position the carrier above its workbench destination."""
    PLACE = "place"
    """Lower until the bench supports the rack and all samples remain seated."""
    RELEASE = "release"
    """Open the gripper while the bench supports the carrier."""
    WITHDRAW = "withdraw"
    """Raise the empty fingertips clear of the handle."""
    COMPLETE = "complete"
    """All required samples remain supported on the placed rack."""


class RackTransferField(StrEnum):
    """Measured execution feedback for the loaded carrier."""

    STATE = "state"
    """Execution lifecycle."""
    PHASE = "phase"
    """Current measured transport stage."""
    POSITION = "position"
    """Measured rack origin."""
    FORCE = "handleForce"
    """Measured opposing fingertip normal load."""
    ERROR = "error"
    """Physical failure explanation."""


@dataclass(frozen=True)
class RackTransferParameters:
    """Conservative actuator references and measured acceptance tolerances."""

    period: float = 0.01
    """Cartesian control period in simulated seconds."""
    speed: float = 0.04
    """Maximum Cartesian reference speed in metres per second."""
    base_speed: float = 0.08
    """Maximum planar base reference speed in metres per second."""
    base_check_period: float = 0.05
    """Interval between measured obstacle previews in simulated seconds."""
    base_lookahead: float = 0.04
    """Distance checked ahead of a moving base in metres."""
    bench_depth: float = 0.0
    """Planar depth displacement placing the bench within the arm workspace."""
    torso_speed: float = 0.03
    """Maximum torso reference speed during empty-hand recovery in metres per second."""
    dock: tuple[float, float] = (-0.7, -0.37)
    """Planar base displacement giving access to the open drawer."""
    corridor_depth: float = -0.50
    """Rack centre depth while passing in front of the workbench."""
    transit_height: float = 1.04
    """Rack base height clearing the worktop during transport."""
    approach_clearance: float = 0.30
    """Vertical empty-hand clearance above the carry bar."""
    open_finger: float = 0.25
    """Open jaw target for approach and release."""
    closed_finger: float = 0.0
    """Contact-limited closure demand while carrying the robust handle."""
    finger_torque: float = 3.0
    """Bounded finger torque while carrying the robust rack handle."""
    finger_stiffness: float = 60.0
    """Finger stiffness in Newton-metres per radian during carrier transport."""
    contact_force: float = 0.02
    """Minimum force on each opposing fingertip in Newtons."""
    contact_time: float = 0.20
    """Continuous opposing grasp evidence required before lifting, in seconds."""
    lost_contact_time: float = 0.5
    """Maximum interruption of a confirmed grasp in seconds."""
    position_tolerance: float = 0.006
    """Final carrier placement tolerance in metres."""
    hand_tolerance: float = 0.006
    """Hand waypoint tolerance in metres."""
    upright_tolerance: float = 0.15
    """Maximum rack tilt during loaded transport in radians."""
    sample_tolerance: float = 0.012
    """Maximum sample-root displacement from its moving slot in metres."""
    phase_timeout: float = 45.0
    """Maximum settling duration for a physical transport stage in seconds."""
    stow_time: float = 8.0
    """Smooth empty-arm retraction duration in seconds."""
    stow_settling_time: float = 1.0
    """Additional settling after reaching the empty-hand posture, in seconds."""
    release_time: float = 3.0
    """Minimum supported release duration in seconds."""
    withdrawal_height: float = 0.15
    """Vertical clearance above the released carry bar in metres."""


# %% measured rack transport
@dataclass
class LaboratoryRackTransfer:
    """Carry an upright rack through bounded robot actuators and friction contacts."""

    physics: LaboratoryRobotPhysics
    """Authoritative articulated laboratory physics."""
    required_sample_keys: tuple[str, ...]
    """Independent vessels that must remain seated throughout transport."""
    parameters: RackTransferParameters = field(default_factory=RackTransferParameters)
    """Motion limits and acceptance thresholds."""
    state: ProgramState = field(init=False, default=ProgramState.IDLE)
    """Execution lifecycle."""
    phase: RackTransferPhase = field(init=False, default=RackTransferPhase.STOW)
    """Current measured transport stage."""
    error: str | None = field(init=False, default=None)
    """Physical failure explanation."""
    hand: LaboratoryHandMotion = field(init=False)
    """Cartesian servo for the left gripper."""
    handle: int = field(init=False)
    """Authored carry-bar collision geometry."""
    destination: np.ndarray = field(init=False)
    """Authored workbench position of the rack root."""
    rotation: Rotation = field(init=False)
    """Top-approach gripper orientation and subsequent measured grasp orientation."""
    grip_offset: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    """Hand offset in the rack frame measured when the grasp is accepted."""
    target: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    """Final hand position for the current Cartesian stage."""
    reference: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    """Bounded Cartesian hand reference."""
    base_target: np.ndarray = field(init=False, default_factory=lambda: np.zeros(2))
    """Final planar displacement for the current travel stage."""
    base_path: LaboratoryBasePath | None = field(init=False, default=None)
    """Collision sweep including the carried rack and all samples."""
    base_check_time: float = field(init=False, default=0.0)
    """Elapsed time since the most recent dynamic base check."""
    start_joints: dict[str, float] = field(init=False, default_factory=dict)
    """Measured configuration before initial arm retraction."""
    finger_range: np.ndarray = field(init=False)
    """Original glass-safe finger torque limits."""
    finger_gain: np.ndarray = field(init=False)
    """Original finger actuator gains."""
    finger_bias: np.ndarray = field(init=False)
    """Original finger actuator bias parameters."""
    elapsed: float = field(init=False, default=0.0)
    """Active simulated duration."""
    phase_time: float = field(init=False, default=0.0)
    """Active simulated duration of the current stage."""
    control_time: float = field(init=False, default=0.0)
    """Accumulated Cartesian control period."""
    contact_duration: float = field(init=False, default=0.0)
    """Continuous opposing grasp evidence in seconds."""
    lost_contact_duration: float = field(init=False, default=0.0)
    """Duration of an interrupted confirmed grasp."""
    grasp_verified: bool = field(init=False, default=False)
    """Whether opposing carry-bar contacts were confirmed."""
    handle_force: float = field(init=False, default=0.0)
    """Current sum of opposing fingertip normal loads."""
    maximum_handle_force: float = field(init=False, default=0.0)
    """Largest observed carry-bar normal load."""

    def __post_init__(self) -> None:
        """Resolve the carrier and capture actuator settings without changing poses."""
        self.hand = LaboratoryHandMotion(self.physics, shoulder_posture=0.3)
        self.handle = self.physics.model.geom(RackGeometry.HANDLE).id
        self.destination = self.physics.model.body_pos[self.physics.rack.body].copy()
        self.rotation = Rotation.from_matrix(
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        )
        finger = self.physics.model.actuator(JointReference.GRIPPER_JOINT).id
        self.finger_range = self.physics.model.actuator_forcerange[finger].copy()
        self.finger_gain = self.physics.model.actuator_gainprm[finger].copy()
        self.finger_bias = self.physics.model.actuator_biasprm[finger].copy()

    @property
    def position(self) -> np.ndarray:
        """Measured rack-root position in world coordinates."""
        return self.physics.data.xpos[self.physics.rack.body].copy()

    @property
    def handle_position(self) -> np.ndarray:
        """Measured carry-bar centre in world coordinates."""
        return self.physics.data.geom_xpos[self.handle].copy()

    def start(self) -> None:
        """Begin or resume carrying while preserving all dynamic coordinates."""
        if self.state not in (ProgramState.IDLE, ProgramState.PAUSED):
            return
        if self.state == ProgramState.IDLE:
            if not self.physics.rack.movable:
                raise InvalidPhysicsTarget(
                    "The sample rack is not independently movable"
                )
            if any(
                key not in self.physics.rack.assignments
                for key in self.required_sample_keys
            ):
                raise InvalidPhysicsTarget(
                    "The rack transport references an unknown sample"
                )
            if self._placed():
                self.state = ProgramState.SUCCEEDED
                self.phase = RackTransferPhase.COMPLETE
                return
            if (
                self.physics.drawer.position
                < self.physics.drawer.open_position - self.parameters.position_tolerance
            ):
                raise InvalidPhysicsTarget(
                    "Open the drawer fully before retrieving the rack"
                )
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
        self.state = ProgramState.RUNNING

    def pause(self) -> None:
        """Retain the currently held actuator references during a pause."""
        if self.state == ProgramState.RUNNING:
            self.state = ProgramState.PAUSED

    def cancel(self) -> None:
        """Restore the ordinary finger profile when physical ownership ends."""
        self._restore_actuators()
        if self.state in (ProgramState.RUNNING, ProgramState.PAUSED):
            self.state = ProgramState.FAILED
            self.error = "Rack transfer was cancelled"

    def _restore_actuators(self) -> None:
        """Restore the glass-safe finger gain, bias and bounded torque."""
        finger = self.physics.model.actuator(JointReference.GRIPPER_JOINT).id
        self.physics.model.actuator_forcerange[finger] = self.finger_range
        self.physics.model.actuator_gainprm[finger] = self.finger_gain
        self.physics.model.actuator_biasprm[finger] = self.finger_bias

    def _enter(self, phase: RackTransferPhase) -> None:
        """Set the next intention from the measured hand and carrier poses."""
        self.phase = phase
        self.phase_time = self.contact_duration = self.lost_contact_duration = 0.0
        self.reference = self.hand.position
        if phase in (
            RackTransferPhase.DOCK,
            RackTransferPhase.TRAVEL,
            RackTransferPhase.DOCK_BENCH,
        ):
            measured = self.physics.robot_state()
            self.base_target = np.asarray(self.parameters.dock).copy()
            held = ()
            if phase in (RackTransferPhase.TRAVEL, RackTransferPhase.DOCK_BENCH):
                self.base_target = np.array(
                    [
                        measured[RobotBaseJoint.HORIZONTAL]
                        + (
                            self.destination[0] - self.position[0]
                            if phase == RackTransferPhase.TRAVEL
                            else 0.0
                        ),
                        (
                            measured[RobotBaseJoint.DEPTH]
                            if phase == RackTransferPhase.TRAVEL
                            else self.parameters.bench_depth
                        ),
                    ]
                )
                held = (RackGeometry.BODY, *self.required_sample_keys)
            self.base_path = LaboratoryBasePath(self.physics, held_objects=held)
            self.base_path.validate(
                dict(
                    zip(
                        (RobotBaseJoint.HORIZONTAL, RobotBaseJoint.DEPTH),
                        self.base_target,
                    )
                )
            )
        elif phase == RackTransferPhase.CLEARANCE:
            self.target = self.handle_position + [
                0.0,
                0.0,
                self.parameters.approach_clearance,
            ]
        elif phase == RackTransferPhase.APPROACH:
            self.target = self.handle_position
        elif phase == RackTransferPhase.LIFT:
            rack_rotation = Rotation.from_matrix(
                self.physics.data.xmat[self.physics.rack.body].reshape(3, 3)
            )
            self.grip_offset = rack_rotation.inv().apply(
                self.hand.position - self.position
            )
            self.rotation = rack_rotation.inv() * self.hand.rotation
            self.target = (
                np.array(
                    [
                        self.position[0],
                        self.parameters.corridor_depth,
                        self.parameters.transit_height,
                    ]
                )
                + self.grip_offset
            )
        elif phase == RackTransferPhase.ALIGN:
            self.target = (
                self.destination
                + [0.0, 0.0, self.parameters.transit_height - self.destination[2]]
                + self.grip_offset
            )
        elif phase == RackTransferPhase.PLACE:
            self.target = self.destination + self.grip_offset
        elif phase == RackTransferPhase.RELEASE:
            self.target = self.hand.position
        elif phase == RackTransferPhase.WITHDRAW:
            self.target = self.hand.position + [
                0.0,
                0.0,
                self.parameters.withdrawal_height,
            ]

    def advance(self) -> None:
        """Issue joint references while all carrier and vessel motion remains physical."""
        if self.state != ProgramState.RUNNING:
            return
        self.elapsed += self.physics.timestep
        self.phase_time += self.physics.timestep
        self.control_time += self.physics.timestep
        if self.control_time < self.parameters.period:
            return
        elapsed = self.control_time
        self.control_time = 0.0
        if self.phase == RackTransferPhase.STOW:
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
            torso = "pr2/torso_lift_joint"
            targets[torso] = float(
                np.clip(
                    self.physics.initial_positions[torso],
                    self.physics.joint_targets[torso]
                    - self.parameters.torso_speed * elapsed,
                    self.physics.joint_targets[torso]
                    + self.parameters.torso_speed * elapsed,
                )
            )
            self.physics.set_joint_targets(targets)
            return
        if self.phase in (
            RackTransferPhase.DOCK,
            RackTransferPhase.TRAVEL,
            RackTransferPhase.DOCK_BENCH,
        ):
            self._advance_base(elapsed)
            return
        direction = self.target - self.reference
        self.reference += direction * min(
            1.0, self.parameters.speed * elapsed / max(np.linalg.norm(direction), 1e-12)
        )
        closed = self.phase in (
            RackTransferPhase.GRASP,
            RackTransferPhase.LIFT,
            RackTransferPhase.ALIGN,
            RackTransferPhase.PLACE,
        )
        self.hand.advance(
            self.reference,
            self.rotation,
            self.parameters.closed_finger if closed else self.parameters.open_finger,
            elapsed,
        )

    def _advance_base(self, elapsed: float) -> None:
        """Check the moving assembly ahead of bounded planar base references."""
        names = (RobotBaseJoint.HORIZONTAL, RobotBaseJoint.DEPTH)
        self.base_check_time += elapsed
        if self.base_check_time >= self.parameters.base_check_period:
            measured = self.physics.robot_state()
            preview = {
                name: measured[name]
                + float(
                    np.clip(
                        value - measured[name],
                        -self.parameters.base_lookahead,
                        self.parameters.base_lookahead,
                    )
                )
                for name, value in zip(names, self.base_target)
            }
            self.base_path.validate(preview)
            self.base_check_time = 0.0
        targets = {
            name: float(
                np.clip(
                    value,
                    self.physics.joint_targets[name]
                    - self.parameters.base_speed * elapsed,
                    self.physics.joint_targets[name]
                    + self.parameters.base_speed * elapsed,
                )
            )
            for name, value in zip(names, self.base_target)
        }
        self.physics.set_joint_targets(targets)

    def _contacts(self) -> bool:
        """Measure independent normal loads from both fingertips on the carry bar."""
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
        return all(force >= self.parameters.contact_force for force in forces.values())

    def _samples_retained(self) -> bool:
        """Check every independent sample against its measured moving slot."""
        if self.physics.data.xmat[self.physics.rack.body].reshape(3, 3)[2, 2] < np.cos(
            self.parameters.upright_tolerance
        ):
            return False
        for key in self.required_sample_keys:
            body = self.physics.model.body(key).id
            if (
                np.linalg.norm(
                    self.physics.data.xpos[body] - self.physics.rack.slot_position(key)
                )
                > self.parameters.sample_tolerance
            ):
                return False
            if self.physics.data.xmat[body].reshape(3, 3)[2, 2] < np.cos(
                self.parameters.upright_tolerance
            ):
                return False
        return True

    def _placed(self) -> bool:
        """Require correct pose, bench support and separately supported sample contents."""
        occupants = self.physics.rack.occupants()
        return (
            np.linalg.norm(self.position - self.destination)
            < self.parameters.position_tolerance
            and self.physics.rack.supported_by(LaboratoryBody.ROOM)
            and all(
                occupants[self.physics.rack.assignments[key]] == key
                for key in self.required_sample_keys
            )
        )

    def observe(self) -> None:
        """Advance only after measured grasp, retained samples and support agree."""
        if self.state != ProgramState.RUNNING:
            return
        contact = self._contacts()
        self.contact_duration = (
            self.contact_duration + self.physics.timestep if contact else 0.0
        )
        carrying = self.phase in (
            RackTransferPhase.LIFT,
            RackTransferPhase.TRAVEL,
            RackTransferPhase.DOCK_BENCH,
            RackTransferPhase.ALIGN,
            RackTransferPhase.PLACE,
        )
        if carrying:
            self.lost_contact_duration = (
                0.0 if contact else self.lost_contact_duration + self.physics.timestep
            )
            if self.lost_contact_duration > self.parameters.lost_contact_time:
                self._fail("The hand lost opposing rack-handle contact")
                return
            if not self._samples_retained():
                self._fail("A sample left its upright rack position during transport")
                return
        if (
            self.phase == RackTransferPhase.STOW
            and self.phase_time
            >= self.parameters.stow_time + self.parameters.stow_settling_time
        ):
            self._enter(RackTransferPhase.DOCK)
        elif self.phase in (
            RackTransferPhase.DOCK,
            RackTransferPhase.TRAVEL,
            RackTransferPhase.DOCK_BENCH,
        ):
            measured = self.physics.robot_state()
            base = np.array(
                [measured[RobotBaseJoint.HORIZONTAL], measured[RobotBaseJoint.DEPTH]]
            )
            if np.linalg.norm(base - self.base_target) < self.parameters.hand_tolerance:
                self._enter(
                    RackTransferPhase.CLEARANCE
                    if self.phase == RackTransferPhase.DOCK
                    else (
                        RackTransferPhase.DOCK_BENCH
                        if self.phase == RackTransferPhase.TRAVEL
                        else RackTransferPhase.ALIGN
                    )
                )
        elif (
            self.phase == RackTransferPhase.CLEARANCE
            and np.linalg.norm(self.hand.position - self.target)
            < self.parameters.hand_tolerance
        ):
            self._enter(RackTransferPhase.APPROACH)
        elif (
            self.phase == RackTransferPhase.APPROACH
            and np.linalg.norm(self.hand.position - self.target)
            < self.parameters.hand_tolerance
        ):
            self._enter(RackTransferPhase.GRASP)
        elif (
            self.phase == RackTransferPhase.GRASP
            and self.contact_duration >= self.parameters.contact_time
        ):
            self.grasp_verified = True
            self._enter(RackTransferPhase.LIFT)
        elif (
            self.phase in (RackTransferPhase.LIFT, RackTransferPhase.ALIGN)
            and np.linalg.norm(self.hand.position - self.target)
            < self.parameters.hand_tolerance
        ):
            self._enter(
                RackTransferPhase.TRAVEL
                if self.phase == RackTransferPhase.LIFT
                else RackTransferPhase.PLACE
            )
        elif self.phase == RackTransferPhase.PLACE and self._placed():
            self._enter(RackTransferPhase.RELEASE)
        elif (
            self.phase == RackTransferPhase.RELEASE
            and self.phase_time > self.parameters.release_time
            and self.handle_force < self.parameters.contact_force
        ):
            self._enter(RackTransferPhase.WITHDRAW)
        elif (
            self.phase == RackTransferPhase.WITHDRAW
            and np.linalg.norm(self.hand.position - self.target)
            < self.parameters.hand_tolerance
        ):
            if self._placed():
                self.state = ProgramState.SUCCEEDED
                self.phase = RackTransferPhase.COMPLETE
                self._restore_actuators()
            else:
                self._fail("The released rack or its samples did not remain supported")
        if self.phase_time > self.parameters.phase_timeout:
            self._fail(
                f"The robot did not complete the rack {self.phase.value} movement"
            )

    def _fail(self, reason: str) -> None:
        """Stop the transport and restore ordinary finger actuator settings."""
        self.error = reason
        self.state = ProgramState.FAILED
        self._restore_actuators()

    def snapshot(self) -> dict[str, object]:
        """Report physical carrier position and current grasp evidence."""
        return {
            RackTransferField.STATE: self.state,
            RackTransferField.PHASE: self.phase,
            RackTransferField.POSITION: self.position.tolist(),
            RackTransferField.FORCE: self.handle_force,
            RackTransferField.ERROR: self.error,
        }
