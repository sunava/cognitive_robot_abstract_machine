"""Contact-driven sample transport between physical laboratory instruments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import StrEnum

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cramera.laboratory_mixing_program import (
    LaboratoryMixingProgram,
    MixingPhase,
    MixingStage,
)
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import JointReference, ProgramState
from cramera.laboratory_physics_robot import RobotBaseJoint
from cramera.laboratory_refrigerator_command import RefrigeratorField


# %% reusable movement intentions
class SampleGrasp(StrEnum):
    """Approach directions compatible with the destination's physical opening."""

    FRONT = "front"
    """Horizontal fingers enter through a cabinet or nozzle's open front."""
    TOP = "top"
    """Inclined fingers enter a deep chamber through its open upper rim."""


@dataclass(frozen=True)
class SampleTransferStage(MixingStage):
    """A sample pose intention with optional straight base travel."""

    base_targets: dict[str, float] | None = None
    """World-aligned planar displacements relative to the initial robot base."""


@dataclass(kw_only=True)
class LaboratorySampleTransfer(LaboratoryMixingProgram):
    """Transport one real sample using opposing fingertips and measured support."""

    object_key: str
    """Freely simulated vessel being transported."""
    destination: tuple[float, float, float] | np.ndarray
    """Desired supported vessel-root position in world coordinates."""
    support_body: str | None = None
    """Physical body whose upward contact must support a released sample."""
    release_at_destination: bool = True
    """Whether the destination ends with supported release or a held vessel."""
    from_held: bool = False
    """Whether a verified physical grasp already exists at construction."""
    grasp: SampleGrasp = SampleGrasp.FRONT
    """Direction from which the gripper approaches and releases the vessel."""
    source_position: np.ndarray = field(init=False)
    """Measured sample root before beginning this transfer."""
    source_rotation: Rotation = field(init=False)
    """Measured sample-axis tilt with axial rotation excluded."""
    base_start: dict[str, float] = field(init=False, default_factory=dict)
    """Measured planar displacements when the current base stage starts."""
    base_speed: float = 0.12
    """Maximum straight base reference speed in meters per second."""
    torso_speed: float = 0.03
    """Maximum torso reference speed in meters per second."""
    transit_height: float = 1.14
    """Vessel-root clearance while traveling between bench instruments."""
    lift_position_tolerance: float = 0.008
    """Finite free-air reference error in meters, within the authored clearance margins."""
    front_depth: float = -0.15
    """Open transport corridor in front of the worktop instruments."""
    cabinet_front_depth: float = -0.55
    """Open corridor allowing vertical lowering ahead of the cabinet door."""
    upper_cabinet_base_offset: float = 0.20
    """Rightward base offset clearing the open door at upper shelf height."""
    centrifuge_base: tuple[float, float] = (-0.30, 0.10)
    """Planar dock placing the raised upper grip within the rotor workspace."""
    centrifuge_clear_depth: float = -0.25
    """Base retreat clearing the centrifuge lid's full closing arc."""
    top_corridor_offset: float = -0.20
    """Lateral root position relative to the mobile base during upper transport."""
    top_extraction_corridor_offset: float = -0.30
    """Lateral return corridor retaining wrist margin after rotor extraction."""
    top_rack_base_offset: float = 0.20
    """Additional rightward docking margin for returning an upper grip to the rack."""
    top_grip_height: float = 0.135
    """Upper pinch height clearing the rotor holder walls, in meters."""
    top_grip_inclination: float = math.radians(30.0)
    """Inclination about the horizontal jaw axis during upper access, in radians."""
    top_shoulder_posture: float = -0.25
    """Raised-elbow shoulder reference for clearing the centrifuge chamber."""

    def __post_init__(self) -> None:
        """Read actual vessel and arm coordinates without modifying physical state."""
        if self.object_key not in self.physics.objects:
            raise InvalidPhysicsTarget("The sample is not present in the laboratory")
        if self.release_at_destination and self.support_body is None:
            raise InvalidPhysicsTarget("A sample destination requires a support body")
        self.source_position = self.physics.data.xpos[
            self.physics.objects[self.object_key].body_id
        ].copy()
        source_axis = self.physics.data.xmat[
            self.physics.objects[self.object_key].body_id
        ].reshape(3, 3)[:, 2]
        self.source_rotation = Rotation.align_vectors(
            source_axis[None, :], np.array([[0.0, 0.0, 1.0]])
        )[0]
        self.destination = np.asarray(self.destination, dtype=float)
        self.arm_names = [
            name
            for name in self.physics.joint_targets
            if name.startswith("pr2/l_") and "gripper" not in name
        ]
        self.arm_indices = [
            int(self.physics.model.jnt_dofadr[self.physics.model.joint(name).id])
            for name in self.arm_names
        ]
        self.grip_rotation = Rotation.from_matrix(
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        )
        if self.grasp == SampleGrasp.TOP:
            self.parameters = replace(
                self.parameters,
                shoulder_posture=self.top_shoulder_posture,
                tool_height=self.top_grip_height,
            )
            diagonal = math.sqrt(0.5)
            self.grip_rotation = Rotation.from_matrix(
                np.array(
                    [
                        [0.0, diagonal, diagonal],
                        [0.0, diagonal, -diagonal],
                        [-1.0, 0.0, 0.0],
                    ]
                )
            )
            self.grip_rotation = (
                Rotation.from_rotvec(
                    np.array([diagonal, diagonal, 0.0]) * self.top_grip_inclination
                )
                * self.grip_rotation
            )
        self.reset()

    def _recipe(self) -> list[SampleTransferStage]:
        """Route the held vessel above fixtures and through the cabinet's front."""
        source = self.source_position
        target = np.asarray(self.destination)
        top_transport_offset = (
            self.top_extraction_corridor_offset
            if self.grasp == SampleGrasp.TOP and self._at_centrifuge(source)
            else self.top_corridor_offset
        )
        stages = []
        rotation = tuple(Rotation.identity().as_quat())
        mobile = RobotBaseJoint.HORIZONTAL in self.physics.robot_joints
        state = self.physics.robot_state()
        base = np.array(
            [
                state.get(RobotBaseJoint.HORIZONTAL, 0.0),
                state.get(RobotBaseJoint.DEPTH, 0.0),
            ]
        )

        def append(
            phase: MixingPhase,
            position: np.ndarray,
            duration: float,
            closed: bool,
            joints: dict[str, float] | None = None,
            base_targets: dict[str, float] | None = None,
            orientation: Rotation | None = None,
        ) -> None:
            """Append one smooth vessel or base intention."""
            stages.append(
                SampleTransferStage(
                    phase,
                    self.object_key,
                    tuple(position),
                    rotation if orientation is None else tuple(orientation.as_quat()),
                    duration,
                    closed,
                    joints,
                    base_targets=base_targets,
                )
            )

        def travel(
            position: np.ndarray, destination: np.ndarray, closed: bool
        ) -> np.ndarray:
            """Translate a fixed arm posture with force-limited planar base servos."""
            nonlocal base
            if abs(destination[0] - base[0]) > self.parameters.position_tolerance:
                if abs(base[1]) > self.parameters.position_tolerance:
                    position = travel(position, np.array([base[0], 0.0]), closed)
                if abs(destination[1]) > self.parameters.position_tolerance:
                    position = travel(position, np.array([destination[0], 0.0]), closed)
            delta = destination - base
            if not mobile or np.linalg.norm(delta) < 0.002:
                return position
            position = position + np.array([*delta, 0.0])
            append(
                MixingPhase.LIFT if closed else MixingPhase.APPROACH,
                position,
                np.max(np.abs(delta)) / self.base_speed + 2.0,
                closed,
                base_targets={
                    RobotBaseJoint.HORIZONTAL: float(destination[0]),
                    RobotBaseJoint.DEPTH: float(destination[1]),
                },
            )
            base = destination.copy()
            return position

        lift = np.array([0.0, 0.0, self.parameters.lift_height])
        front = np.array([0.0, -self.parameters.approach_distance, 0.0])
        if not self.from_held:
            torso = "pr2/torso_lift_joint"
            recovery = self.physics.initial_positions[torso]
            difference = recovery - self.physics.robot_state()[torso]
            if abs(difference) > self.parameters.position_tolerance:
                ready = self._virtual_pose()[0] + np.array([0.0, 0.0, difference])
                append(
                    MixingPhase.APPROACH,
                    ready,
                    abs(difference) / self.torso_speed + 2.0,
                    False,
                    {torso: recovery},
                )
            if self.grasp == SampleGrasp.FRONT and not self.physics.front_grasp_ready:
                append(
                    MixingPhase.APPROACH,
                    source + lift,
                    self.parameters.front_transition_time,
                    False,
                    self.FRONT_TRANSITION,
                )
            source_base = self._base_target(source)
            if (
                mobile
                and np.linalg.norm(source_base - base)
                > self.parameters.position_tolerance
            ):
                ready = np.array(
                    [
                        base[0]
                        + (
                            self.top_corridor_offset
                            if self.grasp == SampleGrasp.TOP
                            else 0.16
                        ),
                        base[1] + self.front_depth,
                        self.transit_height,
                    ]
                )
                append(MixingPhase.APPROACH, ready, 7.0, False)
                travel(ready, source_base, False)
            append(
                MixingPhase.APPROACH,
                source + lift,
                5.0,
                False,
                orientation=self.source_rotation,
            )
            if self.grasp == SampleGrasp.FRONT:
                append(
                    MixingPhase.APPROACH,
                    source + front,
                    5.0,
                    False,
                    orientation=self.source_rotation,
                )
            append(
                MixingPhase.APPROACH,
                source,
                5.0,
                False,
                orientation=self.source_rotation,
            )
            append(
                MixingPhase.GRASP, source, 2.0, True, orientation=self.source_rotation
            )
            raised = source + lift
            if self.grasp == SampleGrasp.TOP and self._at_centrifuge(source):
                raised[2] = max(self.transit_height, raised[2])
        else:
            withdrawal = source + np.array([0.0, -0.10, 0.0])
            append(MixingPhase.LIFT, withdrawal, 5.0, True)
            raised = withdrawal + lift
        append(
            MixingPhase.LIFT,
            raised,
            5.0,
            True,
            orientation=self.source_rotation if not self.from_held else None,
        )
        corridor = np.array(
            [
                (
                    base[0] + top_transport_offset
                    if self.grasp == SampleGrasp.TOP
                    else source[0]
                ),
                self.front_depth + base[1],
                max(self.transit_height, raised[2]),
            ]
        )
        append(MixingPhase.PLACE, corridor, 6.0, True)
        desired_base = self._base_target(target)
        if (
            mobile
            and np.linalg.norm(desired_base - base) > self.parameters.position_tolerance
        ):
            corridor = travel(corridor, desired_base, True)
        if self.grasp == SampleGrasp.TOP and self._at_rack(target):
            corridor = travel(
                corridor,
                np.array([desired_base[0] + self.top_rack_base_offset, base[1]]),
                True,
            )
            append(
                MixingPhase.PLACE,
                np.array([corridor[0], corridor[1], target[2] + lift[2]]),
                6.0,
                True,
            )
        if target[2] < 0.8:
            setback = self.cabinet_front_depth - self.front_depth
            corridor = travel(corridor, np.array([base[0], setback]), True)
            front_target = np.array(
                [target[0], self.cabinet_front_depth, target[2] + 0.045]
            )
            append(MixingPhase.PLACE, front_target, 12.0, True)
            front_target = travel(front_target, np.array([base[0], -0.06]), True)
            append(MixingPhase.PLACE, target + np.array([0.0, 0.0, 0.045]), 8.0, True)
        elif self.release_at_destination:
            above = target + lift
            if self.grasp == SampleGrasp.TOP and self._at_centrifuge(target):
                above[2] = max(self.transit_height, above[2])
            append(MixingPhase.PLACE, above, 6.0, True)
        else:
            station_front = target + np.array([0.0, -0.10, 0.0])
            append(MixingPhase.PLACE, station_front + lift, 5.0, True)
            append(MixingPhase.PLACE, station_front, 5.0, True)
        append(MixingPhase.PLACE, target, 7.0, True)
        if self.release_at_destination:
            append(MixingPhase.RELEASE, target, 1.5, False)
            if target[2] < 0.8:
                withdrawal = np.array([target[0], self.front_depth, target[2]])
                append(MixingPhase.SETTLE, withdrawal, 4.0, False)
                withdrawal = travel(
                    withdrawal,
                    np.array([base[0], self.cabinet_front_depth - self.front_depth]),
                    False,
                )
                append(
                    MixingPhase.APPROACH,
                    np.array([withdrawal[0], withdrawal[1], self.transit_height]),
                    10.0,
                    False,
                )
            elif self.grasp == SampleGrasp.TOP:
                withdrawal = target + lift
                if self._at_centrifuge(target):
                    withdrawal[2] = max(self.transit_height, withdrawal[2])
                append(MixingPhase.SETTLE, withdrawal, 5.0, False)
                parked = np.array(
                    [
                        base[0] + top_transport_offset,
                        base[1] + self.front_depth,
                        withdrawal[2],
                    ]
                )
                append(
                    MixingPhase.APPROACH,
                    parked,
                    5.0,
                    False,
                )
                if self._at_centrifuge(target):
                    travel(
                        parked, np.array([base[0], self.centrifuge_clear_depth]), False
                    )
            else:
                withdrawal = np.array(
                    [
                        target[0],
                        max(-0.06, min(self.front_depth, target[1] - 0.14)),
                        target[2],
                    ]
                )
                append(MixingPhase.SETTLE, withdrawal, 4.0, False)
                append(MixingPhase.APPROACH, withdrawal + lift, 4.0, False)
        return stages

    def _horizontal_base_target(self, position: np.ndarray) -> float:
        """Preserve authored rack and nozzle access while moving to distant devices."""
        station = np.asarray(self.physics.pipetting.target_position)
        if np.linalg.norm(position - station) < 0.03 or self._at_rack(position):
            return 0.0
        pan = np.asarray(self.physics.scene["laboratory"]["scale"]["panPosition"])
        if np.linalg.norm(position - pan) < 0.03:
            return float(position[0] - 0.16)
        slots = self.physics.refrigerator.metadata.get(RefrigeratorField.SLOTS, [])
        if slots:
            lowest = min(slot[RefrigeratorField.POSITION][2] for slot in slots)
            if any(
                np.linalg.norm(position - np.asarray(slot[RefrigeratorField.POSITION]))
                < 0.03
                and slot[RefrigeratorField.POSITION][2]
                > lowest + self.parameters.position_tolerance
                for slot in slots
            ):
                return float(position[0] + self.upper_cabinet_base_offset)
        return float(position[0] + 0.1)

    def _at_rack(self, position: np.ndarray) -> bool:
        """Recognize a supported pose within an authored rack aperture."""
        return any(
            np.linalg.norm(position - np.asarray(slot["pose"][:3])) < 0.03
            for slot in self.physics.scene["laboratory"]["rackSlots"]
        )

    def _base_target(self, position: np.ndarray) -> np.ndarray:
        """Select a measured-workspace dock for the requested support surface."""
        if self._at_centrifuge(position):
            return np.asarray(self.centrifuge_base)
        return np.array([self._horizontal_base_target(position), 0.0])

    def _at_centrifuge(self, position: np.ndarray) -> bool:
        """Recognize insertion poses belonging to the current rotating holders."""
        return any(
            np.linalg.norm(position - self.physics.centrifuge.slot_position(slot))
            < 0.03
            for slot in self.physics.centrifuge.slots
        )

    def _enter_stage(self) -> None:
        """Capture physical origins for bounded base and arm motion."""
        torso = "pr2/torso_lift_joint"
        if self.stage.position[2] < 0.8 and torso not in self.arm_names:
            self.arm_names.append(torso)
            self.arm_indices.append(
                int(self.physics.model.jnt_dofadr[self.physics.model.joint(torso).id])
            )
        super()._enter_stage()
        if self.stage.joint_targets is not None:
            measured = self.physics.robot_state()
            self.start_joint_targets.update(
                {name: measured[name] for name in self.stage.joint_targets}
            )
        state = self.physics.robot_state()
        self.base_start = {name: state.get(name, 0.0) for name in RobotBaseJoint}

    def _complete_stage(self) -> None:
        """Retain the calibrated front-grasp branch across successive transfers."""
        if self.stage.joint_targets is not None:
            measured = self.physics.robot_state()
            error = max(
                abs(measured[name] - target)
                for name, target in self.stage.joint_targets.items()
            )
            if error > self.parameters.position_tolerance:
                if (
                    self.stage_time
                    > self.stage.duration + self.parameters.stage_timeout
                ):
                    self._fail(
                        "The robot did not reach its calibrated approach posture"
                    )
                return
        elif self.stage.base_targets is not None:
            measured = self.physics.robot_state()
            error = max(
                abs(measured[name] - target)
                for name, target in self.stage.base_targets.items()
            )
            if error > self.parameters.position_tolerance:
                if (
                    self.stage_time
                    > self.stage.duration + self.parameters.stage_timeout
                ):
                    self._fail("The robot base did not reach its clear travel waypoint")
                return
        elif self.stage.phase == MixingPhase.LIFT and self.stage.closed:
            position, rotation = self._pose()
            desired_axis = Rotation.from_quat(self.stage.rotation).apply(
                [0.0, 0.0, 1.0]
            )
            measured_axis = rotation.apply([0.0, 0.0, 1.0])
            position_error = np.linalg.norm(position - self._stage_position())
            axis_error = math.acos(
                float(np.clip(np.dot(desired_axis, measured_axis), -1.0, 1.0))
            )
            if (
                position_error > self.lift_position_tolerance
                or axis_error > self.parameters.axis_tolerance
            ):
                if (
                    self.stage_time
                    > self.stage.duration + self.parameters.stage_timeout
                ):
                    self._fail("The held sample did not reach its lift clearance")
                return
        elif (
            self.stage.phase in (MixingPhase.APPROACH, MixingPhase.SETTLE)
            and self.stage.joint_targets is None
        ):
            error = np.linalg.norm(self._virtual_pose()[0] - self._stage_position())
            if error > self.parameters.position_tolerance:
                if (
                    self.stage_time
                    > self.stage.duration + self.parameters.stage_timeout
                ):
                    self._fail(
                        "The open gripper did not reach its approach or clearance waypoint"
                    )
                return
        if (
            self.stage.joint_targets is not None
            and "pr2/l_shoulder_pan_joint" in self.stage.joint_targets
        ):
            self.physics.front_grasp_ready = True
        super()._complete_stage()

    def start(self) -> None:
        """Start a contact-driven transfer without refilling or repositioning vessels."""
        if self.state not in (ProgramState.IDLE, ProgramState.PAUSED):
            return
        actuator = self.physics.model.actuator(JointReference.GRIPPER_JOINT).id
        self.physics.model.actuator_forcerange[actuator] = (
            -self.parameters.finger_torque,
            self.parameters.finger_torque,
        )
        self.physics.release()
        if self.grasp == SampleGrasp.TOP:
            self.physics.front_grasp_ready = False
        self.state = ProgramState.RUNNING
        if self.from_held and not self.grasp_verified:
            super().observe()
            if not self.holding:
                self._fail("The continued transfer has no opposing fingertip contacts")
                return
            self.grasp_verified = True
            self.grasped_objects.add(self.object_key)
            self._enter_stage()

    def advance(self) -> None:
        """Advance either the ordinary Cartesian servo or bounded planar travel."""
        if self.stage.base_targets is None:
            super().advance()
            return
        if self.state != ProgramState.RUNNING:
            return
        self.stage_time += self.physics.timestep
        self.elapsed += self.physics.timestep
        maximum = self.base_speed * self.stage_time
        targets = {
            name: self.base_start[name]
            + float(np.clip(target - self.base_start[name], -maximum, maximum))
            for name, target in self.stage.base_targets.items()
        }
        self.physics.set_joint_targets(targets)

    def _set_joint_targets(self, targets: dict[str, float], duration: float) -> None:
        """Limit torso travel while retaining bounded arm and finger feedback."""
        torso = "pr2/torso_lift_joint"
        if torso in targets:
            previous = self.physics.joint_targets[torso]
            increment = self.torso_speed * duration
            targets[torso] = float(
                np.clip(targets[torso], previous - increment, previous + increment)
            )
        super()._set_joint_targets(targets, duration)

    def _supported_seating(self, object_id: int) -> bool:
        """Require the destination body to carry the quiet upright sample's weight."""
        if not self._placing_for_release() or not self.grasp_verified:
            return False
        return self._supported_destination(
            object_id, self.parameters.position_tolerance
        )

    def _supported_destination(self, object_id: int, position_tolerance: float) -> bool:
        """Measure quiet upright support at the actual destination before or after release."""
        position = self.physics.data.xpos[object_id]
        rotation = Rotation.from_matrix(self.physics.data.xmat[object_id].reshape(3, 3))
        if np.linalg.norm(
            position - self.destination
        ) > position_tolerance or rotation.apply([0.0, 0.0, 1.0])[2] < math.cos(
            self.parameters.upright_tolerance
        ):
            return False
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.physics.model,
            self.physics.data,
            mujoco.mjtObj.mjOBJ_BODY,
            object_id,
            velocity,
            0,
        )
        if (
            np.linalg.norm(velocity[:3]) > self.parameters.seating_angular_speed
            or np.linalg.norm(velocity[3:]) > self.parameters.seating_linear_speed
        ):
            return False
        support_body = self.physics.model.body(self.support_body).id
        support = 0.0
        for index in range(self.physics.data.ncon):
            contact = self.physics.data.contact[index]
            first = int(self.physics.model.geom_bodyid[contact.geom1])
            second = int(self.physics.model.geom_bodyid[contact.geom2])
            if {first, second} != {object_id, support_body} or contact.efc_address < 0:
                continue
            upward = contact.frame[2] * (1 if first == support_body else -1)
            if upward < self.parameters.seating_normal_alignment:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.physics.model, self.physics.data, index, force)
            support += max(0.0, float(force[0])) * upward
        weight = self.physics.model.body_mass[object_id] * np.linalg.norm(
            self.physics.model.opt.gravity
        )
        return support >= weight * self.parameters.seating_weight_fraction

    def _finish(self) -> None:
        """Accept only measured held alignment or a contact-supported release."""
        self.released = self.gripper_force < self.parameters.contact_threshold
        if self.destination_error > self.parameters.destination_tolerance:
            self._fail("The sample did not reach its requested destination")
        elif self.release_at_destination and not self.released:
            self._fail("The sample remained in the gripper after placement")
        elif self.release_at_destination and not self._supported_destination(
            self.physics.objects[self.object_key].body_id,
            self.parameters.destination_tolerance,
        ):
            self._fail(
                "The released sample is not upright and supported at its destination"
            )
        elif not self.release_at_destination and not self.holding:
            self._fail("The sample lost its grasp at the instrument")
        else:
            self.state = ProgramState.SUCCEEDED

    @property
    def destination_error(self) -> float:
        """Measure the manipulated vessel's error relative to its requested root."""
        position = self.physics.data.xpos[self.physics.objects[self.object_key].body_id]
        return float(np.linalg.norm(position - self.destination))
