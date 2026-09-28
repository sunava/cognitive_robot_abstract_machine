"""Native two-handed screwdriver actions with an explicitly ideal threaded joint."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from typing_extensions import ClassVar

from coraplex.datastructures.enums import Arms, ExecutionType, TaskStatus
from coraplex.plans.executables import Executable, GiskardExecutable
from coraplex.plans.factories import sequential
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan_node import ExecutionBoundaryNode, PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.base import BaseMotion
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from coraplex.view_manager import ViewManager
from cramera.workshop_gaze import (
    FollowingPointing,
    ManipulationWithGaze,
    WorkshopMoveToolCenterPoint,
    WorkshopAnchoredToolCenterPoint,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.open_close import Close
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body


# %% measured engagement
class WorkshopThreadName(StrEnum):
    """Native task and mechanism names retained in plan recordings."""

    THREAD = "ideal_pilot_thread"
    """Explicit kinematic thread model, without cutting or force simulation."""
    DRIVE = "Drive threaded screw"
    """Control the driver and screw's coupled rotation and translation."""
    HOLD_SUPPORT = "Hold supporting gripper"
    """Keep the supporting hand at its measured world pose."""


@dataclass
class ScrewEngagementFailed(PlanFailure):
    """Measured screwdriver or screw alignment prevents safe thread engagement."""

    reason: str
    """Failed geometric or execution precondition."""

    def error_message(self) -> str:
        """Describe the condition that prevented the thread operation."""
        return self.reason


@dataclass
class ScrewAssembly:
    """Install an ideal helical constraint only after measured blade and pilot alignment."""

    screw: Body
    """Screw whose local origin is its tip and whose positive Z points to its head."""
    driver: Body
    """Held screwdriver whose root is the blade tip and positive Z points to its handle."""
    workpiece: Body
    """Fixed pilot-hole support with its mouth on the local Z axis."""
    pitch: float = 0.006
    """Ideal axial travel per clockwise revolution, in metres."""
    depth: float = 0.11
    """Final screw-tip insertion below the pilot mouth, in metres."""
    mouth_height: float = 0.14
    """Pilot mouth height in the workpiece frame, in metres."""
    slot_floor: float = 0.122
    """Blade seating height above the screw tip, in metres."""
    position_tolerance: float = 0.002
    """Maximum blade and pilot position errors accepted at engagement."""
    angle_tolerance: float = 0.025
    """Maximum orientation error accepted at engagement, in radians."""
    maximum_angular_velocity: float = 1.5
    """Bound on the ideal thread joint velocity, in radians per second."""

    def install(self, world: World) -> ScrewConnection:
        """Replace the screw attachment while retaining its measured world pose."""
        workpiece_T_screw = world.compute_forward_kinematics_np(
            self.workpiece, self.screw
        )
        screw_T_driver = world.compute_forward_kinematics_np(self.screw, self.driver)
        self._validate(workpiece_T_screw, screw_T_driver)
        remaining = float(self.depth - (self.mouth_height - workpiece_T_screw[2, 3]))
        limits = DegreeOfFreedomLimits(
            lower=DerivativeMap(
                position=-remaining * math.tau / self.pitch,
                velocity=-self.maximum_angular_velocity,
            ),
            upper=DerivativeMap(position=0.0, velocity=self.maximum_angular_velocity),
        )
        previous = self.screw.parent_connection
        with world.modify_world():
            connection = ScrewConnection.create_with_dofs(
                world=world,
                parent=self.workpiece,
                child=self.screw,
                name=PrefixedName(WorkshopThreadName.THREAD, str(self.screw.name)),
                parent_T_connection_expression=HomogeneousTransformationMatrix(
                    workpiece_T_screw,
                    reference_frame=self.workpiece,
                    child_frame=self.screw,
                ),
                axis=Vector3.Z(),
                screw_pitch=self.pitch,
                dof_limits=limits,
            )
            world.add_connection(connection)
            world.remove_connection(previous)
            world.delete_orphaned_dofs()
        return connection

    def _validate(
        self, workpiece_T_screw: np.ndarray, screw_T_driver: np.ndarray
    ) -> None:
        """Require a seated, aligned blade and an already entered pilot hole."""
        if not all(
            math.isfinite(value) and value > 0
            for value in (
                self.pitch,
                self.depth,
                self.mouth_height,
                self.slot_floor,
                self.position_tolerance,
                self.angle_tolerance,
                self.maximum_angular_velocity,
            )
        ):
            raise ScrewEngagementFailed(
                "Thread geometry and tolerances must be finite and positive"
            )
        if isinstance(self.screw.parent_connection, ScrewConnection):
            raise ScrewEngagementFailed("The screw already has a threaded connection")
        if (
            np.linalg.norm(screw_T_driver[:3, 3] - [0, 0, self.slot_floor])
            > self.position_tolerance
        ):
            raise ScrewEngagementFailed("The blade tip is not seated in the screw slot")
        orientation_error = math.acos(
            float(np.clip((np.trace(screw_T_driver[:3, :3]) - 1) / 2, -1, 1))
        )
        if orientation_error > self.angle_tolerance:
            raise ScrewEngagementFailed(
                "The blade orientation does not match the screw slot"
            )
        if (
            np.linalg.norm(workpiece_T_screw[:2, 3]) > self.position_tolerance
            or math.acos(float(np.clip(workpiece_T_screw[2, 2], -1, 1)))
            > self.angle_tolerance
        ):
            raise ScrewEngagementFailed("The screw is not aligned with the pilot axis")
        insertion = self.mouth_height - workpiece_T_screw[2, 3]
        if not -self.position_tolerance <= insertion < self.depth:
            raise ScrewEngagementFailed(
                "The screw tip must be at the pilot mouth before engagement"
            )


# %% deferred world modification
@dataclass(eq=False, repr=False)
class EngageThreadNode(ExecutionBoundaryNode):
    """Defer the measured thread installation until preceding motions complete."""

    assembly: ScrewAssembly = field(kw_only=True)
    """Geometry validated immediately before replacing the screw attachment."""

    def notify(self) -> None:
        """Keep symbolic plan expansion free of mechanical state changes."""

    def parse(self) -> EngageThreadExecutable:
        """Create the execution-time geometry check and connection change."""
        return EngageThreadExecutable(node=self, context=self.plan.context)


@dataclass
class EngageThreadExecutable(Executable):
    """Install the ideal thread within native plan lifecycle and execution controls."""

    node: EngageThreadNode = field(kw_only=True)
    """Execution boundary retaining the requested thread geometry."""
    pause_interval: ClassVar[float] = 0.01
    """Wall-time interval for observing paused plan controls."""

    def execute(self) -> None:
        """Install a simulation constraint only after resume and measured validation."""
        with self.node.execution_scope():
            while self.node.is_paused and not self.node.is_interrupted:
                time.sleep(self.pause_interval)
            if self.node.is_interrupted:
                self.node.status = TaskStatus.INTERRUPTED
                return
            if GiskardExecutable.execution_type is ExecutionType.NO_EXECUTION:
                return
            if GiskardExecutable.execution_type is not ExecutionType.SIMULATED:
                raise ScrewEngagementFailed(
                    "Ideal thread actions require simulated robot execution"
                )
            self.node.assembly.install(self.context.world)


# %% attachment-dependent tool goals
@dataclass(eq=False, repr=False)
class HeldBodyGoal(Sequence):
    """Resolve the measured tool attachment after prior grasps have executed."""

    motion: MoveHeldBodyMotion = field(kw_only=True)
    """Deferred body-relative target retaining ordinary tool-motion mapping."""

    def expand(self, context: MotionStatechartContext) -> None:
        """Build the ordinary native Cartesian motion from the current attachment."""
        self.nodes = [self.motion.resolved_motion().motion_chart]
        if self.motion.support_arm is not None:
            support = (
                ViewManager()
                .get_end_effector_view(self.motion.support_arm, self.motion.robot)
                .tool_frame
            )
            self.nodes = [
                ManipulationWithGaze(
                    nodes=[
                        self.nodes[0],
                        CartesianPose(
                            name=WorkshopThreadName.HOLD_SUPPORT,
                            root_link=context.world.root,
                            tip_link=support,
                            goal_pose=Pose(reference_frame=support),
                            translation_threshold=0.002,
                            orientation_threshold=0.01,
                        ),
                    ]
                )
            ]
        super().expand(context)


@dataclass
class MoveHeldBodyMotion(BaseMotion):
    """Move an attached body to a reference-relative pose, or retreat a tool locally."""

    arm: Arms
    """Arm whose current attachment determines the tool target."""
    reference: Body | None = None
    """Target reference body; the active tool is used when omitted."""
    body: Body | None = None
    """Held body whose origin should reach the target; omitted for a bare tool."""
    height: float = 0.0
    """Local target Z translation from the reference origin."""
    retreat: float = 0.0
    """Local negative tool-X distance for withdrawing after release."""
    support_arm: Arms | None = None
    """Optional other hand whose world pose must remain fixed during alignment."""
    requires_individual_execution: ClassVar[bool] = True
    """Resolve goals only after preceding native motions complete."""

    @property
    def _motion_chart(self) -> MotionStatechartNode:
        """Return a deferred native tool goal."""
        return HeldBodyGoal(motion=self)

    def resolved_motion(self) -> MoveToolCenterPointMotion:
        """Compute the tool goal using the held body's measured attachment."""
        tool = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        reference = self.reference if self.reference is not None else tool
        reference_T_target = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-self.retreat, z=self.height
        ).to_np()
        map_T_target = (
            self.world.compute_forward_kinematics_np(self.world.root, reference)
            @ reference_T_target
        )
        if self.body is not None:
            map_T_target = map_T_target @ self.world.compute_forward_kinematics_np(
                self.body, tool
            )
        motion = MoveToolCenterPointMotion(
            target=HomogeneousTransformationMatrix(
                map_T_target, reference_frame=self.world.root
            ).to_pose(),
            arm=self.arm,
            max_linear_velocity=0.08,
            max_angular_velocity=0.35,
            position_threshold=0.0005,
            orientation_threshold=0.008,
        )
        motion.plan_node = self.plan_node
        return motion


@dataclass(eq=False, repr=False)
class TurnScrewGoal(Sequence):
    """Resolve the installed thread after its preceding execution boundary."""

    motion: TurnScrewMotion = field(kw_only=True)
    """Thread motion whose measured mechanism is available only at execution time."""

    def expand(self, context: MotionStatechartContext) -> None:
        """Check the current attachment before compiling the native mechanism goal."""
        self.nodes = [
            self.motion.resolved_chart(target) for target in self.motion.joint_targets()
        ]
        super().expand(context)


@dataclass
class TurnScrewMotion(BaseMotion):
    """Drive a native helical joint while keeping the screwdriver seated in its head."""

    screw: Body
    """Screw already attached to its workpiece by a native screw connection."""
    driver: Body
    """Screwdriver attached to the robot's driving hand."""
    driver_arm: Arms = Arms.RIGHT
    """Driving hand followed by the optical gaze task."""
    support_arm: Arms = Arms.LEFT
    """Other hand whose current world pose must remain fixed."""
    turns: float | None = None
    """Clockwise turns from initial engagement, or complete insertion when omitted."""
    requires_individual_execution: ClassVar[bool] = True
    """Compile after the thread-installation execution boundary."""
    maximum_turns_per_stage: ClassVar[float] = 2.0
    """Bound each native rotation stage so long threads receive enough control periods."""

    @property
    def _motion_chart(self) -> MotionStatechartNode:
        """Keep thread validation deferred until its individual executable starts."""
        return TurnScrewGoal(motion=self)

    def joint_targets(self) -> list[float]:
        """Divide the measured remaining thread travel into bounded clockwise stages."""
        if not isinstance(self.screw.parent_connection, ScrewConnection):
            raise ScrewEngagementFailed(
                "The screw must engage the pilot thread before turning"
            )
        if self.turns is not None and (
            not math.isfinite(self.turns) or self.turns <= 0
        ):
            raise ScrewEngagementFailed("Engagement turns must be finite and positive")
        connection = self.screw.parent_connection
        final = float(connection.dof.limits.lower.position)
        if self.turns is not None:
            return [max(final, -self.turns * math.tau)]
        current = float(connection.position)
        count = max(
            1, int((current - final) / (self.maximum_turns_per_stage * math.tau)) + 1
        )
        return [
            current + (final - current) * index / count for index in range(1, count)
        ] + [final]

    def resolved_chart(self, target: float) -> MotionStatechartNode:
        """Combine native mechanism control, support-hand hold and continuous gaze."""
        support = (
            ViewManager().get_end_effector_view(self.support_arm, self.robot).tool_frame
        )
        driving = (
            ViewManager().get_end_effector_view(self.driver_arm, self.robot).tool_frame
        )
        camera = self.robot.get_default_camera().root
        return ManipulationWithGaze(
            nodes=[
                Close(
                    tip_link=self.driver,
                    environment_link=self.screw,
                    goal_joint_state=target,
                ),
                CartesianPose(
                    name=WorkshopThreadName.HOLD_SUPPORT,
                    root_link=self.world.root,
                    tip_link=support,
                    goal_pose=Pose(reference_frame=support),
                    translation_threshold=0.002,
                    orientation_threshold=0.01,
                ),
                FollowingPointing(
                    root_link=self.robot.get_torso().root,
                    tip_link=camera,
                    goal_point=Point3(reference_frame=driving),
                    pointing_axis=Vector3.Z(reference_frame=camera),
                    max_velocity=WorkshopMoveToolCenterPoint.maximum_head_velocity,
                    threshold=WorkshopMoveToolCenterPoint.pointing_threshold,
                    weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                ),
                WorkshopAnchoredToolCenterPoint.base_hold(self.world, self.robot),
            ],
            name=WorkshopThreadName.DRIVE,
        )


# %% independent native actions
@dataclass
class AlignScrewdriverAction(ActionDescription):
    """Approach along the screw axis and seat the held blade in its slot."""

    screw: Body
    """Screw held on the pilot axis by the supporting hand."""
    driver: Body
    """Already grasped screwdriver."""
    slot_floor: float = 0.122
    """Blade seating height above the screw tip."""
    driver_arm: Arms = Arms.RIGHT
    """Hand carrying the driver."""
    clearance: float = 0.005
    """Axial approach clearance from the slot, in metres."""
    support_arm: Arms = Arms.LEFT
    """Hand holding the screw stationary during blade approach and seating."""

    @property
    def _action_plan(self) -> PlanNode:
        """Build separate native approach and seating motions."""
        return sequential(
            [
                MoveHeldBodyMotion(
                    arm=self.driver_arm,
                    reference=self.screw,
                    body=self.driver,
                    height=self.slot_floor + self.clearance,
                    support_arm=self.support_arm,
                ),
                MoveHeldBodyMotion(
                    arm=self.driver_arm,
                    reference=self.screw,
                    body=self.driver,
                    height=self.slot_floor,
                    support_arm=self.support_arm,
                ),
            ],
            context=self.context,
        )


@dataclass
class EngageScrewAction(ActionDescription):
    """Validate blade engagement and start the thread with the supporting hand held still."""

    screw: Body
    """Screw held at the pilot mouth."""
    driver: Body
    """Driver whose blade has reached the head slot."""
    workpiece: Body
    """Pilot-hole support."""
    pitch: float = 0.006
    """Ideal thread pitch, in metres per revolution."""
    depth: float = 0.11
    """Requested final insertion depth."""
    mouth_height: float = 0.14
    """Local workpiece pilot mouth height."""
    slot_floor: float = 0.122
    """Local screw blade seating height."""
    engagement_turns: float = 0.25
    """Initial turns before opening the supporting hand."""
    driver_arm: Arms = Arms.RIGHT
    """Hand carrying the driver."""
    support_arm: Arms = Arms.LEFT
    """Hand held still around the screw head during engagement."""

    @property
    def _action_plan(self) -> PlanNode:
        """Install the validated ideal joint and execute the initial native rotation."""
        assembly = ScrewAssembly(
            screw=self.screw,
            driver=self.driver,
            workpiece=self.workpiece,
            pitch=self.pitch,
            depth=self.depth,
            mouth_height=self.mouth_height,
            slot_floor=self.slot_floor,
        )
        return sequential(
            [
                EngageThreadNode(assembly=assembly),
                TurnScrewMotion(
                    screw=self.screw,
                    driver=self.driver,
                    driver_arm=self.driver_arm,
                    support_arm=self.support_arm,
                    turns=self.engagement_turns,
                ),
            ],
            context=self.context,
        )


@dataclass
class ReleaseScrewSupportAction(ActionDescription):
    """Open and withdraw the supporting hand while leaving the thread engaged."""

    screw: Body
    """Engaged screw whose thread attachment remains intact."""
    support_arm: Arms = Arms.LEFT
    """Hand that supported the screw head."""
    retreat: float = 0.15
    """Withdrawal distance opposite the tool's approach axis."""
    driver_arm: Arms = Arms.RIGHT
    """Hand kept fixed so the seated blade stays engaged during support withdrawal."""

    @property
    def _action_plan(self) -> PlanNode:
        """Release fingers and retract without detaching the native screw connection."""
        return sequential(
            [
                MoveGripperMotion(motion=GripperState.OPEN, gripper=self.support_arm),
                MoveHeldBodyMotion(
                    arm=self.support_arm,
                    retreat=self.retreat,
                    support_arm=self.driver_arm,
                ),
            ],
            context=self.context,
        )


@dataclass
class DriveScrewAction(ActionDescription):
    """Complete insertion through the native ideal thread and seated driver constraint."""

    screw: Body
    """Screw already engaged in the pilot thread."""
    driver: Body
    """Screwdriver retained in the driving hand."""
    driver_arm: Arms = Arms.RIGHT
    """Hand driving the screw."""
    support_arm: Arms = Arms.LEFT
    """Retracted hand held stationary while driving."""

    @property
    def _action_plan(self) -> PlanNode:
        """Drive the screw to the configured native joint limit."""
        return sequential(
            [
                TurnScrewMotion(
                    screw=self.screw,
                    driver=self.driver,
                    driver_arm=self.driver_arm,
                    support_arm=self.support_arm,
                )
            ],
            context=self.context,
        )
