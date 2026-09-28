"""Native loaded-box retrieval before a two-handed workshop assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from cramera.workshop_navigation import WorkshopTranslateAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from cramera.laboratory_grasp import ContactGripperMotion
from cramera.workshop_demo import PickUpScrewAction, WorkshopDemo
from cramera.workshop_actions import MoveHeldBodyMotion
from cramera.workshop_gaze import WorkshopAnchoredToolCenterPoint
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


# %% measured carrier grasp
@dataclass
class BoxHandleGrasp(GraspDescription):
    """Approach the front carry bar with both fingers above the shelf surface."""

    handle_offset: tuple[float, float, float] = (0.0, -0.204, 0.055)
    """Carry bar centre in the bin root frame, in metres."""

    def pose_sequence(
        self, target_T_grasp_pose: Pose, body: Body | None = None, reverse: bool = False
    ) -> list[Pose]:
        """Withdraw toward the aisle before any lateral movement near the shelf posts."""
        world = self.end_effector._world
        map_T_box = world.transform(
            target_T_grasp_pose.to_homogeneous_matrix(), world.root
        )
        box_T_tool = HomogeneousTransformationMatrix.from_xyz_rpy(
            *self.handle_offset, yaw=math.pi / 2
        )
        target = map_T_box @ box_T_tool
        values = [
            HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.05) @ target,
            target,
            HomogeneousTransformationMatrix.from_xyz_rpy(z=0.09) @ target,
        ]
        result = [value.to_pose() for value in values]
        for pose in result:
            pose.reference_frame = world.root
        return result[::-1] if reverse else result


@dataclass
class PickUpBoxAction(PickUpScrewAction):
    """Resolve closing against the semantic carry handle rather than the wide bin."""

    def create_grasp_motion(self) -> ContactGripperMotion:
        """Use the reached handle geometry for the fingertip stopping position."""
        return ContactGripperMotion(
            GripperState.CLOSE,
            self.arm,
            object_designator=self.world.get_body_by_name("screw_box_handle"),
            contact_clearance=0.00025,
            finger_velocity=0.1,
            joint_position_threshold=0.0001,
        )

    @property
    def _action_plan(self) -> PlanNode:
        """Retain the upright carry pose while lifting the loose contents."""
        _, _, lift = self.grasp_description.grasp_pose_sequence(self.object_designator)
        return sequential(
            [
                self._grasp_attempt_plan(),
                MoveToolCenterPointMotion(
                    lift,
                    self.arm,
                    max_linear_velocity=self.lift_linear_velocity,
                    max_angular_velocity=0.3,
                ),
            ]
        )


@dataclass
class MoveCarriedObjectMotion(MoveHeldBodyMotion):
    """Resolve a world-space object destination from the measured attachment."""

    destination: Pose = field(kw_only=True)
    """Object pose to reach after the preceding grasp has executed."""
    speed: float = field(default=0.06, kw_only=True)
    """Maximum translation velocity in metres per second."""

    def resolved_motion(self) -> MoveToolCenterPointMotion:
        """Defer the object-to-tool transformation until controller expansion."""
        from coraplex.view_manager import ViewManager

        tool = ViewManager.get_end_effector_view(self.arm, self.robot).tool_frame
        object_T_tool = self.world.compute_forward_kinematics(self.body, tool)
        target = (self.destination.to_homogeneous_matrix() @ object_T_tool).to_pose()
        target.reference_frame = self.world.root
        motion = MoveToolCenterPointMotion(
            target, self.arm, max_linear_velocity=self.speed, max_angular_velocity=0.4
        )
        motion.plan_node = self.plan_node
        return motion


@dataclass
class MoveHeldObjectAction(ActionDescription):
    """Move a grasped body using the measured object-to-tool transform."""

    body: Body
    """Body held by the selected hand."""
    target: Pose
    """Desired body pose in the world."""
    arm: Arms = Arms.LEFT
    """Hand retaining the object."""
    speed: float = 0.06
    """Maximum tool translation speed in metres per second."""

    @property
    def _action_plan(self) -> PlanNode:
        """Evaluate the grasp transform only after the preceding pickup."""
        return sequential(
            [
                MoveCarriedObjectMotion(
                    arm=self.arm,
                    body=self.body,
                    destination=self.target,
                    speed=self.speed,
                )
            ]
        )


# %% box retrieval
@dataclass
class RetrieveScrewBoxAction(ActionDescription):
    """Retrieve a loaded container through the free aisle to its bench station.

    The native carrier follows controller poses while the WorkshopContents callback
    updates its independent contents from a separate contact simulation.
    """

    box: Body
    """Container retrieved from the right shelf."""
    screws: tuple[Body, ...]
    """Loose contents carried upright and released on the workbench."""
    destination: Pose
    """Box pose supported by the workbench."""
    bench_base: Pose
    """Collision-checked driving goal in front of the assembly station."""

    @property
    def _action_plan(self) -> PlanNode:
        """Keep extraction, navigation and placement as separate native actions."""
        grasp = BoxHandleGrasp(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            self.robot.left_arm.end_effector,
        )
        aisle = Pose.from_xyz_rpy(
            self.robot.root.global_pose.x,
            -1.62,
            yaw=math.pi / 2,
            reference_frame=self.world.root,
        )
        carrier_x = (
            self.destination.x + self.robot.root.global_pose.x - self.box.global_pose.x
        )
        bench_aisle = Pose.from_xyz_rpy(
            carrier_x, -1.62, yaw=math.pi / 2, reference_frame=self.world.root
        )
        carrier_station = Pose.from_xyz_rpy(
            carrier_x,
            self.bench_base.y - 0.20,
            yaw=math.pi / 2,
            reference_frame=self.world.root,
        )
        return sequential(
            [
                PickUpBoxAction(
                    self.box,
                    Arms.LEFT,
                    grasp,
                    pre_approach_linear_velocity=0.10,
                    final_approach_linear_velocity=0.025,
                    lift_linear_velocity=0.04,
                ),
                WorkshopTranslateAction(aisle),
                WorkshopTranslateAction(bench_aisle),
                WorkshopTranslateAction(carrier_station),
                PlaceAction(
                    self.box,
                    self.destination,
                    Arms.LEFT,
                    transport_linear_velocity=0.07,
                    placing_linear_velocity=0.025,
                ),
                ParkArmsAction(Arms.LEFT),
                NavigateAction(
                    self.bench_base, keep_joint_states=True, face_travel_direction=False
                ),
            ]
        )


@dataclass
class WorkshopTransportDemo(WorkshopDemo):
    """Prepare a mobile native context for shelf retrieval and two-handed assembly."""

    stationary_base: bool = False
    """Preserve the drive for explicit collision-aware navigation actions."""

    def build_context(self, world: World) -> Context:
        """Permit only named manipulation contacts, leaving furniture avoidance active."""
        context = super().build_context(world)
        context.alternative_motion_mappings = [WorkshopAnchoredToolCenterPoint]
        objects = [
            world.get_body_by_name(name)
            for name in ("screw_box", "screw_box_handle", "driver")
            if world.get_bodies_by_name(name)
        ]
        right = context.robot.right_arm
        left = context.robot.left_arm
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=objects[:2],
                    body_group_b=[
                        world.get_body_by_name(f"screw_{index}")
                        for index in range(1, 6)
                    ],
                )
            )
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=left.end_effector.bodies_with_collision,
                    body_group_b=objects[:2],
                )
            )
            driver = world.get_body_by_name("driver")
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=right.end_effector.bodies_with_collision,
                    body_group_b=[driver],
                )
            )
            world.collision_manager.add_default_rule(
                AvoidCollisionBetweenGroups(
                    body_group_a=right.bodies_with_collision,
                    body_group_b=[
                        body
                        for body in world.bodies
                        if body not in context.robot.bodies
                        and body.collision
                        and body is not driver
                    ],
                    buffer_zone_distance=0.001,
                    violated_distance=0.0,
                )
            )
        return context

    def retrieval_action(self, context: Context) -> RetrieveScrewBoxAction:
        """Resolve portable shelf contents and the authored bench destination."""
        world = context.world
        box_entry = next(
            item
            for item in self.workshop.scene["workshop"]["bins"]
            if item["key"] == "screw_box"
        )
        destination = Pose.from_xyz_quaternion(
            *box_entry["benchPose"], reference_frame=world.root
        )
        bench_base = Pose.from_xyz_rpy(
            0.16, -0.86, yaw=math.pi / 2, reference_frame=world.root
        )
        return RetrieveScrewBoxAction(
            world.get_body_by_name("screw_box"),
            tuple(world.get_body_by_name(f"screw_{index}") for index in range(1, 6)),
            destination,
            bench_base,
        )
