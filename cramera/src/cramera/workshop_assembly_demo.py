"""Record native shelf retrieval and supported two-handed screw driving."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.plans.factories import sequential
from coraplex.plans.attachment_nodes import AttachNode
from coraplex.plans.plan import Plan
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import ActionNode, PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from coraplex.view_manager import ViewManager
from cramera.generated_json import write_json_atomically
from cramera.live.bridge import Bridge
from cramera.live.live_bundle import build_live_scene
from cramera.live.recording_bundle import write_recording_bundle
from cramera.live.visualization import LiveVisualization
from cramera.scene_presentation import ScenePresentation
from cramera.workshop_demo import PickUpScrewAction, ScrewGrasp
from cramera.workshop_transport import MoveHeldObjectAction, WorkshopTransportDemo
from cramera.workshop_navigation import WorkshopTranslateAction
from cramera.workshop_pickup import PickUpLooseScrewAction
from cramera.workshop_world import WorkshopWorld
from cramera.workshop_contents import WorkshopContents
from cramera.workshop_gaze import WorkshopAnchoredToolCenterPoint, ManipulationWithGaze
from cramera.workshop_physics import WorkshopPhysics
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPositionStraight,
    CartesianOrientation,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world import World


# %% assembly acceptance
class AssemblyStage(StrEnum):
    """Visible stages required for the complete native workshop workflow."""

    RETRIEVE = "Retrieve and place screw box"
    PICK_SCREW = "Pick up screw"
    APPROACH = "Move to assembly station"
    PICK_DRIVER = "Pick up screwdriver"
    STABILIZE = "Stabilize screw at pilot hole"
    ALIGN = "Align screwdriver"
    ENGAGE = "Engage screwdriver"
    RELEASE = "Release supporting hand"
    DRIVE = "Drive screw"
    RETURN = "Put screwdriver down"


@dataclass
class WorkshopAssemblyResult:
    """Measured native outcomes required before publishing the complete recording."""

    action_statuses: list[str]
    """Completion status of each top-level plan action."""
    box_position_error: float
    """Distance of the released carrier from its bench pose, in metres."""
    screw_depth: float
    """Measured tip insertion below the pilot mouth, in metres."""
    required_depth: float
    """Authored fully seated insertion depth, in metres."""
    driver_released: bool
    """Whether the tool has been returned to an independent world attachment."""
    support_released: bool
    """Whether the supporting hand withdrew clear of the screw head."""
    minimum_collision_distance: float
    """Smallest signed collision-policy clearance measured during execution."""
    frames: int
    """Number of measured controller states in the recording."""

    @property
    def success(self) -> bool:
        """Reject incomplete action sequences, penetration and unseated assemblies."""
        return bool(
            all(
                math.isfinite(value)
                for value in (
                    self.box_position_error,
                    self.screw_depth,
                    self.required_depth,
                    self.minimum_collision_distance,
                )
            )
            and len(self.action_statuses) == len(AssemblyStage)
            and all(status == "SUCCEEDED" for status in self.action_statuses)
            and self.box_position_error < 0.003
            and abs(self.screw_depth - self.required_depth) < 0.002
            and self.driver_released
            and self.support_released
            and self.minimum_collision_distance >= -0.0005
            and self.frames > 0
        )


@dataclass
class AssemblyObservation(PlanCallback):
    """Track collisions and actual frame intervals for the top-level actions."""

    context: Context = field(kw_only=True)
    """Native execution world being checked."""
    actions: list[ActionDescription] = field(kw_only=True)
    """Ordered top-level designators with user-facing stage names."""
    labels: list[str] = field(kw_only=True)
    """English descriptions matching the action order."""
    bridge: Bridge = field(kw_only=True)
    """World-state recording supplying the final snapshot of each control tick."""
    frame_indices: list[int] = field(default_factory=list)
    """One captured state per control tick, after loose-content integration."""
    ticks: int = 0
    """Number of observed native motion ticks."""
    minimum_distance: float = math.inf
    """Worst measured collision distance outside allowed tool contacts."""
    ranges: list[dict] = field(default_factory=list)
    """Recorded stage intervals for the browser playhead."""

    def on_start(self, node: PlanNode) -> None:
        """Mark stage starts at the actual simulation tick."""
        if isinstance(node, ActionNode):
            for index, action in enumerate(self.actions):
                if node.designator is action:
                    if self.ranges:
                        self.ranges[-1]["endFrame"] = self.ticks - 1
                    self.ranges.append(
                        {
                            "name": self.labels[index],
                            "startFrame": self.ticks,
                            "endFrame": self.ticks,
                        }
                    )
                    print(f"STAGE {index + 1}: {self.labels[index]}", flush=True)

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        """Measure rather than replace the motion controller's body poses."""
        self.ticks += 1
        self.frame_indices.append(self.bridge.recording.frame_count() - 1)
        contacts = self.context.world.collision_manager.compute_collisions().contacts
        if contacts:
            self.minimum_distance = min(
                self.minimum_distance,
                *(float(contact.distance) for contact in contacts),
            )
        if self.ranges:
            self.ranges[-1]["endFrame"] = self.ticks - 1


# %% native grasp and assembly plan
@dataclass
class DriverGrasp(ScrewGrasp):
    """Grip the screwdriver handle axially so wrist roll drives the tool shaft."""

    shaft_offset: float = 0.195
    """Handle centre measured from the driver tip, in metres."""

    approach_clearance: float = 0.025
    """Axial clearance before entering the rear portion of the handle, in metres."""

    def pose_sequence(
        self, target_T_grasp_pose: Pose, body: Body | None = None, reverse: bool = False
    ) -> list[Pose]:
        """Approach the handle along its long axis and lift clear of the bench."""
        world = self.end_effector._world
        target = world.transform(
            target_T_grasp_pose.to_homogeneous_matrix(), world.root
        )
        target = target @ HomogeneousTransformationMatrix.from_xyz_rpy(
            z=self.shaft_offset, pitch=math.pi / 2
        )
        before = target @ HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-self.approach_clearance
        )
        after = HomogeneousTransformationMatrix.from_xyz_rpy(z=0.20) @ target
        poses = [value.to_pose() for value in (before, target, after)]
        for pose in poses:
            pose.reference_frame = world.root
        return poses[::-1] if reverse else poses


@dataclass
class DriverLiftMotion(WorkshopAnchoredToolCenterPoint):
    """Extract the held shaft along a straight line while retaining its orientation."""

    max_linear_velocity: float = field(default=0.04, kw_only=True)
    """Maximum shaft extraction speed in metres per second."""
    max_angular_velocity: float = field(default=0.3, kw_only=True)
    """Reference rate for correcting the retained orientation, in radians per second."""

    @property
    def _motion_chart(self) -> ManipulationWithGaze:
        """Constrain lateral movement until the blade has cleared its support guides."""
        goal = super()._motion_chart
        tool = ViewManager.get_end_effector_view(self.arm, self.robot).tool_frame
        goal.nodes[0] = Parallel(
            [
                CartesianPositionStraight(
                    root_link=self.world.root,
                    tip_link=tool,
                    goal_point=self.target.to_position(),
                    reference_velocity=self.max_linear_velocity,
                    threshold=self.resolved_position_threshold(),
                ),
                CartesianOrientation(
                    root_link=self.world.root,
                    tip_link=tool,
                    goal_orientation=self.target.to_rotation_matrix(),
                    reference_velocity=self.max_angular_velocity,
                    threshold=self.resolved_orientation_threshold(),
                    weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
                ),
            ]
        )
        return goal


@dataclass
class PickUpDriverAction(PickUpScrewAction):
    """Open before the cradle approach and lift the screwdriver with fixed orientation."""

    grasp_position_tolerance: float = 0.001
    """Accepted TCP translation error during the cradle approach, in metres."""

    @property
    def _action_plan(self) -> PlanNode:
        """Keep the blade parallel to its guides until the whole tool clears them."""
        before, target, after = self.grasp_description.grasp_pose_sequence(
            self.object_designator
        )
        return sequential(
            [
                MoveGripperMotion(GripperState.OPEN, self.arm),
                MoveToolCenterPointMotion(
                    before,
                    self.arm,
                    position_threshold=self.grasp_position_tolerance,
                    max_linear_velocity=self.pre_approach_linear_velocity,
                ),
                MoveToolCenterPointMotion(
                    target,
                    self.arm,
                    position_threshold=self.grasp_position_tolerance,
                    max_linear_velocity=self.final_approach_linear_velocity,
                ),
                self.create_grasp_motion(),
                AttachNode(
                    body=self.object_designator,
                    new_parent=ViewManager.get_end_effector_view(
                        self.arm, self.robot
                    ).tool_frame,
                ),
                DriverLiftMotion(
                    after,
                    self.arm,
                    max_linear_velocity=self.lift_linear_velocity,
                    max_angular_velocity=0.3,
                ),
            ]
        )


@dataclass
class WorkshopAssemblyDemo(WorkshopTransportDemo):
    """Ordinary sequential actions running inside the simulated robot context."""

    labels: list[str] = field(default_factory=lambda: list(AssemblyStage))
    """Visible workflow stages associated with the native plan."""

    def actions(self, context: Context) -> list[ActionDescription]:
        """Compose independently reusable retrieval, alignment and screw actions."""
        from cramera.workshop_actions import (
            AlignScrewdriverAction,
            EngageScrewAction,
            ReleaseScrewSupportAction,
            DriveScrewAction,
        )

        world = context.world
        screw, driver, workpiece = [
            world.get_body_by_name(name)
            for name in ("screw_1", "driver", "workshop_assembly")
        ]
        grasp = ScrewGrasp(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            context.robot.left_arm.end_effector,
            manipulation_offset=0.12,
            shaft_offset=0.11,
        )
        driver_grasp = DriverGrasp(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            context.robot.right_arm.end_effector,
        )
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=[screw],
                    body_group_b=[workpiece],
                )
            )
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(body_group_a=[driver], body_group_b=[screw])
            )
        mouth = self.workshop.scene["workshop"]["assembly"]["localMouth"]
        target = (
            workpiece.global_pose.to_homogeneous_matrix()
            @ HomogeneousTransformationMatrix.from_xyz_rpy(*mouth)
        ).to_pose()
        target.reference_frame = world.root
        driver_home = next(
            item.pose for item in self.workshop.objects if item.name == "driver"
        )
        driver_home.reference_frame = world.root
        retrieval = self.retrieval_action(context)
        return [
            retrieval,
            PickUpLooseScrewAction(
                screw,
                Arms.LEFT,
                grasp,
                pre_approach_linear_velocity=0.08,
                final_approach_linear_velocity=0.02,
                lift_linear_velocity=0.04,
            ),
            WorkshopTranslateAction(
                Pose.from_xyz_rpy(0.35, -0.86, reference_frame=world.root)
            ),
            PickUpDriverAction(
                driver,
                Arms.RIGHT,
                driver_grasp,
                pre_approach_linear_velocity=0.08,
                final_approach_linear_velocity=0.02,
                lift_linear_velocity=0.04,
            ),
            MoveHeldObjectAction(screw, target),
            AlignScrewdriverAction(screw, driver),
            EngageScrewAction(screw, driver, workpiece),
            ReleaseScrewSupportAction(screw),
            DriveScrewAction(screw, driver),
            PlaceAction(
                driver,
                driver_home,
                Arms.RIGHT,
                transport_linear_velocity=0.07,
                placing_linear_velocity=0.025,
            ),
        ]

    def run(self) -> WorkshopAssemblyResult:
        """Publish only a complete collision-checked native assembly trajectory."""
        world = self.workshop.build()
        context = self.build_context(world)
        actions = self.actions(context)
        labels = self.labels
        retrieval = actions[0]
        contents = WorkshopContents(
            world=world,
            physics=WorkshopPhysics(self.workshop.bundle_directory),
            control_dt=0.02,
            box=retrieval.box,
            contents=retrieval.screws,
            action=retrieval,
        )
        contents.start()
        plan = sequential(actions, context=context).plan
        visualization = LiveVisualization(
            world,
            port=8731,
            bridge=Bridge(
                presentation=ScenePresentation.from_json(self.workshop.scene)
            ),
        ).start()
        observation = AssemblyObservation(
            context=context, actions=actions, labels=labels, bridge=visualization.bridge
        )
        plan.node_callbacks.extend(
            [contents, observation, visualization.plan_callback(plan)]
        )
        completed = False
        try:
            build_live_scene(visualization.bridge)
            with simulated_robot_advanced:
                plan.perform()
            recorded = visualization.bridge.recording.stop()
            frames = [recorded[index] for index in observation.frame_indices]
            box, screw, workpiece, driver = [
                world.get_body_by_name(name)
                for name in ("screw_box", "screw_1", "workshop_assembly", "driver")
            ]
            assembly_T_screw = world.compute_forward_kinematics(
                workpiece, screw
            ).to_np()
            result = WorkshopAssemblyResult(
                action_statuses=[action.plan_node.status.name for action in actions],
                box_position_error=float(
                    np.linalg.norm(
                        box.global_pose.to_np()[:3, 3]
                        - self.retrieval_action(context).destination.to_np()[:3, 3]
                    )
                ),
                screw_depth=0.14 - float(assembly_T_screw[2, 3]),
                required_depth=0.11,
                driver_released=driver.parent_connection.parent is world.root,
                support_released=float(
                    np.linalg.norm(
                        context.robot.left_arm.end_effector.tool_frame.global_pose.to_np()[
                            :3, 3
                        ]
                        - screw.global_pose.to_np()[:3, 3]
                    )
                )
                > 0.14,
                minimum_collision_distance=observation.minimum_distance,
                frames=len(frames),
            )
            metrics = {
                **asdict(result),
                "success": result.success,
                "contentsPhysics": asdict(contents.measurements),
            }
            if not result.success:
                raise RuntimeError(f"Incomplete workshop assembly: {metrics}")
            scene = write_recording_bundle(
                visualization.bridge,
                frames,
                1.0 / contents.control_dt,
                self.output_directory,
                "aicor_workshop_pr2",
            )
            scene.update(
                {
                    "environmentName": self.workshop.scene["environmentName"],
                    "task": "PR2 · Shelf retrieval and two-handed screw driving",
                    "assetRights": self.workshop.scene["assetRights"],
                    "workshopDemo": {
                        "kind": "native-recording",
                        "steps": labels,
                        "stepRanges": observation.ranges,
                        "result": metrics,
                        "model": "Native Giskard control with ideal grasps and helical thread constraint. Independent MuJoCo contents projected into the carrier frame; no force feedback to the robot or wood cutting.",
                    },
                }
            )
            write_json_atomically(self.output_directory / "scene.json", scene, indent=2)
            write_json_atomically(
                self.output_directory / "acceptance_metrics.json", metrics, indent=2
            )
            self.copy_notices()
            visualization.bridge.recording.scene_name = "aicor_workshop_pr2"
            completed = True
            return result
        finally:
            if not completed:
                self.write_failure(observation, world)
                visualization.bridge.recording.discard()
            visualization.stop()

    def write_failure(self, observation: AssemblyObservation, world: World) -> None:
        """Write finite diagnostics even when no output directory has been created."""
        path = self.output_directory.with_suffix(".failure.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomically(
            path,
            {
                "stage": observation.ranges,
                "minimumCollisionDistance": (
                    observation.minimum_distance
                    if math.isfinite(observation.minimum_distance)
                    else None
                ),
                "poses": {
                    str(body.name): body.global_pose.to_np().tolist()
                    for body in world.bodies
                    if body.name.name
                    in (
                        "screw_box",
                        "screw_1",
                        "driver",
                        "base_footprint",
                        "l_gripper_tool_frame",
                        "r_gripper_tool_frame",
                    )
                },
            },
            indent=2,
        )


def main() -> None:
    """Run the full workshop workflow using the portable authored bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).parent / "resources/aicor_workshop",
    )
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    workshop = WorkshopWorld(
        bundle_directory=options.bundle,
        base_pose=Pose.from_xyz_rpy(1.65, -0.86, yaw=math.pi / 2),
    )
    demo = WorkshopAssemblyDemo(workshop=workshop)
    if options.output is not None:
        demo.output_directory = options.output
    logging.basicConfig(level=logging.INFO)
    print(asdict(demo.run()), flush=True)


if __name__ == "__main__":
    main()
