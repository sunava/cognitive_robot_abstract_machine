"""
Execute a native PR2 screw pickup and insertion into a prepared wooden socket.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from coraplex.datastructures.dataclasses import Context, MotionToleranceConfig
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from cramera import paths
from cramera.generated_json import write_json_atomically
from cramera.laboratory_grasp import ContactGripperMotion
from cramera.laboratory_demo import LaboratoryObservation, LaboratoryResult
from cramera.pr2_gripper import Pr2GripperContact
from cramera.live.bridge import Bridge
from cramera.live.live_bundle import build_live_scene
from cramera.live.recording_bundle import write_recording_bundle
from cramera.live.visualization import LiveVisualization
from cramera.scene_presentation import ScenePresentation
from cramera.workshop_world import WorkshopAsset, WorkshopWorld
from cramera.workshop_gaze import WorkshopMoveToolCenterPoint
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% measured screw grasp


@dataclass
class ScrewGrasp(GraspDescription):
    """
    Approach a shaft point vertically and clear the box before transport.
    """

    shaft_offset: float = 0.070
    """
    Grasp centre along the screw's tip-to-head axis, in metres.
    """

    def pose_sequence(
        self, target_T_grasp_pose: Pose, body: Body | None = None, reverse: bool = False
    ) -> list[Pose]:
        """
        Keep approach and lift in world Z for both lying and upright screws.
        """
        world = self.end_effector._world
        map_T_screw = world.transform(
            target_T_grasp_pose.to_homogeneous_matrix(), world.root
        )
        map_T_tool = map_T_screw @ HomogeneousTransformationMatrix.from_xyz_rpy(
            z=self.shaft_offset
        )
        map_T_clearance = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=self.manipulation_offset)
            @ map_T_tool
        )
        sequence = [
            map_T_clearance.to_pose(),
            map_T_tool.to_pose(),
            map_T_clearance.to_pose(),
        ]
        for pose in sequence:
            pose.reference_frame = world.root
        return list(reversed(sequence)) if reverse else sequence


@dataclass
class PickUpScrewAction(PickUpAction):
    """
    Use collision geometry to stop the PR2 fingers at the screw shaft.
    """

    def create_grasp_motion(self) -> MoveGripperMotion:
        """
        Resolve a geometric finger stop after the final approach is reached.
        """
        return ContactGripperMotion(
            motion=GripperState.CLOSE,
            gripper=self.arm,
            object_designator=self.object_designator,
            contact_clearance=0.00025,
            finger_velocity=self.grasp_closing_velocity,
            joint_position_threshold=0.0001,
        )


@dataclass
class WorkshopObservation(LaboratoryObservation):
    """Verify contact, extraction, placement and all unselected workshop bodies."""

    object_names: tuple[str, ...] = field(kw_only=True)
    """Independent manifest objects whose poses must remain unchanged."""
    initial_base: np.ndarray = field(init=False)
    """Base transform captured before native execution."""
    maximum_base_displacement: float = field(default=0.0, init=False)
    """Largest measured base transform change during the entire plan."""

    def __post_init__(self) -> None:
        """Capture every independently movable body as an acceptance reference."""
        self.initial_pose = self.body.global_pose.to_np().copy()
        self.initial_base = self.context.robot.root.global_pose.to_np().copy()
        self.other_poses = {
            body: body.global_pose.to_np().copy()
            for name in self.object_names
            for body in [self.context.world.get_body_by_name(name)]
            if body is not self.body
        }

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        """Measure the stationary base alongside native contact and lift evidence."""
        super().on_motion_tick(statechart)
        self.maximum_base_displacement = max(
            self.maximum_base_displacement,
            float(
                np.max(
                    np.abs(
                        self.context.robot.root.global_pose.to_np() - self.initial_base
                    )
                )
            ),
        )


@dataclass(frozen=True)
class WorkshopResult(LaboratoryResult):
    """Validated native execution measurements and actual recorded frame count."""

    frames: int = 0
    """Number of captured controller states in the replay."""
    maximum_base_displacement: float = 0.0
    """Largest measured change of a base transform coefficient during execution."""


# %% native context and sequential actions


@dataclass
class WorkshopDemo:
    """
    Compose native pickup and placement in a collision-aware PR2 context.
    """

    workshop: WorkshopWorld = field(default_factory=WorkshopWorld)
    """
    Authored world, independent screw bodies and pilot-hole placement target.
    """

    output_directory: Path = field(
        default_factory=lambda: paths.local_scenes_directory() / "aicor_workshop_pr2"
    )
    """
    Destination for a recorded native replay, separate from the authored assets.
    """

    arm_clearance: float = 0.02
    """Preferred active-arm clearance above the bench, while penetration remains forbidden."""
    position_tolerance: float = 0.0005
    """
    Maximum accepted screw-tip placement error, in metres.
    """

    orientation_tolerance: float = 0.003
    """
    Controller angular tolerance, in radians.
    """

    stationary_base: bool = True
    """Remove the mobile drive for the original fixed-station placement demo."""

    def build_context(self, world: World) -> Context:
        """
        Park the unused arm and keep the base outside the workshop furniture.
        """
        [robot] = world.get_semantic_annotations_by_type(PR2)
        robot.mobile_base.full_body_controlled = self.stationary_base
        if self.stationary_base:
            with world.modify_world():
                world.move_branch_with_fixed_connection(
                    robot.root, robot.root.parent_connection.parent
                )
                world.delete_orphaned_dofs()
        for arm in (robot.left_arm, robot.right_arm):
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        robot.get_torso().get_joint_state_by_type(TorsoState.HIGH).apply_to(world)
        screw = world.get_body_by_name(WorkshopAsset.SCREW)
        gripper = robot.left_arm.end_effector.bodies_with_collision
        obstacles = [
            body
            for body in world.bodies
            if body not in robot.bodies and body is not screw and body.collision
        ]
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=[robot.mobile_base.root],
                    body_group_b=[world.get_body_by_name(WorkshopAsset.FLOOR)],
                )
            )
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(body_group_a=gripper, body_group_b=[screw])
            )
            world.collision_manager.add_default_rule(
                AvoidCollisionBetweenGroups(
                    body_group_a=[
                        body
                        for body in robot.left_arm.bodies_with_collision
                        if body not in gripper
                    ],
                    body_group_b=obstacles,
                    buffer_zone_distance=self.arm_clearance,
                    violated_distance=0.0,
                )
            )
            world.collision_manager.add_default_rule(
                AvoidCollisionBetweenGroups(
                    body_group_a=gripper,
                    body_group_b=obstacles,
                    buffer_zone_distance=0.001,
                    violated_distance=0.0,
                )
            )
            world.collision_manager.add_default_rule(
                AvoidCollisionBetweenGroups(
                    body_group_a=[screw],
                    body_group_b=obstacles,
                    buffer_zone_distance=0.001,
                    violated_distance=0.0,
                )
            )
        return Context(
            world=world,
            robot=robot,
            _debug=False,
            alternative_motion_mappings=[WorkshopMoveToolCenterPoint],
            motion_tolerances=MotionToleranceConfig(
                default_tcp_position_threshold=self.position_tolerance,
                tool_orientation_threshold=self.orientation_tolerance,
            ),
        )

    def actions(self, context: Context) -> list[ActionDescription]:
        """
        Return modular native pickup and placement actions sharing one context.
        """
        body = context.world.get_body_by_name(WorkshopAsset.SCREW)
        target = self.workshop.target_pose
        target.reference_frame = context.world.root
        grasp = ScrewGrasp(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            context.robot.left_arm.end_effector,
            manipulation_offset=0.20,
        )
        return [
            PickUpScrewAction(
                body,
                Arms.LEFT,
                grasp,
                pre_approach_linear_velocity=0.10,
                final_approach_linear_velocity=0.025,
                lift_linear_velocity=0.05,
                grasp_closing_velocity=0.10,
            ),
            PlaceAction(
                body,
                target,
                Arms.LEFT,
                transport_linear_velocity=0.08,
                placing_linear_velocity=0.012,
                retract_linear_velocity=0.08,
            ),
        ]

    def build_plan(self, context: Context) -> Plan:
        """
        Build an ordinary sequential Coraplex plan from the two actions.
        """
        return sequential(self.actions(context), context=context).plan

    def observe(self, context: Context) -> WorkshopObservation:
        """Require geometric fingertip contact and collision-aware manipulation."""
        body = context.world.get_body_by_name(WorkshopAsset.SCREW)
        return WorkshopObservation(
            context=context,
            body=body,
            position_tolerance=self.position_tolerance,
            required_lift=0.20,
            object_names=tuple(item.name for item in self.workshop.objects),
            gripper_contact=Pr2GripperContact(
                context.world,
                context.robot.left_arm.end_effector,
                body,
                clearance=0.00025,
            ),
        )

    def copy_notices(self) -> None:
        """Preserve relative permission and attribution links on original workshop meshes."""
        for filename in ("LICENSE.txt", "NOTICE.md", "rights.json"):
            shutil.copyfile(
                self.workshop.bundle_directory / filename,
                self.output_directory / filename,
            )
        license_identifier = self.workshop.scene["assetRights"]["license"]
        for mesh in self.output_directory.rglob("*.glb"):
            data = mesh.read_bytes()
            document_length = struct.unpack_from("<I", data, 12)[0]
            document = json.loads(data[20 : 20 + document_length])
            rights = document.get("asset", {}).get("extras", {}).get("assetRights", {})
            if rights.get("license") != license_identifier:
                continue
            for field_name in ("licenseFile", "noticeFile"):
                destination = (mesh.parent / rights[field_name]).resolve()
                destination.relative_to(self.output_directory.resolve())
                shutil.copyfile(
                    self.workshop.bundle_directory / destination.name, destination
                )

    def run(self) -> WorkshopResult:
        """
        Execute in the simulated robot environment and save only a valid replay.
        """
        print("Building native workshop world", flush=True)
        world = self.workshop.build()
        context = self.build_context(world)
        plan = self.build_plan(context)
        print("Running native pickup and pilot-hole placement", flush=True)
        visualization = LiveVisualization(
            world,
            bridge=Bridge(
                presentation=ScenePresentation.from_json(self.workshop.scene)
            ),
        ).start()
        plan.node_callbacks.append(visualization.plan_callback(plan))
        observation = self.observe(context)
        plan.node_callbacks.append(observation)
        completed = False
        try:
            build_live_scene(visualization.bridge)
            with simulated_robot_advanced:
                plan.perform()
            frames = visualization.bridge.recording.stop()
            result = WorkshopResult(
                **asdict(observation.validate(plan, self.workshop.target_pose)),
                frames=len(frames),
                maximum_base_displacement=observation.maximum_base_displacement,
            )
            if not result.success or result.maximum_base_displacement > 1e-12:
                raise RuntimeError(
                    f"Native workshop placement failed: {asdict(result)}"
                )
            recording = visualization.bridge.recording
            scene = write_recording_bundle(
                visualization.bridge,
                frames,
                recording.frames_per_second(),
                self.output_directory,
                "aicor_workshop_pr2",
            )
            scene.update(
                {
                    "environmentName": self.workshop.scene["environmentName"],
                    "task": "PR2 · Screw pickup and pilot-hole insertion",
                    "assetRights": self.workshop.scene.get("assetRights", {}),
                    "workshopDemo": {
                        "kind": "native-recording",
                        "steps": ["Pick up screw", "Insert into pilot hole"],
                        "result": asdict(result),
                    },
                }
            )
            write_json_atomically(
                self.output_directory / WorkshopAsset.SCENE, scene, indent=2
            )
            write_json_atomically(
                self.output_directory / "acceptance_metrics.json",
                asdict(result),
                indent=2,
            )
            self.copy_notices()
            recording.scene_name = "aicor_workshop_pr2"
            completed = True
            return result
        finally:
            if not completed:
                contacts = world.collision_manager.compute_collisions().contacts
                diagnostic = {
                    "motions": [
                        {
                            "motion": type(node.designator).__name__,
                            "status": node.status.name,
                        }
                        for node in plan.root.descendants
                        if isinstance(node, MotionNode)
                    ],
                    "screwPose": world.get_body_by_name(WorkshopAsset.SCREW)
                    .global_pose.to_np()
                    .tolist(),
                    "basePose": context.robot.root.global_pose.to_np().tolist(),
                    "minimumCollisionDistance": observation.minimum_collision_distance,
                    "attachedFrames": observation.attached_frames,
                    "contacts": [
                        {
                            "first": str(contact.body_a.name),
                            "second": str(contact.body_b.name),
                            "distance": float(contact.distance),
                        }
                        for contact in contacts
                    ],
                }
                write_json_atomically(
                    self.output_directory.with_suffix(".failure.json"),
                    diagnostic,
                    indent=2,
                )
                visualization.bridge.recording.discard()
            visualization.stop()


def main() -> None:
    """
    Execute a native woodworking plan with optional bundle and replay paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    demo = WorkshopDemo()
    if options.bundle is not None:
        demo.workshop = WorkshopWorld(bundle_directory=options.bundle)
    if options.output is not None:
        demo.output_directory = options.output
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(asdict(demo.run()), indent=2))


if __name__ == "__main__":
    main()
