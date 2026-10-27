"""
What a recorded scene knows, as entities a query ranges over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from typing_extensions import Dict, List, Optional

from cramera_protocol.bundle.scene import (
    PartRole,
    PartSide,
    PlanNode,
    RobotDescription,
    RobotPart,
    SceneDescription,
)
from cramera_protocol.bundle.trajectory import Trajectory
from cramera_protocol.questions import QueryScope
from cramera_bridge.questions.desk import QuestionDesk
from cramera_bridge.questions.highlights import EntityNameKeys
from cramera_bridge.questions.knowledge import QueryableKnowledge, QueryDomain
from cramera_bridge.questions.presets import (
    DetectedEventQuestions,
    Preset,
    QueryVariable,
    RobotQuestions,
)
from cramera_bridge.questions.recorded.bundle import RecordedBundle
from cramera_bridge.questions.recorded.entities import (
    DetectedEventRecord,
    JointMotion,
    Position,
    RecordedArm,
    RecordedGripper,
    RecordedObject,
    RecordedPlanNode,
    RecordedRobot,
    RecordedStep,
)
from cramera_bridge.questions.recorded.urdf import UrdfStructure
from cramera_bridge.questions.workspace_classes import WorkspaceClassIndex

RECORDED_PRECISION = 3
"""
Decimal places of a recorded position, duration or joint value.
"""

PLAN_NODE_PREFIX = "plan_node_"
"""
The start of the name given to a recorded plan node without an identity.
"""

NAME_SEPARATOR = "/"
"""
Separates a model prefix from a link or joint name.
"""

# %% legacy robot parts


class LegacyRobotPart(StrEnum):
    """
    The part names the earliest recordings listed under ``robot.parts`` without
    annotations; each is read as the role and side it names.
    """

    LEFT_ARM = "left_arm"
    """
    The left arm.
    """

    RIGHT_ARM = "right_arm"
    """
    The right arm.
    """

    ARM = "arm"
    """
    The only arm.
    """

    LEFT_GRIPPER = "left_gripper"
    """
    The left arm's gripper.
    """

    RIGHT_GRIPPER = "right_gripper"
    """
    The right arm's gripper.
    """

    GRIPPER = "gripper"
    """
    The only arm's gripper.
    """

    def side(self) -> Optional[PartSide]:
        """
        :return: The side the part is on, or None for a robot with one arm.
        """
        match self:
            case LegacyRobotPart.LEFT_ARM | LegacyRobotPart.LEFT_GRIPPER:
                return PartSide.LEFT
            case LegacyRobotPart.RIGHT_ARM | LegacyRobotPart.RIGHT_GRIPPER:
                return PartSide.RIGHT
        return None

    def carrying_arm(self) -> Optional[LegacyRobotPart]:
        """
        :return: The arm carrying a gripper, or None for an arm.
        """
        match self:
            case LegacyRobotPart.LEFT_GRIPPER:
                return LegacyRobotPart.LEFT_ARM
            case LegacyRobotPart.RIGHT_GRIPPER:
                return LegacyRobotPart.RIGHT_ARM
            case LegacyRobotPart.GRIPPER:
                return LegacyRobotPart.ARM
        return None

    def part(self, links: List[str]) -> RobotPart:
        """
        :param links: The part's links as the recording lists them.
        :return: The part as an annotated recording describes it.
        """
        arm = self.carrying_arm()
        return RobotPart(
            name=self.value,
            role=PartRole.ARM if arm is None else PartRole.END_EFFECTOR,
            side=self.side(),
            links=links,
            attached_to=None if arm is None else arm.value,
        )


@dataclass(frozen=True)
class RecordedParts:
    """
    The robot parts a recording describes.
    """

    robot: RobotDescription
    """
    The recorded robot.
    """

    def parts(self) -> List[RobotPart]:
        """
        :return: The annotated parts, or the parts read from the earliest recordings'
            part names; names that are no legacy part name are left out.
        """
        if self.robot.part_annotations:
            return list(self.robot.part_annotations)
        legacy = {member.value for member in LegacyRobotPart}
        return [
            LegacyRobotPart(name).part(links)
            for name, links in self.robot.parts.items()
            if name in legacy
        ]


# %% knowledge


@dataclass
class RecordedKnowledge:
    """
    The entities of one recorded scene.
    """

    bundle: RecordedBundle
    """
    The bundle the entities were read from.
    """

    robot: Optional[RecordedRobot] = None
    """
    The recorded robot.
    """

    parts: List[RobotPart] = field(default_factory=list)
    """
    The recorded robot's parts.
    """

    grippers: List[RecordedGripper] = field(default_factory=list)
    """
    The robot's end effectors.
    """

    arms: List[RecordedArm] = field(default_factory=list)
    """
    The robot's arms.
    """

    objects: List[RecordedObject] = field(default_factory=list)
    """
    The free objects.
    """

    steps: List[RecordedStep] = field(default_factory=list)
    """
    The steps of the run.
    """

    joints: List[JointMotion] = field(default_factory=list)
    """
    How far each recorded joint moved.
    """

    plan_nodes: List[RecordedPlanNode] = field(default_factory=list)
    """
    The nodes of the executed plans.
    """

    events: List[DetectedEventRecord] = field(default_factory=list)
    """
    What detectors saw.
    """

    @classmethod
    def of_bundle(cls, bundle: RecordedBundle) -> RecordedKnowledge:
        """
        :param bundle: A recorded bundle.
        :return: Its entities.
        """
        knowledge = cls(bundle=bundle)
        scene = bundle.scene
        knowledge.objects = knowledge.read_objects(scene)
        if scene.robot is not None:
            knowledge.read_robot(scene.robot)
        knowledge.steps = knowledge.read_steps(scene)
        knowledge.joints = knowledge.read_joints(bundle.trajectory)
        knowledge.plan_nodes = knowledge.read_plan_nodes(scene.plan_trees)
        knowledge.events = [
            DetectedEventRecord(
                name=event.name,
                event_type=event.event_type,
                timestamp=datetime.fromisoformat(event.timestamp),
                tracked_object=event.tracked_object,
                with_object=event.with_object,
            )
            for event in scene.detected_events
        ]
        return knowledge

    # %% reading

    @staticmethod
    def read_objects(scene: SceneDescription) -> List[RecordedObject]:
        """
        :param scene: The scene description.
        :return: The free objects at their spawn positions.
        """
        return [
            RecordedObject(
                name=entry.identifier,
                key=entry.key,
                height_metres=entry.height,
                position=Position(
                    round(entry.spawn.x, RECORDED_PRECISION),
                    round(entry.spawn.y, RECORDED_PRECISION),
                    round(entry.spawn.z, RECORDED_PRECISION),
                ),
            )
            for entry in scene.objects
        ]

    def read_robot(self, robot: RobotDescription) -> None:
        """
        Read the robot, its arms and their grippers from its parts.

        :param robot: The recorded robot.
        """
        self.parts = RecordedParts(robot).parts()
        grippers = {
            part.attached_to: RecordedGripper(name=part.name, side=part.side)
            for part in self.parts
            if part.role is PartRole.END_EFFECTOR
        }
        robot_name = robot.label or robot.name
        self.arms = [
            RecordedArm(
                name=part.name,
                side=part.side,
                robot=robot_name,
                gripper=grippers.get(part.name),
            )
            for part in self.parts
            if part.role is PartRole.ARM
        ]
        self.grippers = list(grippers.values())
        self.robot = RecordedRobot(name=robot_name, arm_count=len(self.arms))

    def read_steps(self, scene: SceneDescription) -> List[RecordedStep]:
        """
        :param scene: The scene description.
        :return: The steps of the run.
        """
        objects = {entity.name: entity for entity in self.objects}
        objects.update({entity.key: entity for entity in self.objects})
        return [
            RecordedStep(
                name=segment.step,
                index=index,
                start_frame=segment.start,
                end_frame=segment.end,
                duration_seconds=(
                    None
                    if not scene.frames_per_second
                    else round(
                        (segment.end - segment.start) / scene.frames_per_second,
                        RECORDED_PRECISION,
                    )
                ),
                action=segment.action,
                performed_by=self.arm_labelled(segment.arm),
                picks=objects.get(segment.picks) if segment.picks else None,
            )
            for index, segment in enumerate(scene.segments)
        ]

    def arm_labelled(self, label: Optional[str]) -> Optional[RecordedArm]:
        """
        :param label: The arm label a recorder wrote: an arm's name or its side.
        :return: The arm it names, or None.
        """
        if label is None:
            return None
        return next(
            (
                arm
                for arm in self.arms
                if label == arm.name
                or (arm.side is not None and label.lower() == arm.side.value)
            ),
            None,
        )

    def read_joints(self, trajectory: Optional[Trajectory]) -> List[JointMotion]:
        """
        :param trajectory: The recorded trajectory.
        :return: The range each joint moved over, by its prefixed name.
        """
        if trajectory is None:
            return []
        values: Dict[str, List[float]] = {}
        for frame in trajectory.frames:
            for joint, value in frame.items():
                values.setdefault(joint, []).append(value)
        return [
            JointMotion(
                name=joint,
                part=self.part_moved_by(joint),
                minimum=round(min(recorded), RECORDED_PRECISION),
                maximum=round(max(recorded), RECORDED_PRECISION),
                travelled=round(max(recorded) - min(recorded), RECORDED_PRECISION),
            )
            for joint, recorded in sorted(values.items())
        ]

    def part_moved_by(self, joint: str) -> Optional[str]:
        """
        :param joint: A joint's prefixed name as the trajectory writes it.
        :return: The part whose links contain the link the joint carries, or None.
        """
        structure = self.bundle.robot_structure or UrdfStructure()
        carried = {urdf_joint.name: urdf_joint.child for urdf_joint in structure.joints}
        local = joint.rsplit(NAME_SEPARATOR, 1)[-1]
        link = carried.get(joint) or carried.get(local)
        if link is None:
            return None
        for part in self.parts:
            part_links = {name.rsplit(NAME_SEPARATOR, 1)[-1] for name in part.links}
            if link.rsplit(NAME_SEPARATOR, 1)[-1] in part_links:
                return part.name
        return None

    @classmethod
    def read_plan_nodes(cls, trees: List[PlanNode]) -> List[RecordedPlanNode]:
        """
        :param trees: The recorded plan roots.
        :return: Every node, parents before their children; a node recorded without an
            identity is named after its position.
        """
        collected: List[RecordedPlanNode] = []
        cls.collect_plan_nodes(trees, None, collected)
        return collected

    @classmethod
    def collect_plan_nodes(
        cls,
        trees: List[PlanNode],
        parent: Optional[str],
        collected: List[RecordedPlanNode],
    ) -> None:
        """
        :param trees: Recorded plan nodes.
        :param parent: The name of their parent, or None for roots.
        :param collected: The nodes read so far, extended by these and their
            descendants.
        """
        for tree in trees:
            name = tree.identifier or f"{PLAN_NODE_PREFIX}{len(collected)}"
            collected.append(
                RecordedPlanNode(
                    name=name,
                    kind=tree.kind,
                    label=tree.label,
                    status=tree.status,
                    parent=parent,
                    target=tree.target,
                    arm=tree.arm,
                )
            )
            cls.collect_plan_nodes(tree.children, name, collected)

    # %% questions

    def knowledge(self) -> List[QueryableKnowledge]:
        """
        :return: The scene's current state, and its detections when it has any.
        """
        current = QueryableKnowledge(
            scope=QueryScope.CURRENT_STATE,
            domains=[
                QueryDomain(QueryVariable.ROBOT, RecordedRobot, self.robots()),
                QueryDomain(QueryVariable.ARM, RecordedArm, self.arms),
                QueryDomain(QueryVariable.GRIPPER, RecordedGripper, self.grippers),
                QueryDomain(QueryVariable.OBJECT, RecordedObject, self.objects),
                QueryDomain(QueryVariable.STEP, RecordedStep, self.steps),
                QueryDomain(QueryVariable.JOINT, JointMotion, self.joints),
                QueryDomain(QueryVariable.PLAN_NODE, RecordedPlanNode, self.plan_nodes),
            ],
        )
        if not self.events:
            return [current]
        detections = QueryableKnowledge(
            scope=QueryScope.DETECTED_EVENTS,
            domains=[
                QueryDomain(QueryVariable.EVENT, DetectedEventRecord, self.events)
            ],
        )
        return [current, detections]

    def robots(self) -> List[RecordedRobot]:
        """
        :return: The recorded robot, or nothing for a scene without one.
        """
        return [] if self.robot is None else [self.robot]

    def event_questions(self) -> DetectedEventQuestions:
        """
        :return: The questions about the detected event types.
        """
        return DetectedEventQuestions.of_types(
            event.event_type for event in self.events
        )

    def presets(self) -> List[Preset]:
        """
        :return: The robot questions and the questions about detections.
        """
        robot_questions = RobotQuestions(len(self.robots())).presets()
        return (robot_questions if self.robot else []) + self.event_questions().listed()

    def desk(self, class_index: WorkspaceClassIndex) -> QuestionDesk:
        """
        :param class_index: The workspace classes a query may name.
        :return: The desk answering questions about the scene; answers light objects up
            under their keys.
        """
        return QuestionDesk(
            title=self.bundle.name,
            knowledge=self.knowledge(),
            presets=self.presets(),
            unlisted_presets=self.event_questions().unlisted(),
            highlight_keys=EntityNameKeys(
                keys_by_name={entity.name: entity.key for entity in self.objects},
                known_keys=frozenset(entity.key for entity in self.objects),
            ),
            class_index=class_index,
        )
