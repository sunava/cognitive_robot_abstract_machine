"""
The views of a recorded scene's knowledge graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Dict, List, Optional

from cramera_protocol.bundle.scene import PartRole, PartSide, PlanNodeKind, RobotPart
from cramera_protocol.questions import (
    GraphEdgeKind,
    GraphLayout,
    GraphView,
    GraphViewName,
    LegendEntry,
)
from cramera_bridge.questions.exceptions import UnknownGraphView
from cramera_bridge.questions.graph.builder import GraphBuilder
from cramera_bridge.questions.recorded.knowledge_base import (
    NAME_SEPARATOR,
    RecordedKnowledge,
)

# %% groups


class LabelledGroup(StrEnum):
    """
    A colour group whose legend label is its value written out.
    """

    def label(self) -> str:
        """
        :return: The group as its legend row names it.
        """
        return self.value.replace("_", " ").capitalize()

    @classmethod
    def legend(cls) -> List[LegendEntry]:
        """
        :return: One legend row per group.
        """
        return [LegendEntry(group=group.value, label=group.label()) for group in cls]


class KnowledgeGroup(LabelledGroup):
    """
    The colour groups of the knowledge overview.
    """

    ROBOT = "robot"
    """
    The robot.
    """

    ARM = "arm"
    """
    An arm.
    """

    GRIPPER = "gripper"
    """
    An end effector.
    """

    OBJECT = "object"
    """
    A free object.
    """

    STEP = "step"
    """
    A step of the run.
    """

    EVENT = "event"
    """
    A detected event.
    """


class PlanNodeGroup(LabelledGroup):
    """
    The colour groups of the plan view.
    """

    ACTION = "action"
    """
    An action.
    """

    MOTION = "motion"
    """
    A motion.
    """

    CONDITION = "condition"
    """
    A condition.
    """

    ATTACHMENT = "attachment"
    """
    Attaching or detaching an object.
    """

    OTHER = "other_plan_node"
    """
    Any other plan node.
    """


PLAN_NODE_GROUPS = {
    PlanNodeKind.ACTION: PlanNodeGroup.ACTION,
    PlanNodeKind.MOTION: PlanNodeGroup.MOTION,
    PlanNodeKind.CONDITION: PlanNodeGroup.CONDITION,
    PlanNodeKind.ATTACHMENT: PlanNodeGroup.ATTACHMENT,
}
"""
The colour group of each plan node kind the plan view colours apart; every other kind is
drawn as :attr:`PlanNodeGroup.OTHER`.
"""


class KinematicsGroup(LabelledGroup):
    """
    The colour groups of the kinematic tree.
    """

    BASE = "base"
    """
    A link outside every arm.
    """

    LEFT_ARM = "left_arm"
    """
    A link of the left arm.
    """

    RIGHT_ARM = "right_arm"
    """
    A link of the right arm.
    """

    ARM = "arm"
    """
    A link of an arm without a side.
    """

    GRIPPER = "gripper"
    """
    A link of an end effector.
    """

    @classmethod
    def of_part(cls, part: Optional[RobotPart]) -> KinematicsGroup:
        """
        :param part: The part a link belongs to, or None.
        :return: The link's colour group.
        """
        if part is None:
            return cls.BASE
        if part.role is PartRole.END_EFFECTOR:
            return cls.GRIPPER
        match part.side:
            case PartSide.LEFT:
                return cls.LEFT_ARM
            case PartSide.RIGHT:
                return cls.RIGHT_ARM
        return cls.ARM


class GraphText(StrEnum):
    """
    The texts the views show.
    """

    NO_PLAN = (
        "This scene holds no recorded plan. A live run's plan shows here once the "
        "run is saved and opened from the scene picker."
    )
    """
    The plan view of a scene without a plan.
    """

    NO_STATECHART = (
        "This scene holds no recorded motion statechart. A live run's statechart "
        "shows here once the run is saved and opened from the scene picker."
    )
    """
    The statechart view of a scene without statecharts.
    """

    NO_ROBOT_MODEL = "This scene has no robot model."
    """
    The kinematics view of a scene without a robot.
    """

    HAS_ARM = "has arm"
    """
    A robot's arm.
    """

    CARRIES = "carries"
    """
    An arm's end effector.
    """

    PICKS = "picks"
    """
    The object a step picked up.
    """

    PERFORMED_BY = "performed by"
    """
    The arm that performed a step.
    """

    ABOUT = "about"
    """
    The object an event is about.
    """

    HAS_STEP = "has step"
    """
    A plan node's child.
    """


# %% views


class GraphViewBuilder(ABC):
    """
    Builds one view of a recorded scene's knowledge graph.
    """

    @abstractmethod
    def view_name(self) -> GraphViewName:
        """
        :return: The view this builder builds.
        """

    @abstractmethod
    def build(self, knowledge: RecordedKnowledge) -> GraphView:
        """
        :param knowledge: The entities of the recorded scene.
        :return: The view.
        """


@dataclass(frozen=True)
class KnowledgeOverview(GraphViewBuilder):
    """
    The robot, its arms, the objects, the steps and the detections, and how they relate.
    """

    def view_name(self) -> GraphViewName:
        return GraphViewName.KNOWLEDGE

    def build(self, knowledge: RecordedKnowledge) -> GraphView:
        graph = GraphBuilder()
        for robot in knowledge.robots():
            graph.add_node(
                robot.name,
                robot.name,
                KnowledgeGroup.ROBOT,
                [f"{robot.arm_count} arms"],
            )
        for arm in knowledge.arms:
            graph.add_node(
                arm.name,
                arm.name,
                KnowledgeGroup.ARM,
                [] if arm.side is None else [arm.side.value],
            )
            graph.add_edge(
                arm.robot, arm.name, GraphEdgeKind.PROPERTY, GraphText.HAS_ARM
            )
            if arm.gripper is not None:
                graph.add_node(
                    arm.gripper.name, arm.gripper.name, KnowledgeGroup.GRIPPER, []
                )
                graph.add_edge(
                    arm.name,
                    arm.gripper.name,
                    GraphEdgeKind.PROPERTY,
                    GraphText.CARRIES,
                )
        for entity in knowledge.objects:
            graph.add_node(entity.key, entity.name, KnowledgeGroup.OBJECT, [entity.key])
        for step in knowledge.steps:
            identifier = f"{KnowledgeGroup.STEP}{NAME_SEPARATOR}{step.index}"
            graph.add_node(
                identifier,
                step.name,
                KnowledgeGroup.STEP,
                [] if step.action is None else [step.action],
            )
            if step.picks is not None:
                graph.add_edge(
                    identifier, step.picks.key, GraphEdgeKind.PROPERTY, GraphText.PICKS
                )
            if step.performed_by is not None:
                graph.add_edge(
                    identifier,
                    step.performed_by.name,
                    GraphEdgeKind.TYPE,
                    GraphText.PERFORMED_BY,
                )
        objects = {entity.name: entity.key for entity in knowledge.objects}
        for index, event in enumerate(knowledge.events):
            identifier = f"{KnowledgeGroup.EVENT}{NAME_SEPARATOR}{index}"
            graph.add_node(
                identifier,
                event.name,
                KnowledgeGroup.EVENT,
                [event.event_type, event.timestamp.isoformat()],
            )
            if event.tracked_object in objects:
                graph.add_edge(
                    identifier,
                    objects[event.tracked_object],
                    GraphEdgeKind.PROPERTY,
                    GraphText.ABOUT,
                )
        return GraphView(
            view=self.view_name(),
            nodes=graph.nodes,
            edges=graph.edges,
            details=graph.details,
            legend=KnowledgeGroup.legend(),
        )


@dataclass(frozen=True)
class PlanTree(GraphViewBuilder):
    """
    The executed plan as a tree, each node with the native status it ended in.
    """

    def view_name(self) -> GraphViewName:
        return GraphViewName.PLAN

    def build(self, knowledge: RecordedKnowledge) -> GraphView:
        graph = GraphBuilder()
        for node in knowledge.plan_nodes:
            lines = [node.kind] + [
                line for line in (node.target, node.arm) if line is not None
            ]
            graph.add_node(
                node.name,
                node.label,
                PLAN_NODE_GROUPS.get(node.kind, PlanNodeGroup.OTHER),
                lines,
                status=node.status,
                kind=node.kind,
                parent=node.parent,
            )
            if node.parent is not None:
                graph.add_edge(
                    node.parent, node.name, GraphEdgeKind.PROPERTY, GraphText.HAS_STEP
                )
        return GraphView(
            view=self.view_name(),
            nodes=graph.nodes,
            edges=graph.edges,
            details=graph.details,
            legend=PlanNodeGroup.legend(),
            layout=GraphLayout.HIERARCHY,
            empty=GraphText.NO_PLAN,
        )


@dataclass(frozen=True)
class RecordedStatechartView(GraphViewBuilder):
    """
    The motion statecharts the run went through, which the browser replays frame by
    frame.
    """

    def view_name(self) -> GraphViewName:
        return GraphViewName.STATECHART

    def build(self, knowledge: RecordedKnowledge) -> GraphView:
        return GraphView(
            view=self.view_name(),
            layout=GraphLayout.HIERARCHY,
            empty=GraphText.NO_STATECHART,
            recorded=knowledge.bundle.statecharts,
        )


@dataclass(frozen=True)
class KinematicTree(GraphViewBuilder):
    """
    The robot model's links as nodes and its joints as edges; movable joints are drawn
    solid, fixed ones dashed, and links are coloured by the part they belong to.
    """

    def view_name(self) -> GraphViewName:
        return GraphViewName.KINEMATICS

    def build(self, knowledge: RecordedKnowledge) -> GraphView:
        structure = knowledge.bundle.robot_structure
        if structure is None:
            return GraphView(view=self.view_name(), empty=GraphText.NO_ROBOT_MODEL)
        parts: Dict[str, RobotPart] = {}
        for part in knowledge.parts:
            for link in part.links:
                parts.setdefault(self.bare_name(link), part)
        graph = GraphBuilder()
        carrying = structure.joints_by_child()
        for link in structure.links:
            joint = carrying.get(link)
            lines = [] if joint is None else [f"{joint.name} ({joint.joint_type})"]
            group = KinematicsGroup.of_part(parts.get(self.bare_name(link)))
            graph.add_node(link, link, group, lines)
        for joint in structure.joints:
            graph.add_edge(
                joint.parent,
                joint.child,
                GraphEdgeKind.PROPERTY if joint.is_movable() else GraphEdgeKind.TYPE,
                f"{joint.name} ({joint.joint_type})",
            )
        return GraphView(
            view=self.view_name(),
            nodes=graph.nodes,
            edges=graph.edges,
            details=graph.details,
            legend=KinematicsGroup.legend(),
            layout=GraphLayout.HIERARCHY,
        )

    @staticmethod
    def bare_name(link: str) -> str:
        """
        :param link: A link name, with or without the robot's prefix.
        :return: The name without the prefix; recordings name a part's links and the
            robot model's links with or without it independently.
        """
        return link.rsplit(NAME_SEPARATOR, 1)[-1]


@dataclass(frozen=True)
class GraphViews:
    """
    The view builders by the name of the view they build.
    """

    builders: List[GraphViewBuilder] = field(
        default_factory=lambda: [
            KnowledgeOverview(),
            PlanTree(),
            RecordedStatechartView(),
            KinematicTree(),
        ]
    )
    """
    The builders.
    """

    def build(self, name: str, knowledge: RecordedKnowledge) -> GraphView:
        """
        :param name: A view name, a :class:`GraphViewName` value.
        :param knowledge: The entities of the recorded scene.
        :return: The view.
        :raises UnknownGraphView: When no builder builds a view of that name.
        """
        for builder in self.builders:
            if builder.view_name().value == name:
                return builder.build(knowledge)
        raise UnknownGraphView(name)
