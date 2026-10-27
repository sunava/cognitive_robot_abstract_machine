"""
The views of a recorded scene's knowledge graph.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from cramera_protocol.bundle.scene import PartRole, RobotPart
from cramera_protocol.questions import GraphEdgeKind, GraphLayout, GraphViewName
from cramera_bridge.questions.exceptions import DuplicateGraphNode, UnknownGraphView
from cramera_bridge.questions.graph.builder import GraphBuilder
from cramera_protocol.bundle.scene import PlanNodeKind
from cramera_bridge.questions.graph.views import (
    PLAN_NODE_GROUPS,
    GraphText,
    GraphViews,
    KinematicsGroup,
    KnowledgeGroup,
    PlanNodeGroup,
)
from cramera_bridge.questions.recorded.bundle import RecordedBundle
from cramera_bridge.questions.recorded.knowledge_base import RecordedKnowledge
from cramera_bridge.questions.recorded.urdf import UrdfStructure
from .dataset.files import PREFIXED_LINKS_URDF, SceneName


@pytest.fixture
def knowledge(catalog) -> RecordedKnowledge:
    """
    :return: The knowledge of the annotated recording.
    """
    return RecordedKnowledge.of_bundle(
        RecordedBundle.of_catalog(catalog, SceneName.ANNOTATED)
    )


# %% views


def test_the_overview_relates_robot_arms_objects_steps_and_events(knowledge):
    view = GraphViews().build(GraphViewName.KNOWLEDGE, knowledge)
    groups = {node.identifier: node.group for node in view.nodes}
    assert groups[knowledge.robot.name] == KnowledgeGroup.ROBOT
    assert groups[knowledge.objects[0].key] == KnowledgeGroup.OBJECT
    labels = {edge.label for edge in view.edges}
    assert {
        GraphText.HAS_ARM,
        GraphText.CARRIES,
        GraphText.PICKS,
        GraphText.ABOUT,
    } <= labels
    assert [entry.group for entry in view.legend] == list(KnowledgeGroup)


def test_the_plan_tree_keeps_native_statuses_and_groups_by_kind(knowledge):
    view = GraphViews().build(GraphViewName.PLAN, knowledge)
    assert view.layout is GraphLayout.HIERARCHY
    assert [node.status for node in view.nodes] == [
        node.status for node in knowledge.plan_nodes
    ]
    assert [node.group for node in view.nodes] == [
        PLAN_NODE_GROUPS.get(node.kind, PlanNodeGroup.OTHER)
        for node in knowledge.plan_nodes
    ]
    assert PLAN_NODE_GROUPS[PlanNodeKind.ATTACHMENT] is PlanNodeGroup.ATTACHMENT
    assert PlanNodeKind.SEQUENCE not in PLAN_NODE_GROUPS


def test_the_statechart_view_carries_the_recorded_statecharts(knowledge):
    view = GraphViews().build(GraphViewName.STATECHART, knowledge)
    assert view.recorded == knowledge.bundle.statecharts


def test_the_kinematic_tree_colours_links_by_part_and_dashes_fixed_joints(knowledge):
    view = GraphViews().build(GraphViewName.KINEMATICS, knowledge)
    groups = {node.identifier: node.group for node in view.nodes}
    assert groups["l_wrist_link"] == KinematicsGroup.LEFT_ARM
    assert groups["l_gripper_link"] == KinematicsGroup.GRIPPER
    assert groups["base_link"] == KinematicsGroup.BASE
    kinds = {edge.destination: edge.kind for edge in view.edges}
    assert kinds["l_gripper_link"] is GraphEdgeKind.TYPE
    assert kinds["l_wrist_link"] is GraphEdgeKind.PROPERTY


def test_the_kinematic_tree_matches_parts_to_links_named_below_the_robot_prefix(
    knowledge,
):
    prefixed = RecordedKnowledge(
        bundle=replace(
            knowledge.bundle,
            robot_structure=UrdfStructure.of_file(PREFIXED_LINKS_URDF),
        ),
        parts=knowledge.parts,
    )
    view = GraphViews().build(GraphViewName.KINEMATICS, prefixed)
    groups = {node.identifier: node.group for node in view.nodes}
    assert groups["pr2/l_wrist_link"] == KinematicsGroup.LEFT_ARM
    assert groups["pr2/l_gripper_link"] == KinematicsGroup.GRIPPER
    assert groups["pr2/base_link"] == KinematicsGroup.BASE


def test_the_kinematic_tree_is_drawn_as_a_hierarchy(knowledge):
    view = GraphViews().build(GraphViewName.KINEMATICS, knowledge)
    assert view.layout is GraphLayout.HIERARCHY


def test_a_scene_without_a_robot_model_has_an_empty_kinematic_tree(knowledge):
    bare = RecordedKnowledge(
        bundle=type(knowledge.bundle)(name="bare", scene=knowledge.bundle.scene)
    )
    view = GraphViews().build(GraphViewName.KINEMATICS, bare)
    assert view.nodes == [] and view.empty == GraphText.NO_ROBOT_MODEL


def test_an_unknown_view_is_refused(knowledge):
    with pytest.raises(UnknownGraphView):
        GraphViews().build("weather", knowledge)


def test_a_node_identity_is_used_once():
    builder = GraphBuilder()
    builder.add_node("a", "a", KnowledgeGroup.OBJECT, [])
    with pytest.raises(DuplicateGraphNode):
        builder.add_node("a", "a", KnowledgeGroup.OBJECT, [])


def test_an_arm_without_a_side_is_drawn_as_an_arm():
    arm = RobotPart(name="arm", role=PartRole.ARM)
    assert KinematicsGroup.of_part(arm) is KinematicsGroup.ARM
