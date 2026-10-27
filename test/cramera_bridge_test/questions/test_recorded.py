"""
What a recorded scene knows.
"""

from __future__ import annotations

import json

import pytest

from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.bundle.scene import PartRole, PartSide, SceneDescription
from cramera_protocol.questions import QueryRequest, QueryScope
from cramera_bridge.questions.presets import (
    DetectedEventQuestion,
    QueryVariable,
    RobotQuestion,
)
from cramera_bridge.questions.recorded.bundle import BundleReader, RecordedBundle
from cramera_bridge.questions.recorded.knowledge_base import (
    PLAN_NODE_PREFIX,
    LegacyRobotPart,
    RecordedKnowledge,
)
from cramera_bridge.questions.recorded.urdf import UrdfJointType
from cramera_bridge.questions.workspace_classes import WorkspaceClassIndex
from .dataset.files import SceneName


@pytest.fixture
def annotated(catalog) -> RecordedKnowledge:
    """
    :return: The knowledge of the annotated recording.
    """
    return RecordedKnowledge.of_bundle(
        RecordedBundle.of_catalog(catalog, SceneName.ANNOTATED)
    )


@pytest.fixture
def legacy(catalog) -> RecordedKnowledge:
    """
    :return: The knowledge of one of the earliest recordings.
    """
    return RecordedKnowledge.of_bundle(
        RecordedBundle.of_catalog(catalog, SceneName.LEGACY)
    )


def scene_data(scenes, name: SceneName) -> dict:
    """
    :param scenes: The scenes directory.
    :param name: A bundle of it.
    :return: The bundle's decoded scene description.
    """
    return json.loads((scenes / name / BundleFile.SCENE).read_text(encoding="utf-8"))


# %% bundle


def test_the_bundle_reads_every_declared_file(catalog):
    bundle = RecordedBundle.of_catalog(catalog, SceneName.ANNOTATED)
    assert bundle.trajectory.frame_count() == len(bundle.statecharts.frames)
    assert bundle.robot_structure.joints[0].joint_type is UrdfJointType.PRISMATIC
    assert bundle.robot_structure.joint_carrying("base_link") is None


def test_a_rewritten_scene_changes_the_stamp(catalog, scenes):
    reader = BundleReader(catalog, SceneName.ANNOTATED)
    before = reader.stamp()
    path = scenes / SceneName.ANNOTATED / BundleFile.SCENE
    path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    assert reader.stamp() >= before


# %% entities


def test_arms_and_grippers_come_from_the_annotations(annotated, scenes):
    data = scene_data(scenes, SceneName.ANNOTATED)
    annotations = SceneDescription.from_json(data).robot.part_annotations
    [arm] = annotated.arms
    assert arm.name == annotations[0].name
    assert arm.gripper.name == annotations[1].name
    assert arm.side is PartSide.LEFT
    assert annotated.robot.arm_count == 1
    assert annotated.robot.name == data["robot"]["label"]


def test_the_earliest_part_names_read_as_roles_and_sides(legacy):
    roles = {part.name: (part.role, part.side) for part in legacy.parts}
    assert roles[LegacyRobotPart.LEFT_ARM] == (PartRole.ARM, PartSide.LEFT)
    assert roles[LegacyRobotPart.LEFT_GRIPPER] == (PartRole.END_EFFECTOR, PartSide.LEFT)
    assert legacy.arms[0].gripper.name == LegacyRobotPart.LEFT_GRIPPER


def test_legacy_parts_without_a_side_belong_to_the_only_arm():
    gripper = LegacyRobotPart.GRIPPER.part([])
    assert gripper.attached_to == LegacyRobotPart.ARM
    assert LegacyRobotPart.ARM.side() is None


def test_objects_keep_their_keys_and_spawn_positions(annotated, scenes):
    data = scene_data(scenes, SceneName.ANNOTATED)["objects"][0]
    milk = annotated.objects[0]
    assert (milk.name, milk.key, milk.height_metres) == (
        data["id"],
        data["key"],
        data["height"],
    )
    assert milk.position.x == round(data["spawn"][0], 3)


def test_steps_name_their_arm_and_object(annotated):
    prepare, transport = annotated.steps
    assert prepare.performed_by is None and prepare.picks is None
    assert transport.performed_by is annotated.arms[0]
    assert transport.picks is annotated.objects[0]
    assert transport.duration_seconds == 1.0


def test_joints_report_their_range_and_the_part_they_move(annotated):
    joints = {joint.name: joint for joint in annotated.joints}
    wrist = joints["pr2/l_wrist_flex_joint"]
    assert (wrist.minimum, wrist.maximum, wrist.travelled) == (-0.4, -0.1, 0.3)
    assert wrist.part == annotated.arms[0].name
    assert joints["pr2/torso_lift_joint"].part is None


def test_plan_nodes_keep_their_native_status_and_get_names(annotated, scenes):
    tree = scene_data(scenes, SceneName.ANNOTATED)["planTrees"][0]
    names = [node.name for node in annotated.plan_nodes]
    assert names[:2] == [tree["id"], tree["children"][0]["id"]]
    assert names[2] == f"{PLAN_NODE_PREFIX}2"
    assert annotated.plan_nodes[1].status == tree["children"][0]["status"]
    assert annotated.plan_nodes[2].parent == tree["children"][0]["id"]


def test_detections_are_their_own_scope(annotated, legacy):
    assert [knowledge.scope for knowledge in annotated.knowledge()] == [
        QueryScope.CURRENT_STATE,
        QueryScope.DETECTED_EVENTS,
    ]
    assert [knowledge.scope for knowledge in legacy.knowledge()] == [
        QueryScope.CURRENT_STATE
    ]


# %% questions


def test_the_desk_offers_robot_and_detection_questions(annotated):
    desk = annotated.desk(WorkspaceClassIndex())
    texts = [preset.text for preset in desk.presets]
    assert texts == [
        RobotQuestion.WHICH_ROBOT,
        RobotQuestion.WHICH_ARMS,
        DetectedEventQuestion.WHAT_AND_WHEN,
    ]
    assert len(desk.unlisted_presets) == len(annotated.event_questions().event_types)


def test_an_answer_lights_objects_up_under_their_keys(annotated):
    desk = annotated.desk(WorkspaceClassIndex())
    answer = desk.answer_query(QueryRequest(code=f"an(entity({QueryVariable.OBJECT}))"))
    assert answer.highlight == sorted(entity.key for entity in annotated.objects)


def test_detections_are_answered_with_replay_windows(annotated):
    desk = annotated.desk(WorkspaceClassIndex())
    preset = annotated.event_questions().listed()[0]
    answer = desk.answer_query(QueryRequest(code=preset.code, scope=preset.scope))
    assert answer.count == len(annotated.events)
    assert all(window is not None for window in answer.replay)
