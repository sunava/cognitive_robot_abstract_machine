"""
Recording the answers of recorded bundles for viewers without krrood.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.questions import GraphViewName, QueryRequest, RecordedAnswers
from cramera_bridge.questions.answer_recording import AnswerRecorder, RecordOption, main
from cramera_bridge.questions.recorded.bundle import RecordedBundle
from cramera_bridge.questions.recorded.knowledge_base import RecordedKnowledge
from cramera_bridge.questions.workspace_classes import WorkspaceClassIndex

from .dataset.files import SceneName

RECORDING_TIME = datetime(2026, 10, 23, 9, 30, 15, 123456, tzinfo=timezone.utc)
"""
When the tests pretend to record.
"""


@pytest.fixture
def recorder(scenes) -> AnswerRecorder:
    """
    :return: A recorder of the dataset's scenes without the workspace's classes.
    """
    return AnswerRecorder(
        scenes, class_index=WorkspaceClassIndex(), clock=lambda: RECORDING_TIME
    )


def test_every_bundle_of_the_directory_is_offered(recorder):
    assert sorted(recorder.scene_names()) == sorted(scene.value for scene in SceneName)


def test_the_recorded_answers_are_the_desks_answers(recorder):
    recorded = recorder.record(SceneName.ANNOTATED)
    knowledge = RecordedKnowledge.of_bundle(
        RecordedBundle.of_catalog(recorder.catalog(), SceneName.ANNOTATED)
    )
    desk = knowledge.desk(recorder.class_index)
    assert recorded.presets == desk.preset_catalog()
    offline = [
        preset for preset in recorded.presets.presets if not preset.requires_live
    ]
    assert [(answer.code, answer.scope) for answer in recorded.answers] == [
        (preset.code, preset.scope) for preset in offline
    ]
    for answer in recorded.answers:
        request = QueryRequest(
            code=answer.code, scope=answer.scope, scene=SceneName.ANNOTATED
        )
        assert answer.answer == desk.answer_query(request)


def test_every_graph_view_is_recorded(recorder):
    recorded = recorder.record(SceneName.ANNOTATED)
    assert [graph.view for graph in recorded.graphs] == list(GraphViewName)
    assert recorded.graph(GraphViewName.KINEMATICS) == recorder.graph_views.build(
        GraphViewName.KINEMATICS.value,
        RecordedKnowledge.of_bundle(
            RecordedBundle.of_catalog(recorder.catalog(), SceneName.ANNOTATED)
        ),
    )


def test_the_recording_time_is_stamped_in_whole_seconds(recorder):
    assert recorder.record(SceneName.LEGACY).recorded_at == "2026-10-23T09:30:15Z"


def test_the_command_writes_the_answers_into_each_named_bundle(scenes):
    assert main([str(scenes), RecordOption.SCENE.value, SceneName.LEGACY.value]) == 0
    written = scenes / SceneName.LEGACY / BundleFile.RECORDED_ANSWERS
    recorded = RecordedAnswers.from_json(json.loads(written.read_text("utf-8")))
    assert recorded.scene == SceneName.LEGACY
    assert not (scenes / SceneName.ANNOTATED / BundleFile.RECORDED_ANSWERS).exists()


def test_the_command_records_every_bundle_without_names(scenes):
    assert main([str(scenes)]) == 0
    for scene in SceneName:
        assert (scenes / scene / BundleFile.RECORDED_ANSWERS).is_file()
