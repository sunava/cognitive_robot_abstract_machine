"""
The desk answering every kind of question.
"""

from __future__ import annotations

import threading

import pytest

from cramera_protocol.questions import (
    QueryRequest,
    QueryScope,
    QuestionRequest,
    VocabularyKind,
)
from cramera_bridge.questions.desk import Attempt, QuestionDesk
from cramera_bridge.questions.exceptions import UnknownScope, UnknownVocabularyName
from cramera_bridge.questions.knowledge import QueryableKnowledge, QueryDomain
from cramera_bridge.questions.presets import DirectAnswerPreset, Preset
from .dataset.entities import CoordinatesFormat, ShelfItem
from .dataset.files import QueryFile, QuerySource

WAITING_SECONDS = 0.2
"""
How long a test waits to see that a query does not run.
"""

ITEMS = [ShelfItem(name="anvil", size=9.0)]
LISTING = Preset("which items are there?", "an(entity(item))")
BROKEN = Preset("what is broken?", QuerySource(QueryFile.BROKEN).text())
DIRECT = DirectAnswerPreset("where does the anvil go?", "an(entity(item))")
OTHER_SCOPE = Preset(
    "what happened?", "an(entity(item))", scope=QueryScope.DETECTED_EVENTS
)


@pytest.fixture
def desk() -> QuestionDesk:
    """
    :return: A desk over the items, offering questions some of which cannot be worded.
    """
    return QuestionDesk(
        title="shelf",
        knowledge=[
            QueryableKnowledge(
                scope=QueryScope.CURRENT_STATE,
                domains=[QueryDomain("item", ShelfItem, ITEMS)],
            )
        ],
        presets=[LISTING, BROKEN, DIRECT, OTHER_SCOPE],
        unlisted_presets=[Preset("give me all anvils", "an(entity(item))")],
        value_formats=[CoordinatesFormat()],
    )


# %% catalog


def test_the_catalog_words_only_presets_whose_queries_build(desk):
    catalog = desk.preset_catalog()
    worded = {preset.text: preset.verbalization for preset in catalog.presets}
    assert worded[LISTING.text] is not None
    assert (
        worded[BROKEN.text] is worded[DIRECT.text] is worded[OTHER_SCOPE.text] is None
    )
    assert catalog.title == desk.title


def test_the_catalog_lists_each_scope_with_its_variables(desk):
    [scope] = desk.preset_catalog().scopes
    assert scope.name is QueryScope.CURRENT_STATE
    assert scope.variables == ["item"]
    assert scope.label == desk.knowledge[0].heading()


# %% answers


def test_a_query_is_answered(desk):
    answer = desk.answer_query(QueryRequest(code=LISTING.code))
    assert answer.ok and answer.count == len(ITEMS)


def test_a_failing_query_becomes_an_answer_naming_its_error(desk):
    answer = desk.answer_query(
        QueryRequest(code=QuerySource(QueryFile.UNKNOWN_NAME).text())
    )
    assert answer.ok is False
    assert answer.error.startswith(NameError.__name__)
    assert answer.rows == []


def test_a_scope_that_is_not_offered_is_refused(desk):
    with pytest.raises(UnknownScope):
        desk.answer_query(QueryRequest(code="1", scope=QueryScope.EPISODIC_MEMORY))
    with pytest.raises(UnknownScope):
        desk.vocabulary(QueryScope.DETECTED_EVENTS)


def test_the_vocabulary_and_members_of_a_scope(desk):
    entries = desk.vocabulary(QueryScope.CURRENT_STATE).entries
    assert entries[0].kind is VocabularyKind.VARIABLE
    members = desk.members(QueryScope.CURRENT_STATE, "item")
    assert members.name == "item" and members.members
    with pytest.raises(UnknownVocabularyName):
        desk.members(QueryScope.CURRENT_STATE, "nothing")


def test_a_question_matches_listed_and_unlisted_presets(desk):
    assert desk.answer_question(QuestionRequest(text=LISTING.text)).preset.code == (
        LISTING.code
    )
    unlisted = desk.unlisted_presets[0]
    match = desk.answer_question(QuestionRequest(text=unlisted.text))
    assert match.preset.text == unlisted.text


def test_a_query_waits_while_another_holds_the_desk(desk):
    answers = []
    thread = threading.Thread(
        target=lambda: answers.append(
            desk.answer_query(QueryRequest(code=LISTING.code))
        )
    )
    with desk.lock:
        thread.start()
        thread.join(WAITING_SECONDS)
        assert thread.is_alive()
    thread.join()
    assert answers[0].ok


# %% attempts


def test_an_attempt_keeps_the_result_or_the_failure():
    assert Attempt.of(lambda: 3).result == 3
    failed = Attempt.of(lambda: 1 / 0)
    assert not failed.succeeded
    assert failed.error.startswith(ZeroDivisionError.__name__)
