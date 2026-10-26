"""
Ready-made questions and recognizing which one a question asks.
"""

from __future__ import annotations

import json

from cramera_protocol.questions import QueryScope, Verbalization
from cramera_bridge.questions.matching import (
    MINIMUM_SIMILARITY,
    UNMATCHED_QUESTION_REPLY,
    QuestionMatcher,
    SpokenText,
    WordedPreset,
)
from cramera_bridge.questions.presets import (
    EVENT_CLASS_SUFFIX,
    DetectedEventQuestion,
    DetectedEventQuestions,
    DirectAnswerPreset,
    Preset,
    RobotQuestion,
    RobotQuestions,
)

from .dataset.files import EVENT_TYPES_WITH_QUERY_CODE

ROBOT = Preset(RobotQuestion.WHICH_ROBOT, "the(entity(robot))")
ARMS = Preset(RobotQuestion.WHICH_ARMS, "an(entity(arm))")
EVENTS = DetectedEventQuestions.of_types(["PickUpEvent", "PlacingEvent", "PickUpEvent"])

# %% presets


def test_one_robot_is_asked_about_as_the_robot():
    assert [preset.text for preset in RobotQuestions(1).presets()] == [
        RobotQuestion.WHICH_ROBOT,
        RobotQuestion.WHICH_ARMS,
    ]
    assert [preset.text for preset in RobotQuestions(2).presets()] == [
        RobotQuestion.ALL_ROBOTS,
        RobotQuestion.ALL_ARMS,
    ]


def test_detections_offer_one_listed_question_and_one_per_type():
    assert [preset.text for preset in EVENTS.listed()] == [
        DetectedEventQuestion.WHAT_AND_WHEN
    ]
    assert EVENTS.event_types == ("PickUpEvent", "PlacingEvent")
    unlisted = EVENTS.unlisted()
    assert unlisted[0].text == f"give me all pick up {EVENT_CLASS_SUFFIX.lower()}s"
    assert "'PickUpEvent'" in unlisted[0].code
    assert {preset.scope for preset in unlisted} == {QueryScope.DETECTED_EVENTS}


def test_an_event_type_that_is_no_class_name_is_never_put_into_a_query():
    event_types = json.loads(EVENT_TYPES_WITH_QUERY_CODE.read_text(encoding="utf-8"))
    questions = DetectedEventQuestions.of_types(event_types).unlisted()
    valid = [name for name in event_types if name.isidentifier()]
    assert [question.code for question in questions] == [
        question.code for question in DetectedEventQuestions.of_types(valid).unlisted()
    ]


def test_nothing_detected_offers_no_questions():
    assert DetectedEventQuestions().listed() == []


def test_a_direct_answer_is_never_worded_in_advance():
    assert ROBOT.can_be_worded()
    assert not DirectAnswerPreset("where can it go?", "place()").can_be_worded()


def test_a_description_carries_the_wording():
    wording = Verbalization(text="the robot", html="the robot")
    description = ROBOT.description(wording)
    assert (description.text, description.verbalization) == (ROBOT.text, wording)


# %% matching


def test_the_exact_wording_is_recognized_whatever_its_case_and_punctuation():
    matcher = QuestionMatcher([WordedPreset(ROBOT), WordedPreset(ARMS)])
    match = matcher.match("Which ROBOT is this")
    assert match.matched and match.preset.code == ROBOT.code


def test_a_verbalized_wording_is_recognized():
    worded = WordedPreset(ARMS, Verbalization(text="Find an Arm", html="Find an Arm"))
    match = QuestionMatcher([WordedPreset(ROBOT), worded]).match("find an arm please")
    assert match.preset.code == ARMS.code


def test_an_unrelated_question_gets_the_sorry_reply():
    match = QuestionMatcher([WordedPreset(ROBOT), WordedPreset(ARMS)]).match(
        "how is the weather in Bremen"
    )
    assert not match.matched
    assert match.reply == UNMATCHED_QUESTION_REPLY
    assert match.similarity < MINIMUM_SIMILARITY


def test_nothing_on_offer_matches_nothing():
    assert not QuestionMatcher([]).match("which robot is this").matched


def test_the_named_type_decides_and_one_word_spellings_count():
    matcher = QuestionMatcher([WordedPreset(preset) for preset in EVENTS.unlisted()])
    picked = matcher.match("give me all pickup events")
    placed = matcher.match("give me the placing events")
    assert picked.preset.code == EVENTS.unlisted()[0].code
    assert placed.preset.code == EVENTS.unlisted()[1].code


def test_words_are_read_without_case_and_punctuation():
    assert SpokenText("Pick-up, now!").words() == ["pick", "up", "now"]
    assert "pickup" in SpokenText("pick up").spellings()
