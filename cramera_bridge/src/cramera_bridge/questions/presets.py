"""
Ready-made questions: a label for a button and the query answering it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Iterable, List, Optional, Tuple

from cramera_protocol.questions import PresetDescription, QueryScope, Verbalization

WORD_START = re.compile(r"(?<!^)(?=[A-Z])")
"""
Where a word starts inside a class name, which is where a space goes when the name is
said out loud.
"""

# %% presets


@dataclass
class Preset:
    """
    One ready-made question and the query answering it.
    """

    text: str
    """
    The question as its button shows it.
    """

    code: str
    """
    The query answering it.
    """

    scope: QueryScope = QueryScope.CURRENT_STATE
    """
    The body of knowledge it asks.
    """

    requires_live: bool = False
    """
    Whether answering it needs a running world.
    """

    def can_be_worded(self) -> bool:
        """
        :return: Whether its query may be built to read it back as English before it is
            asked.
        """
        return True

    def description(self, verbalization: Optional[Verbalization]) -> PresetDescription:
        """
        :param verbalization: Its query read back as English, if it was.
        :return: The preset as the protocol describes it.
        """
        return PresetDescription(
            text=self.text,
            code=self.code,
            requires_live=self.requires_live,
            scope=self.scope,
            verbalization=verbalization,
        )


@dataclass
class DirectAnswerPreset(Preset):
    """
    A question whose query does work when built, such as a geometric computation, so it
    is neither built nor worded before it is asked.
    """

    def can_be_worded(self) -> bool:
        return False


# %% robot questions


class RobotQuestion(StrEnum):
    """
    The questions about a scene's robots.
    """

    WHICH_ROBOT = "which robot is this?"
    """
    Names the scene's one robot.
    """

    WHICH_ARMS = "which arm does it have?"
    """
    Lists the arms of the scene's one robot.
    """

    ALL_ROBOTS = "which robots are in this scene?"
    """
    Lists every robot of a scene with several.
    """

    ALL_ARMS = "which arms belong to each robot?"
    """
    Lists the arms of every robot of a scene with several.
    """


class QueryVariable(StrEnum):
    """
    The names of the ready-made variables questions are written with.
    """

    ROBOT = "robot"
    """
    The robots.
    """

    ARM = "arm"
    """
    The arms.
    """

    GRIPPER = "gripper"
    """
    The grippers.
    """

    OBJECT = "scene_object"
    """
    The free objects.
    """

    STEP = "episode"
    """
    The recorded steps of the run.
    """

    JOINT = "joint"
    """
    The joints and how far they moved.
    """

    PLAN_NODE = "plan_node"
    """
    The nodes of the executed plan.
    """

    EVENT = "event"
    """
    The detected events.
    """


@dataclass(frozen=True)
class EntityListing:
    """
    Writes the query listing every entity a variable ranges over.
    """

    variable: str
    """
    The variable.
    """

    def every(self) -> str:
        """
        :return: The query listing every entity.
        """
        return f"an(entity({self.variable}))"

    def the_one(self) -> str:
        """
        :return: The query naming the one entity.
        """
        return f"the(entity({self.variable}))"


@dataclass(frozen=True)
class RobotQuestions:
    """
    The robot questions suited to how many robots a scene holds.
    """

    robot_count: int
    """
    How many robots the scene holds.
    """

    def presets(self) -> List[Preset]:
        """
        :return: Questions about the one robot, or about every robot of several.
        """
        robots = EntityListing(QueryVariable.ROBOT)
        arms = EntityListing(QueryVariable.ARM)
        if self.robot_count <= 1:
            return [
                Preset(RobotQuestion.WHICH_ROBOT, robots.the_one()),
                Preset(RobotQuestion.WHICH_ARMS, arms.every()),
            ]
        return [
            Preset(RobotQuestion.ALL_ROBOTS, robots.every()),
            Preset(RobotQuestion.ALL_ARMS, arms.every()),
        ]


# %% one question per type


@dataclass(frozen=True)
class PresetsPerType:
    """
    The same question asked once per type a record can be, such as "give me all pick up
    events" for every kind of event.
    """

    class_suffix: str
    """
    The word every one of the types' class names ends in, such as ``Event``.
    """

    class_names: Tuple[str, ...]
    """
    The class name of every type a question may name.
    """

    code: str
    """
    The query, with ``{type_name}`` where the class name goes.
    """

    scope: QueryScope = QueryScope.CURRENT_STATE
    """
    The body of knowledge the questions ask.
    """

    def questions(self) -> List[Preset]:
        """
        :return: One question per type, worded the way it is asked out loud; a name
            that is no class name, which a crafted bundle could list to put code into
            the query, gets none.
        """
        return [
            Preset(
                f"give me all {self.spoken_type(name)} {self.class_suffix.lower()}s",
                self.code.format(type_name=name),
                scope=self.scope,
            )
            for name in self.class_names
            if name.isidentifier()
        ]

    def spoken_type(self, class_name: str) -> str:
        """
        :param class_name: A type's class name, such as ``PickUpEvent``.
        :return: The type as a question names it, such as ``pick up``.
        """
        return WORD_START.sub(" ", class_name.removesuffix(self.class_suffix)).lower()


# %% detected events


class DetectedEventQuestion(StrEnum):
    """
    The questions about detected events.
    """

    WHAT_AND_WHEN = "what was detected, and when?"
    """
    Lists every detection with its type and moment.
    """


EVENT_CLASS_SUFFIX = "Event"
"""
The word every detected event's class name ends in.
"""


@dataclass(frozen=True)
class DetectedEventQuestions:
    """
    The questions about detected events, for the event types on offer.
    """

    event_types: Tuple[str, ...] = field(default_factory=tuple)
    """
    The class name of every event type a question may name.
    """

    @classmethod
    def of_types(cls, event_types: Iterable[str]) -> DetectedEventQuestions:
        """
        :param event_types: The types of the detections on offer, with repetitions.
        :return: The questions about those types.
        """
        return cls(tuple(sorted(set(event_types))))

    def listed(self) -> List[Preset]:
        """
        :return: The questions shown as buttons; none when nothing was detected.
        """
        if not self.event_types:
            return []
        event = QueryVariable.EVENT.value
        return [
            Preset(
                DetectedEventQuestion.WHAT_AND_WHEN,
                f"set_of({event}.name, {event}.event_type, {event}.timestamp)",
                scope=QueryScope.DETECTED_EVENTS,
            )
        ]

    def unlisted(self) -> List[Preset]:
        """
        :return: One question per event type, recognized when asked but not shown.
        """
        event = QueryVariable.EVENT.value
        return PresetsPerType(
            class_suffix=EVENT_CLASS_SUFFIX,
            class_names=self.event_types,
            code=f"an(entity({event}).where({event}.event_type == '{{type_name}}'))",
            scope=QueryScope.DETECTED_EVENTS,
        ).questions()
