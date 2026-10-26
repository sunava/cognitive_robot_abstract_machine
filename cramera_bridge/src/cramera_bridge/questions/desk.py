"""
The desk answering every kind of question about one world or one recorded scene.

Queries are arbitrary code a person wrote and run with ``exec`` and ``eval``. The desk
is the one place where a failure of that code becomes an answer saying what went wrong;
a request naming something the desk does not offer is refused with an error instead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from typing_extensions import Any, Callable, Generic, List, Optional, Sequence, TypeVar

from cramera_protocol.questions import (
    Members,
    PresetCatalog,
    QueryAnswer,
    QueryRequest,
    QueryScope,
    QuestionMatch,
    QuestionRequest,
    ScopeDescription,
    Verbalization,
    Vocabulary,
)
from cramera_bridge.questions.exceptions import UnknownScope
from cramera_bridge.questions.highlights import (
    EntityNameKeys,
    HighlightKeys,
    ValueFormat,
)
from cramera_bridge.questions.knowledge import QueryableKnowledge
from cramera_bridge.questions.matching import QuestionMatcher, WordedPreset
from cramera_bridge.questions.presets import Preset
from cramera_bridge.questions.runner import (
    DEFAULT_ROW_LIMIT,
    AnswerRenderer,
    QueryRunner,
)
from cramera_bridge.questions.verbalization import QueryVerbalizer
from cramera_bridge.questions.vocabulary import QueryVocabulary
from cramera_bridge.questions.workspace_classes import WorkspaceClassIndex

__all__ = ["HighlightKeys", "QuestionDesk", "Attempt"]

Outcome = TypeVar("Outcome")
"""
What an attempted piece of query code returns.
"""

# %% attempts


@dataclass(frozen=True)
class Attempt(Generic[Outcome]):
    """
    The outcome of running query code: its result, or the failure it raised.
    """

    result: Optional[Outcome] = None
    """
    What the code returned, when it succeeded.
    """

    error: Optional[str] = None
    """
    The failure as ``<ExceptionType>: <message>``, when it failed.
    """

    @classmethod
    def of(cls, action: Callable[[], Outcome]) -> Attempt[Outcome]:
        """
        Run query code, turning any failure into data.

        Query code is whatever a person typed, so any exception it raises is part of its
        answer rather than a fault of the desk; this is the only place such failures are
        caught.

        :param action: Runs the query code.
        :return: The result or the failure.
        """
        try:
            return cls(result=action())
        except Exception as failure:
            return cls(error=f"{type(failure).__name__}: {failure}")

    @property
    def succeeded(self) -> bool:
        """
        Whether the code ran without raising.
        """
        return self.error is None


# %% desk


@dataclass
class QuestionDesk:
    """
    Answers queries and questions about the bodies of knowledge it offers.
    """

    title: str
    """
    What the questions are about.
    """

    knowledge: List[QueryableKnowledge]
    """
    The bodies of knowledge on offer, in the order their questions are shown.
    """

    presets: List[Preset]
    """
    The ready-made questions shown as buttons.
    """

    unlisted_presets: List[Preset] = field(default_factory=list)
    """
    Ready-made questions recognized when asked but not shown.
    """

    highlight_keys: HighlightKeys = field(default_factory=EntityNameKeys)
    """
    Maps answer values to the viewer keys they light up.
    """

    class_index: WorkspaceClassIndex = field(default_factory=WorkspaceClassIndex)
    """
    The workspace classes a query may name.
    """

    row_limit: int = DEFAULT_ROW_LIMIT
    """
    The most rows an answer holds.
    """

    value_formats: Sequence[ValueFormat] = field(default_factory=list)
    """
    Formats for answer values whose own representation is unreadable.
    """

    verbalizer: QueryVerbalizer = field(default_factory=QueryVerbalizer)
    """
    Reads queries back as English.
    """

    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    """
    Serializes building and evaluating queries: krrood's symbol graph is not safe to use
    from several threads at once.
    """

    # %% bodies of knowledge

    def knowledge_of(self, scope: QueryScope) -> QueryableKnowledge:
        """
        :param scope: A body of knowledge.
        :return: The knowledge of that scope.
        :raises UnknownScope: When the desk offers none.
        """
        for knowledge in self.knowledge:
            if knowledge.scope is scope:
                return knowledge
        raise UnknownScope(
            scope.value, [knowledge.scope.value for knowledge in self.knowledge]
        )

    def runner_of(self, knowledge: QueryableKnowledge) -> QueryRunner:
        """
        :param knowledge: A body of knowledge the desk offers.
        :return: A runner over its variables.
        """
        return QueryRunner(
            domains=knowledge.domains,
            extra_names=knowledge.extra_names,
            evaluation=knowledge.evaluation,
            class_index=self.class_index,
            verbalizer=self.verbalizer,
        )

    def vocabulary_of(self, scope: QueryScope) -> QueryVocabulary:
        """
        :param scope: A body of knowledge.
        :return: The names a query of it may use.
        :raises UnknownScope: When the desk offers no such knowledge.
        """
        knowledge = self.knowledge_of(scope)
        return QueryVocabulary(
            domains=knowledge.domains,
            extra_names=knowledge.extra_names,
            class_index=self.class_index,
        )

    # %% answers

    def preset_catalog(self) -> PresetCatalog:
        """
        :return: The ready-made questions, each read back as English where its query
            builds, and the bodies of knowledge on offer.
        """
        return PresetCatalog(
            title=self.title,
            presets=[
                worded.preset.description(worded.verbalization)
                for worded in self.worded_presets()
            ],
            scopes=[
                ScopeDescription(
                    name=knowledge.scope,
                    label=knowledge.heading(),
                    variables=knowledge.variable_names(),
                )
                for knowledge in self.knowledge
            ],
        )

    def vocabulary(self, scope: QueryScope) -> Vocabulary:
        """
        :param scope: A body of knowledge.
        :return: The names a query of it may use.
        :raises UnknownScope: When the desk offers no such knowledge.
        """
        return Vocabulary(entries=self.vocabulary_of(scope).entries())

    def members(self, scope: QueryScope, name: str) -> Members:
        """
        :param scope: A body of knowledge.
        :param name: A type's name or a variable's.
        :return: The members that may follow the name's dot.
        :raises UnknownScope: When the desk offers no such knowledge.
        :raises UnknownVocabularyName: When no type of that name is in reach.
        """
        return Members(name=name, members=self.vocabulary_of(scope).members_of(name))

    def answer_query(self, request: QueryRequest) -> QueryAnswer:
        """
        :param request: A query.
        :return: Its answer, or the failure of its code as an answer that is not ok.
        :raises UnknownScope: When the desk offers no knowledge of the request's scope.
        """
        knowledge = self.knowledge_of(request.scope)
        runner = self.runner_of(knowledge)
        renderer = AnswerRenderer(
            entity_types=runner.entity_types(),
            highlight_keys=self.highlight_keys,
            value_formats=self.value_formats,
            limit=self.row_limit,
        )
        with self.lock:
            attempt = Attempt.of(lambda: runner.answer(request.code, renderer))
        if not attempt.succeeded:
            return QueryAnswer(ok=False, error=attempt.error)
        return attempt.result

    def answer_question(self, request: QuestionRequest) -> QuestionMatch:
        """
        :param request: A plain-language question.
        :return: The ready-made question it asks, or the reply that none matches.
        """
        offered = self.worded_presets() + [
            WordedPreset(preset) for preset in self.unlisted_presets
        ]
        return QuestionMatcher(offered).match(request.text)

    # %% wording

    def worded_presets(self) -> List[WordedPreset]:
        """
        :return: The shown ready-made questions with their queries read back as English.
        """
        return [
            WordedPreset(preset, self.verbalization_of(preset))
            for preset in self.presets
        ]

    def verbalization_of(self, preset: Preset) -> Optional[Verbalization]:
        """
        :param preset: A ready-made question.
        :return: Its query read back as English, or None when it may not be built, its
            scope is not offered or its code does not build into a query.
        """
        offered = {knowledge.scope for knowledge in self.knowledge}
        if not preset.can_be_worded() or preset.scope not in offered:
            return None
        runner = self.runner_of(self.knowledge_of(preset.scope))
        with self.lock:
            attempt: Attempt[Any] = Attempt.of(lambda: runner.verbalize(preset.code))
        return attempt.result
