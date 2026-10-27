"""
The bodies of knowledge questions are asked of: named, typed sets of objects and how a
query about them is answered.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from krrood.entity_query_language.evaluable import Evaluable
from typing_extensions import Any, Dict, Iterable, List, Optional, Type

from cramera_protocol.questions import QueryScope

SCOPE_HEADING_SUFFIX = "Queries"
"""
The word a scope's heading ends in, as in "Current State Queries".
"""

# %% domains


@dataclass(frozen=True)
class QueryDomain:
    """
    One ready-made variable: the name a query writes, and what it ranges over.
    """

    name: str
    """
    The name the variable is bound to in a query.
    """

    entity_type: Type[Any]
    """
    The type of the objects the variable ranges over; also in scope under its class
    name.
    """

    objects: Optional[Iterable[Any]] = None
    """
    The objects, iterated anew by every query so a growing or changing source is
    answered with its current content; None when the query's evaluation decides what the
    variable ranges over.
    """


# %% evaluation


class QueryEvaluation(ABC):
    """
    Where the answer to a query is worked out.
    """

    @abstractmethod
    def evaluate(self, expression: Evaluable) -> Any:
        """
        :param expression: A query.
        :return: Its answer.
        """


@dataclass(frozen=True)
class InMemoryEvaluation(QueryEvaluation):
    """
    Answers from the objects the query's domains hold.
    """

    def evaluate(self, expression: Evaluable) -> Any:
        return expression.evaluate()


# %% knowledge


@dataclass
class QueryableKnowledge:
    """
    One body of knowledge questions can be asked of, and how they are answered.
    """

    scope: QueryScope
    """
    Which body of knowledge this is.
    """

    domains: List[QueryDomain]
    """
    The ready-made variables a question about it may range over.
    """

    evaluation: QueryEvaluation = field(default_factory=InMemoryEvaluation)
    """
    Where a question about it is worked out.
    """

    extra_names: Dict[str, Any] = field(default_factory=dict)
    """
    Further names a question about it may use.
    """

    def heading(self) -> str:
        """
        :return: The heading its questions are offered under.
        """
        words = self.scope.value.replace("_", " ").title()
        return f"{words} {SCOPE_HEADING_SUFFIX}"

    def variable_names(self) -> List[str]:
        """
        :return: The names of its ready-made variables.
        """
        return [domain.name for domain in self.domains]
