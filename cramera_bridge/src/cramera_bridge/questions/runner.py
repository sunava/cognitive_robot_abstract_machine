"""
Building, evaluating and rendering one query over a set of declared domains.

The runner knows nothing about where its domains come from: a recorded scene and a
running world are queried the same way.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import datetime

from krrood.entity_query_language import factories
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.scope import eql_factory_namespace
from typing_extensions import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from cramera_protocol.questions import (
    AnswerColumn,
    AnswerKind,
    QueryAnswer,
    ReplayWindow,
    Verbalization,
)
from cramera_bridge.questions.entity import NamedEntity
from cramera_bridge.questions.exceptions import EmptyQuery
from cramera_bridge.questions.highlights import (
    EntityNameKeys,
    HighlightKeys,
    ValueFormat,
)
from cramera_bridge.questions.knowledge import (
    InMemoryEvaluation,
    QueryDomain,
    QueryEvaluation,
)
from cramera_bridge.questions.replay import ReplayMoment
from cramera_bridge.questions.verbalization import QueryVerbalizer
from cramera_bridge.questions.workspace_classes import (
    WorkspaceClassIndex,
    WorkspaceClassNamespace,
)

DEFAULT_ROW_LIMIT = 200
"""
The most rows an answer holds unless asked for fewer.
"""

VALUE_PRECISION = 4
"""
Decimal places of a number in an answer.
"""

QUERY_FILE_NAME = "<query>"
"""
The file name errors in a query's source are reported under.
"""

RESULT_NAME = "result"
"""
The name a query that ends in a statement binds its answer to.
"""

ATTRIBUTE_SEPARATOR = "."
"""
Separates an owner's name from an attribute's in a selected column's name.
"""

NAME_FIELD = "name"
"""
The field that titles an entity row instead of being one of its columns.
"""

# %% rows


@dataclass
class AnswerRow:
    """
    One rendered row and the stretch of a recording worth replaying for it.
    """

    values: Dict[str, Any]
    """
    The row's columns.
    """

    replay: Optional[ReplayWindow] = None
    """
    The stretch worth replaying, or None for a row without a moment.
    """


@dataclass
class AnswerRenderer:
    """
    Renders one evaluated answer into rows, collecting what the viewer lights up.
    """

    entity_types: Tuple[Type[Any], ...] = ()
    """
    Types whose instances are rows of their own, titled by their name.
    """

    highlight_keys: HighlightKeys = field(default_factory=EntityNameKeys)
    """
    Maps answer values to the viewer keys they light up.
    """

    value_formats: Sequence[ValueFormat] = ()
    """
    Formats for values whose own representation is unreadable.
    """

    limit: int = DEFAULT_ROW_LIMIT
    """
    The most rows rendered.
    """

    highlight: List[str] = field(default_factory=list)
    """
    The viewer keys collected while rendering.
    """

    def answer(
        self, result: Any, verbalization: Optional[Verbalization] = None
    ) -> QueryAnswer:
        """
        :param result: An evaluated answer or an inspected value.
        :param verbalization: The query read back as English.
        :return: The answer document.
        """
        rows, more = self.rows_of(result)
        kind = (
            AnswerKind.ENTITIES
            if rows and AnswerColumn.ENTITY in rows[0].values
            else AnswerKind.ROWS
        )
        return QueryAnswer(
            kind=kind,
            rows=[row.values for row in rows],
            count=len(rows),
            more=more,
            highlight=sorted(set(self.highlight)),
            replay=[row.replay for row in rows],
            verbalization=verbalization,
        )

    def rows_of(self, result: Any) -> Tuple[List[AnswerRow], bool]:
        """
        :param result: An evaluated answer or an inspected value.
        :return: The rows, and whether rows were left out at the limit.
        """
        if result is None:
            return [], False
        if self.is_entity(result) or not self.is_collection(result):
            return [self.row_of(result)], False
        rows: List[AnswerRow] = []
        for item in result:
            if len(rows) >= self.limit:
                return rows, True
            rows.append(self.row_of(item))
        return rows, False

    @staticmethod
    def is_entity(value: Any) -> bool:
        """
        :param value: An answer value.
        :return: Whether it is a dataclass instance.
        """
        return is_dataclass(value) and not isinstance(value, type)

    @staticmethod
    def is_collection(value: Any) -> bool:
        """
        :param value: An answer value.
        :return: Whether its items are the rows: an iterable that is no text or mapping.
        """
        return isinstance(value, Iterable) and not isinstance(
            value, (str, bytes, Mapping)
        )

    def row_of(self, item: Any) -> AnswerRow:
        """
        :param item: One answer item.
        :return: Its row.
        """
        if self.is_entity(item):
            return self.entity_row(item)
        if isinstance(item, Mapping):
            return self.selection_row(item)
        return AnswerRow({AnswerColumn.VALUE.value: self.text_of(item)})

    def selection_row(self, item: Mapping) -> AnswerRow:
        """
        :param item: The values a ``set_of`` query selected, by their expressions.
        :return: One column per selected value.
        """
        columns = self.column_names([str(key) for key in item])
        values: Dict[str, Any] = {}
        replay: Optional[ReplayWindow] = None
        for column, value in zip(columns, item.values()):
            self.highlight.extend(self.highlight_keys.keys_of(value))
            if replay is None and isinstance(value, datetime):
                replay = ReplayMoment(value).window()
            values[column] = self.text_of(value)
        return AnswerRow(values, replay)

    @staticmethod
    def column_names(keys: List[str]) -> List[str]:
        """
        :param keys: The selected values' names, such as ``Robot.name``.
        :return: The names without their owner while they stay distinct.
        """
        shortened = [key.rsplit(ATTRIBUTE_SEPARATOR, 1)[-1] for key in keys]
        return shortened if len(set(shortened)) == len(keys) else keys

    def entity_row(self, item: Any) -> AnswerRow:
        """
        :param item: A dataclass instance.
        :return: Its row: its title, its type and its public fields.
        """
        self.highlight.extend(self.highlight_keys.keys_of(item))
        values: Dict[str, Any] = {
            AnswerColumn.ENTITY.value: self.title_of(item) or repr(item),
            AnswerColumn.TYPE.value: type(item).__name__,
        }
        instance_values = vars(item)
        replay: Optional[ReplayWindow] = None
        for entity_field in fields(item):
            if entity_field.name == NAME_FIELD or not entity_field.repr:
                continue
            value = (
                instance_values[entity_field.name]
                if entity_field.name in instance_values
                else entity_field.default
            )
            if value is MISSING:
                continue
            if replay is None and isinstance(value, datetime):
                replay = ReplayMoment(value).window()
            values[entity_field.name] = self.text_of(value)
        return AnswerRow(values, replay)

    def title_of(self, value: Any) -> Optional[str]:
        """
        :param value: An answer value.
        :return: The name of a named entity or of an entity of a declared type, or
            None.
        """
        if isinstance(value, NamedEntity) or isinstance(value, self.entity_types):
            return str(value.name)
        return None

    def text_of(self, value: Any) -> Any:
        """
        :param value: An answer value.
        :return: The value as JSON: numbers, truth values and texts as they are, times
            as text, entities by their title, anything else by a format or its
            representation.
        """
        for value_format in self.value_formats:
            if value_format.accepts(value):
                return value_format.text(value)
        if isinstance(value, datetime):
            return value.isoformat(sep=" ", timespec="seconds")
        if isinstance(value, float):
            return round(value, VALUE_PRECISION)
        if isinstance(value, str):
            self.highlight.extend(self.highlight_keys.keys_of(value))
            return value
        if isinstance(value, (bool, int)) or value is None:
            return value
        if self.is_entity(value):
            self.highlight.extend(self.highlight_keys.keys_of(value))
            return self.title_of(value) or repr(value)
        return repr(value)


# %% runner


@dataclass
class QueryRunner:
    """
    Builds queries in a fresh namespace of ready-made variables and answers them.
    """

    domains: List[QueryDomain]
    """
    The ready-made variables.
    """

    extra_names: Dict[str, Any] = field(default_factory=dict)
    """
    Further names a query may use.
    """

    evaluation: QueryEvaluation = field(default_factory=InMemoryEvaluation)
    """
    Where a query is worked out.
    """

    class_index: WorkspaceClassIndex = field(default_factory=WorkspaceClassIndex)
    """
    The workspace classes a query may name.
    """

    verbalizer: QueryVerbalizer = field(default_factory=QueryVerbalizer)
    """
    Reads queries back as English.
    """

    def entity_types(self) -> Tuple[Type[Any], ...]:
        """
        :return: The types the domains range over.
        """
        return tuple(domain.entity_type for domain in self.domains)

    def namespace(self) -> WorkspaceClassNamespace:
        """
        :return: A namespace with fresh variables, so no two queries share one.
        """
        namespace = WorkspaceClassNamespace(index=self.class_index)
        namespace.update(eql_factory_namespace())
        for domain in self.domains:
            namespace[domain.entity_type.__name__] = domain.entity_type
        for domain in self.domains:
            namespace[domain.name] = factories.variable(
                domain.entity_type, domain=domain.objects
            )
        namespace.update(self.extra_names)
        return namespace

    def build(self, code: str) -> Any:
        """
        Run the source's statements and return its last expression unevaluated.

        :param code: Query source; earlier statements set the query up, and a source
            ending in a statement answers with what it bound to ``result``.
        :return: The query expression or the inspected value.
        :raises EmptyQuery: When the source holds no statement.
        """
        namespace = self.namespace()
        tree = ast.parse(code, filename=QUERY_FILE_NAME, mode="exec")
        if not tree.body:
            raise EmptyQuery()
        last = tree.body[-1]
        if not isinstance(last, ast.Expr):
            exec(compile(tree, QUERY_FILE_NAME, "exec"), namespace)
            return namespace.get(RESULT_NAME)
        setup = ast.Module(body=tree.body[:-1], type_ignores=[])
        exec(compile(setup, QUERY_FILE_NAME, "exec"), namespace)
        expression = ast.Expression(last.value)
        return eval(compile(expression, QUERY_FILE_NAME, "eval"), namespace)

    def verbalize(self, code: str) -> Optional[Verbalization]:
        """
        :param code: Query source.
        :return: The query read back as English, or None when the source builds no
            query or krrood cannot word it.
        """
        built = self.build(code)
        if not isinstance(built, Evaluable):
            return None
        return self.verbalizer.verbalize(built)

    def answer(self, code: str, renderer: AnswerRenderer) -> QueryAnswer:
        """
        :param code: Query source.
        :param renderer: Renders the answer; it should range over :meth:`entity_types`.
        :return: The answer to the query, or the inspected value as rows.
        """
        built = self.build(code)
        if not isinstance(built, Evaluable):
            return renderer.answer(built)
        verbalization = self.verbalizer.verbalize(built)
        return renderer.answer(self.evaluation.evaluate(built), verbalization)
