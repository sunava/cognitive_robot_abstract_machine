"""
Verbalization context — a thin facade composing the microplanning services
threaded through a single
:meth:`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer.build`
call.

The per-pass state is split by concern into three single-responsibility services
(mirroring the microplanning subtasks of Reiter & Dale 2000):

* :class:`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions`
  — coreference, articles, disambiguation, pronouns.
* :class:`~krrood.entity_query_language.verbalization.microplanning.binding_scope.BindingScope`
  — deferred-constraint frames and field-reference overrides.
* :class:`~krrood.entity_query_language.verbalization.microplanning.config.RenderConfig`
  — render-mode flags (query depth, compact predicates).

:class:`VerbalizationContext` keeps a backward-compatible surface (the fields the
rules read directly and the methods they call) by delegating to these services,
so each concern can be reasoned about and tested in isolation.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from typing_extensions import Any

from krrood.entity_query_language.verbalization.microplanning.binding_scope import (
    BindingScope,
)
from krrood.entity_query_language.verbalization.microplanning.config import RenderConfig
from krrood.entity_query_language.verbalization.microplanning.referring import (
    ReferringExpressions,
)

__all__ = ["VerbalizationContext"]


@dataclass
class VerbalizationContext:
    """
    Facade holding the three microplanning services for one verbalization pass.

    Rules reach the services through their :class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.Ctx`
    (``ctx.refer`` / ``ctx.scope`` / ``ctx.config``), so this context exposes only the service
    objects themselves (:attr:`referring`, :attr:`binding`, :attr:`config`) plus the one
    cross-cutting helper that belongs to no single service (:meth:`type_name_of_value`).  Create
    via :meth:`from_expression` to pre-load the disambiguation map.

    * :class:`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions`
      — coreference, articles, disambiguation, pronouns.
    * :class:`~krrood.entity_query_language.verbalization.microplanning.binding_scope.BindingScope`
      — deferred-constraint frames and field-reference overrides.
    * :class:`~krrood.entity_query_language.verbalization.microplanning.config.RenderConfig`
      — render-mode flags (query depth, compact predicates).
    """

    referring: ReferringExpressions = field(default_factory=ReferringExpressions)
    """Coreference / article / disambiguation / pronoun service."""

    binding: BindingScope = field(default_factory=BindingScope)
    """Deferred-constraint frames and field-reference overrides."""

    config: RenderConfig = field(default_factory=RenderConfig)
    """Render-mode flags (query depth, compact predicates)."""

    @classmethod
    def from_expression(cls, expression) -> VerbalizationContext:
        """
        Create a context with the disambiguation map pre-built for *expression*.

        :param expression: Root EQL expression or Query to scan.
        :return: A fresh context whose referring service has its disambiguation map populated.
        :rtype: VerbalizationContext
        """
        return cls(referring=ReferringExpressions.from_expression(expression))

    # ── Value lexicalisation ─────────────────────────────────────────────────

    def type_name_of_value(self, value: Any) -> str:
        """
        Render a Python value as a human-readable string.

        * A bare ``type`` → its ``__name__`` (``Apple`` → ``"Apple"``).
        * A tuple of types → ``"A or B or C"``.
        * A ``datetime`` with no time → ``"May 23, 2026"``; with a time →
          ``"May 23, 2026 at 14:30"``.
        * Anything else → ``repr(value)``.

        :param value: Python value from a
            :class:`~krrood.entity_query_language.core.variable.Literal` node.
        :return: Human-readable string representation.
        :rtype: str
        """
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, tuple) and all(
            isinstance(variable, type) for variable in value
        ):
            return " or ".join(variable.__name__ for variable in value)
        if isinstance(value, datetime.datetime):
            if value.time() == datetime.time.min:
                return value.strftime("%B %-d, %Y")
            return value.strftime("%B %-d, %Y at %H:%M")
        return repr(value)
