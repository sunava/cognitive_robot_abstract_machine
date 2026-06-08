"""
EQL verbalizer — coordinator and one-shot convenience entry point.

:class:`EQLVerbalizer` dispatches an EQL expression tree to the rule engine and
returns a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
tree.  :func:`verbalize_expression` is the simplest entry point — it returns a plain
English string with no colour markup.

For coloured / hierarchical output use
:class:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.verbalization.context import VerbalizationContext
from krrood.entity_query_language.verbalization.fragments.base import VerbFragment, flatten_fragment_to_plain_text
from krrood.entity_query_language.verbalization.rule_engine import RuleEngine
from krrood.entity_query_language.verbalization.rules.registry import ALL_RULES


@dataclass
class EQLVerbalizer:
    """
    Coordinator that maps an EQL expression tree to a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment` tree.

    Dispatches via a :class:`~krrood.entity_query_language.verbalization.rule_engine.RuleEngine` of
    :class:`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule` classes.
    Each rule declares its guard in :meth:`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule.applies`
    and its rendering in :meth:`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule.transform`.
    More-specific subclasses are tried before their parents (MRO-depth priority).

    For simple plain-text output use :func:`verbalize_expression`.
    For coloured / formatted output build a
    :class:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`.
    """

    _engine: RuleEngine = field(init=False, repr=False)
    """Rule dispatcher; sorts rules by MRO depth before first call."""

    def __post_init__(self) -> None:
        self._engine = RuleEngine(ALL_RULES)

    def build(
        self,
        expression: SymbolicExpression,
        context: Optional[VerbalizationContext] = None,
    ) -> VerbFragment:
        """
        Translate *expression* into a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment` tree.

        A fresh :class:`~krrood.entity_query_language.verbalization.context.VerbalizationContext`
        (with a pre-built disambiguation map) is created when *context* is ``None``.

        :param expression: Any EQL symbolic expression.
        :type expression: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param context: Shared verbalization state; created automatically when omitted.
        :type context: ~krrood.entity_query_language.verbalization.context.VerbalizationContext or None
        :return: Root of the fragment tree representing *expression* in natural language.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        if context is None:
            context = VerbalizationContext.from_expression(expression)
        return self._engine.build(expression, context, self)

    def verbalize(
        self,
        expression: SymbolicExpression,
        context: Optional[VerbalizationContext] = None,
    ) -> str:
        """
        Translate *expression* into a plain-text English string.

        Equivalent to ``flatten_fragment_to_plain_text(self.build(expression, context))`` — no colour markup.
        Prefer :class:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`
        when colour or hierarchical layout is needed.

        :param expression: Any EQL symbolic expression.
        :type expression: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param context: Shared verbalization state; created automatically when omitted.
        :type context: ~krrood.entity_query_language.verbalization.context.VerbalizationContext or None
        :return: Plain-text natural-language representation of *expression*.
        :rtype: str
        """
        return flatten_fragment_to_plain_text(self.build(expression, context))


# Shared plain-text verbalizer instance — EQLVerbalizer is stateless, so this
# single instance is safe to reuse across calls.
_verbalizer = EQLVerbalizer()


def verbalize_expression(expression) -> str:
    """
    Verbalize any EQL expression into a plain-text English phrase.

    This is the simplest entry point — it returns plain text with no colour markup.
    For coloured, hierarchical, or hyperlinked output use
    :class:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`
    directly:

    * ``VerbalizationPipeline.plain().verbalize(expr)`` — plain prose (equivalent to this function).
    * ``VerbalizationPipeline.ansi().verbalize(expr)`` — ANSI-coloured prose.
    * ``VerbalizationPipeline.html().verbalize(expr)`` — HTML ``<span>`` coloured prose.

    :param expression: Any EQL expression or :class:`~krrood.entity_query_language.query.query.Query`.
    :return: Plain-text natural-language string.
    :rtype: str
    """
    if isinstance(expression, Query):
        expression.build()
    return _verbalizer.verbalize(expression)
