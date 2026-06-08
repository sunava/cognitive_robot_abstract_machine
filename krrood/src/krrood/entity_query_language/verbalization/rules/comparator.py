"""
Verbalization rule for :class:`~krrood.entity_query_language.operators.comparator.Comparator`
expressions — *"<left> <operator> <right>"*.

Operator selection is delegated to
:func:`~krrood.entity_query_language.verbalization.operator_phrase.comparator_phrase`,
which handles calculation-equality, temporality, negation, and compactness declaratively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.operator_phrase import comparator_phrase
from krrood.entity_query_language.verbalization.rule_engine import VerbalizationRule

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer


class ComparatorRule(VerbalizationRule):
    """
    Verbalizes :class:`~krrood.entity_query_language.operators.comparator.Comparator`
    expressions as *"<left> <operator> <right>"*.

    Operator selection is delegated to
    :func:`~krrood.entity_query_language.verbalization.operator_phrase.comparator_phrase`,
    which handles calc-equality, temporality, negation, and compactness declaratively.

    Falls back to ``expression._name_`` as a plain operator fragment when the
    callable is not registered in
    :class:`~krrood.entity_query_language.verbalization.vocabulary.english.Operators`.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """
        Return ``True`` for :class:`~krrood.entity_query_language.operators.comparator.Comparator` expressions.

        :param expression: Expression to check.
        :param context: Verbalization context (unused).
        :return: ``True`` when *expression* is a :class:`Comparator`.
        :rtype: bool
        """
        return isinstance(expression, Comparator)

    @classmethod
    def transform(
        cls, expression: Comparator, context: VerbalizationContext, verbalizer: EQLVerbalizer
    ) -> VerbFragment:
        """
        Build *"<left> <operator> <right>"* via :func:`~krrood.entity_query_language.verbalization.operator_phrase.comparator_phrase`.

        :param expression: Comparator expression to verbalize.
        :param context: Shared verbalization state.
        :param verbalizer: Parent verbalizer for recursive calls.
        :return: Fragment tree for the comparator phrase.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return comparator_phrase(expression, context, verbalizer)
