"""
The grammar registry — the list of every :class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.PhraseRule`.

``ALL_PHRASE_RULES`` is plain data, so the grammar itself is EQL-queryable, e.g.::

    rule = variable(PhraseRule, domain=ALL_PHRASE_RULES)
    entity(rule).where(rule.construct == Comparator)

It is populated from :mod:`~krrood.entity_query_language.verbalization.grammar.english`
as each construct family is ported (Phase A); until a family lands its rules are
simply absent and the fold falls back to a plain word fragment.
"""

from __future__ import annotations

from typing_extensions import List

from krrood.entity_query_language.verbalization.grammar.english import RULES
from krrood.entity_query_language.verbalization.grammar.phrase_rule import PhraseRule

#: Every phrase rule, in no particular order (``select`` decides specificity).
ALL_PHRASE_RULES: List[PhraseRule] = list(RULES)
