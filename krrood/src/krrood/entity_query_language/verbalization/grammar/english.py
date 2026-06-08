"""
The English grammar — one :class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.PhraseRule`
per EQL construct, as data.

Each rule is a Montague rule-to-rule clause: *for this construct, build this phrase*
(Montague 1970; Bach 1976, the rule-to-rule hypothesis).  A rule only **combines** —
recursion is delegated to ``ctx.child`` (the fold), and cross-cutting decisions to the
microplanning services (``ctx.refer`` / ``ctx.scope`` / ``ctx.config``), morphology, the
coordination module, and the lexicon — so each rule is responsible for a single
construct's surface composition.

Families are ported here one at a time; until a construct's rule is present the engine
falls back to the legacy dispatcher (strangler migration).
"""

from __future__ import annotations

from typing_extensions import List

from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.fragments.factory import phrase
from krrood.entity_query_language.verbalization.grammar.phrase_rule import PhraseRule
from krrood.entity_query_language.verbalization.operator_phrase import (
    comparator_operator,
)

# ── comparator ───────────────────────────────────────────────────────────────

COMPARATOR_RULES: List[PhraseRule] = [
    PhraseRule(
        construct=Comparator,
        name="comparator",
        # "<left> <operator> <right>", e.g. "is greater than 50".
        build=lambda node, ctx: phrase(
            ctx.child(node.left),
            comparator_operator(node, ctx.context),
            ctx.child(node.right),
        ),
    ),
]

# ── the full grammar ───────────────────────────────────────────────────────────

RULES: List[PhraseRule] = [
    *COMPARATOR_RULES,
]
