"""
Restriction **assembler** — render a
:class:`~krrood.entity_query_language.verbalization.grammar.planning.query.RestrictionPlan`
(a subject's WHERE partition) into its two surface pieces: the appositive *"whose <grouped>"*
modifier and the residual *"such that …"* / *"where …"* condition.

Realisation-only, but deliberately **not** an :class:`Assembler` subclass: a restriction yields
a *pair* (modifier, condition), not a single fragment — the caller drops each into its own slot
(inline after the selection, as a noun modifier, or inside an aggregation scope).  Extracted from
``QueryAssembler`` so the three callers share one renderer.

Reference: Reiter & Dale (2000) — content structuring (the WHERE partition is the *plan*).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional, Tuple

from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_and,
    PhraseFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.grammar.phrase_rule import Ctx
from krrood.entity_query_language.verbalization.grammar.planning.query import (
    RestrictionPlan,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    build_between,
    RangeFold,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Keywords,
)


@dataclass
class RestrictionAssembler:
    """Render a :class:`RestrictionPlan` into a ``(whose-modifier, residual-condition)`` pair."""

    ctx: Ctx
    """The per-node context (recursion entry + microplanning services)."""

    def render(
        self, restriction: RestrictionPlan, subject
    ) -> Tuple[Optional[VerbFragment], Optional[VerbFragment]]:
        """The *"whose <grouped>"* modifier (or ``None``) and the residual condition (or ``None``)."""
        grouped_frags = [
            rule.render(item, subject, self.ctx) for rule, item in restriction.grouped
        ]
        whose = None
        if grouped_frags:
            whose = PhraseFragment(
                parts=[
                    Keywords.WHOSE.as_fragment(),
                    oxford_and(grouped_frags, Conjunctions.AND.as_fragment()),
                ]
            )
        residual = (
            self._residual(restriction.residual) if restriction.has_residual else None
        )
        return whose, residual

    def _residual(self, items) -> VerbFragment:
        """The residual conjuncts (raw expressions / folded ranges) joined into one condition."""
        parts: list = []
        for item in items:
            if isinstance(item, RangeFold):
                parts.append(
                    build_between(
                        self.ctx.child(item.chain_expression),
                        self.ctx.child(item.lower_expression),
                        self.ctx.child(item.upper_expression),
                        compact=self.ctx.context.compact_predicates,
                    )
                )
            else:
                parts.append(self.ctx.child(item))
        if len(parts) == 1:
            return parts[0]
        return oxford_and(parts, Conjunctions.AND.as_fragment())
