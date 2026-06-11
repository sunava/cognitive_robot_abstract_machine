"""
Restriction **assembler** — render a
:class:`~krrood.entity_query_language.verbalization.grammar.planning.query.RestrictionPlan`
(a subject's WHERE partition) into its surface pieces: superlative selection modifiers
(*"with the maximum <leaf>"*), the appositive *"whose <grouped>"* modifier, and the residual
*"such that …"* / *"where …"* condition.

Realisation-only, but deliberately **not** an :class:`Assembler` subclass: a restriction yields
*several* pieces (:class:`RestrictionFragments`), not a single fragment — the caller drops each
into its own slot (inline after the selection, as a noun modifier, or inside an aggregation
scope).  Extracted from ``QueryAssembler`` so the three callers share one renderer.

Reference: Reiter & Dale (2000) — content structuring (the WHERE partition is the *plan*).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional

from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_and,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.grammar.aggregation_kinds import (
    AGGREGATION_KIND,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    SuperlativeFold,
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
    Articles,
    Conjunctions,
    Keywords,
    Prepositions,
)


@dataclass(frozen=True)
class RestrictionFragments:
    """The rendered pieces of a subject restriction, each placed by the caller."""

    superlatives: List[VerbFragment] = field(default_factory=list)
    """Selection PP modifiers, e.g. *"with the maximum amount"* (attach right after the noun)."""

    whose: Optional[VerbFragment] = None
    """The appositive *"whose <grouped>"* modifier, or ``None``."""

    residual: Optional[VerbFragment] = None
    """The residual condition for a *"such that …"* / *"where …"* clause, or ``None``."""


@dataclass
class RestrictionAssembler:
    """Render a :class:`RestrictionPlan` into its :class:`RestrictionFragments`."""

    ctx: Ctx
    """The per-node context (recursion entry + microplanning services)."""

    def render(self, restriction: RestrictionPlan, subject) -> RestrictionFragments:
        """Render the superlative modifiers, the *"whose"* modifier, and the residual condition."""
        superlatives = [self._superlative(fold) for fold in restriction.superlatives]
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
        return RestrictionFragments(
            superlatives=superlatives, whose=whose, residual=residual
        )

    def _superlative(self, fold: SuperlativeFold) -> VerbFragment:
        """*"with the maximum <leaf>"* / *"with the minimum <leaf>"* — the folded superlative."""
        leaf = fold.aggregator._leaf_attribute_
        return PhraseFragment(
            parts=[
                Prepositions.WITH.as_fragment(),
                Articles.THE.as_fragment(),
                AGGREGATION_KIND[type(fold.aggregator)].as_fragment(),
                RoleFragment.for_attribute(leaf._owner_class_, leaf._attribute_name_),
            ]
        )

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
                        compact=self.ctx.config.compact_predicates,
                    )
                )
            else:
                parts.append(self.ctx.child(item))
        if len(parts) == 1:
            return parts[0]
        return oxford_and(parts, Conjunctions.AND.as_fragment())
