from __future__ import annotations

import uuid

from typing_extensions import List, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.core.expression_structure import walk_chain
from krrood.entity_query_language.verbalization.navigation_path import (
    build_path_parts,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    oxford_comma,
    PhraseFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.assembler import (
    ConditionAssembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.restriction_assembler import (
    RestrictionAssembler,
    RestrictionFragments,
)
from krrood.entity_query_language.verbalization.grammar.inference.planner import (
    AggregationStatus,
    AntecedentInformation,
    ConsequentBinding,
    InferencePlanner,
    RuleStructure,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    RestrictionPlan,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    fold_range_pairs,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    ExistentialPhrase,
    FallbackNouns,
    GroupKeyPhrases,
    Keywords,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


class InferenceAssembler(Assembler[Entity, RuleStructure]):
    """
    Realise the IF/THEN block from a rule structure.

    Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
    """

    planner = InferencePlanner

    def realize(self, node: Entity, plan: RuleStructure) -> Fragment:
        """
        :param node: The inference-rule query.
        :param plan: The IF/THEN rule structure.
        :return: *"If <antecedents…>, then <consequent…>"* — the two-block IF/THEN form.
        """
        return BlockFragment(
            header=None,
            items=[
                BlockFragment(
                    header=Keywords.IF.as_fragment(), items=self._if_items(plan)
                ),
                BlockFragment(
                    header=Keywords.THEN.as_fragment(), items=self._then_items(plan)
                ),
            ],
        )

    @staticmethod
    def _number(antecedent: AntecedentInformation) -> Number:
        """:return: The grammatical number of an antecedent — plural if and only if aggregated."""
        return Number.of(antecedent.aggregation_status == AggregationStatus.AGGREGATED)

    # ── IF clause ───────────────────────────────────────────────────────────────

    def _if_items(self, structure: RuleStructure) -> List[Fragment]:
        """
        :return: One item per antecedent — *"there's a <Type> whose …, and …"* — plus any unmatched
            conditions; *"true"* when there are none.
        """
        items: List[Fragment] = [
            self._antecedent(antecedent) for antecedent in structure.primary_antecedents
        ]
        items += [
            self.context.child(condition)
            for condition in structure.unmatched_conditions
        ]
        return items or [Keywords.TRUE.as_fragment()]

    def _antecedent(self, antecedent: AntecedentInformation) -> Fragment:
        """:return: *"there's a <Type> whose <conditions>"* — the existential intro with the
        antecedent's conditions woven in by the shared restriction machinery (the same *"whose …,
        and …"* / *"such that …"* form a query selection uses)."""
        intro = self._antecedent_intro(antecedent)
        if not antecedent.conditions or antecedent.variable is None:
            return intro
        restriction = RestrictionAssembler(self.context).render(
            RestrictionPlan(folded=fold_range_pairs(antecedent.conditions)),
            antecedent.variable,
            self._number(antecedent),
        )
        return self._weave(intro, restriction)

    @staticmethod
    def _weave(intro: Fragment, restriction: RestrictionFragments) -> Fragment:
        """:return: The intro with its restriction modifiers attached and any residual appended as a
        *"such that …"* clause."""
        parts: List[Fragment] = [intro, *restriction.modifiers]
        if restriction.residual is not None:
            parts += [Keywords.SUCH_THAT.as_fragment(), restriction.residual]
        return PhraseFragment(parts=parts) if len(parts) > 1 else intro

    def _antecedent_intro(self, antecedent: AntecedentInformation) -> Fragment:
        """:return: *"there's a <Type>"* / *"there are <Types>"* — the antecedent's existential intro."""
        return ExistentialPhrase.for_number(self._number(antecedent)).build_phrase(
            antecedent.type_name, referent_id=self._antecedent_referent_id(antecedent)
        )

    @staticmethod
    def _antecedent_referent_id(
        antecedent: AntecedentInformation,
    ) -> Optional[uuid.UUID]:
        """
        :return: The antecedent's canonical referent id — the selected variable for an entity
            root, else the root's own id (matching the variable the THEN-clause chains reference).
        """
        root = antecedent.root
        if isinstance(root, Entity):
            root.build()
            return getattr(root.selected_variable, "_id_", None)
        return getattr(root, "_id_", None)

    # ── THEN clause ───────────────────────────────────────────────────────────

    def _then_items(self, structure: RuleStructure) -> List[Fragment]:
        """:return: *"there's a <Consequent> whose <field> is <value>, and …"* — the THEN clause, its
        field bindings gathered under one *"whose …, and …"* envelope (the same form a query
        subject restriction uses)."""
        intro: Fragment = ExistentialPhrase.for_number(Number.SINGULAR).build_phrase(
            structure.consequent_type
        )
        bindings = [
            self._binding_predicate(binding)
            for binding in structure.consequent_bindings
        ]
        if not bindings:
            return [intro]
        whose = PhraseFragment(
            parts=[
                Keywords.WHOSE.as_fragment(),
                oxford_comma(bindings, Conjunctions.AND.as_fragment()),
            ]
        )
        return [PhraseFragment(parts=[intro, whose])]

    def _binding_predicate(self, binding: ConsequentBinding) -> Fragment:
        """:return: The bare *"<field> is/are <value>"* predicate for one consequent binding (the
        shared *"whose"* envelope is added once by :meth:`_then_items`)."""
        number = Number.of(binding.is_plural_field)
        return ConditionAssembler(self.context).attribute_predicate(
            binding.field_name, number, self._binding_value(binding)
        )

    def _binding_value(self, binding: ConsequentBinding) -> Fragment:
        """
        :return: The binding's value: *"the <plural chain>"* (aggregated), bare plural, the
            group-key *"common …"* phrase, or the plain rendering.
        """
        if (
            binding.is_plural_field
            and binding.aggregation_status == AggregationStatus.AGGREGATED
        ):
            return PhraseFragment(
                parts=[
                    Articles.THE.as_fragment(),
                    self.context.child(binding.value_expression, number=Number.PLURAL),
                ]
            )
        if binding.is_plural_field:
            return self.context.child(binding.value_expression, number=Number.PLURAL)
        if binding.aggregation_status == AggregationStatus.GROUP_KEY:
            return self._group_key_value(binding.value_expression)
        return self.context.child(binding.value_expression)

    def _group_key_value(self, expression: SymbolicExpression) -> Fragment:
        """:return: *"the common <field> of the <Roots>"* — a binding that refers to a GROUP BY key."""
        chain, current = walk_chain(expression)
        if not chain or not isinstance(current, Variable):
            return self.context.child(expression)
        root_type = (
            current._type_.__name__
            if getattr(current, "_type_", None)
            else FallbackNouns.ENTITY.text
        )
        parts = build_path_parts(chain)
        field = list(reversed(parts))[0].name if parts else root_type
        return GroupKeyPhrases.COMMON_OF.build_phrase(field, root_type)
