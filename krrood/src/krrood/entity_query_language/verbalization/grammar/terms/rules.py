from __future__ import annotations

import dataclasses
import enum
import itertools

from typing_extensions import Any, List, Optional

from krrood.symbol_graph.symbol_graph import Symbol
from krrood.entity_query_language.core.mapped_variable import FlatVariable
from krrood.entity_query_language.core.variable import (
    ExternallySetVariable,
    Literal,
    Variable,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    Fragment,
    NounPhrase,
    oxford_comma,
    PhraseFragment,
    RoleFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    FallbackNouns,
    SetMembership,
    Specificity,
)

#: The most domain values a value-type variable lists before falling back to its type name — a
#: bounded peek keeps the rendering cheap and never enumerates a large (or entity) domain.
_MAX_DOMAIN_CHOICES = 6

#: The value types whose explicit domain is listed (*"one of 1, 2, or 3"*); an entity / ``Symbol``
#: type's domain is the inferred population, which must never be listed.
_PRIMITIVE_VALUE_TYPES = (int, float, str, bool)


class VariableRule(PhraseRule):
    """*"a/an Robot"* (first mention), *"the Robot"* (subsequent), or *"Robot N"* (numbered).

    >>> verbalize_expression(variable(Robot, []))
    'a Robot'
    """

    construct = Variable
    name = "variable"

    def build(self, node: Variable, context: RuleContext) -> Fragment:
        if context.as_value:
            choice = self._domain_choice(node, context)
            if choice is not None:
                return choice
        if context.number is Number.PLURAL:
            return self._plural(node, context)
        noun_form = context.refer.noun_for_parts(node)
        return NounPhrase(
            head=RoleFragment.for_variable(noun_form.label, node),
            definiteness=noun_form.definiteness,
            referent_id=node._id_,
        )

    @staticmethod
    def _domain_choice(node: Variable, context: RuleContext) -> Optional[Fragment]:
        """In value position, a domain-constrained value-type variable says its candidate set —
        *"one of OPTION_A, OPTION_B, or OPTION_C"* (an enum) / *"one of 1, or 2"* (a primitive), or
        just the value for a singleton domain. ``None`` (use the type-name noun) unless the type is
        an ``enum`` / primitive value type and its domain is a small, bounded set — an entity type's
        domain is the inferred population and is never listed.

        :param node: The variable in value position.
        :param context: The per-node context (for value lexicalisation).
        :return: The candidate-set fragment, or ``None`` to fall back to the noun form.
        """
        type_ = getattr(node, "_type_", None)
        is_enum = isinstance(type_, type) and issubclass(type_, enum.Enum)
        if not (is_enum or type_ in _PRIMITIVE_VALUE_TYPES):
            return None
        values = list(
            itertools.islice(
                node._re_enterable_domain_generator_, _MAX_DOMAIN_CHOICES + 1
            )
        )
        if not values or len(values) > _MAX_DOMAIN_CHOICES:
            return None
        fragments: List[Fragment] = [
            RoleFragment(
                text=context.services.type_name_of_value(value),
                role=SemanticRole.LITERAL,
            )
            for value in values
        ]
        if len(fragments) == 1:
            return fragments[0]
        return PhraseFragment(
            parts=[
                SetMembership.ONE_OF.as_fragment(),
                oxford_comma(fragments, Conjunctions.OR.as_fragment()),
            ]
        )

    @staticmethod
    def _plural(node: Variable, context: RuleContext) -> Fragment:
        """Bare plural variable noun phrase (*"Robots"*); the determiner phase drops the article and
        the morphology pass inflects the head.

        A numbered label (*"Robot 2"*) is surface-final — kept singular and bare; a plain type
        name is a plural indefinite noun phrase (the concord table renders it bare-then-pluralised).
        """
        numbered = context.refer.numbered_label(node)
        return NounPhrase(
            head=RoleFragment.for_variable(numbered.text, node),
            number=Number.SINGULAR if numbered.is_numbered else Number.PLURAL,
            definiteness=(
                Definiteness.BARE if numbered.is_numbered else Definiteness.INDEFINITE
            ),
            referent_id=node._id_,
        )


class LiteralRule(PhraseRule):
    """A literal value (e.g. ``42``, ``"hello"``, ``True``), or *"a specific <Type>"* for a concrete
    object literal — we mean its identity, and its ``repr`` can be arbitrarily large.
    """

    construct = Literal
    name = "literal"

    def build(self, node: Literal, context: RuleContext) -> Fragment:
        value = node._value_
        if self._is_concrete_object(value):
            # Identity, not the (possibly huge) repr: "a specific Body".
            return NounPhrase(
                head=RoleFragment.for_variable(type(value).__name__, node),
                definiteness=Definiteness.INDEFINITE,
                pre_head=Specificity.SPECIFIC.as_fragment(),
            )
        return RoleFragment(
            text=context.services.type_name_of_value(value),
            role=SemanticRole.LITERAL,
        )

    @staticmethod
    def _is_concrete_object(value: Any) -> bool:
        """:return: ``True`` when *value* is a concrete domain-object instance (a dataclass or
        :class:`Symbol` instance) — rendered by identity (*"a specific Body"*) rather than its repr.
        A class object, primitive, ``enum`` member, ``datetime``, or ``None`` is not."""
        if isinstance(value, type):
            return False
        return dataclasses.is_dataclass(value) or isinstance(value, Symbol)


class ExternalVariableRule(PhraseRule):
    """*"a/an TypeName"* for an opaque externally-set variable (no coreference)."""

    construct = ExternallySetVariable
    name = "external-variable"

    def build(self, node: ExternallySetVariable, context: RuleContext) -> Fragment:
        type_name = (
            node._type_.__name__
            if getattr(node, "_type_", None)
            else FallbackNouns.VARIABLE.text
        )
        return NounPhrase(head=RoleFragment(text=type_name, role=SemanticRole.VARIABLE))


class FlatVariableRule(PhraseRule):
    """A transparent SetOf wrapper → unwrap to its child (forwarding the requested number)."""

    construct = FlatVariable
    name = "flat-variable"

    def build(self, node: FlatVariable, context: RuleContext) -> Fragment:
        return context.child(node._child_, number=context.number)
