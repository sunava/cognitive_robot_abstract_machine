"""
Unit tests for grammatical-number agreement — now plain lexical inflection
(``Copulas.for_number`` / ``ExistentialPhrase.for_number``) driven by the
:class:`Number` feature, replacing the retired number ``Choice`` systems.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Copulas,
    ExistentialPhrase,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


def _text(fragment) -> str:
    return flatten_fragment_to_plain_text(fragment)


def test_number_of_bridges_boolean_plan_features():
    assert Number.of(True) is Number.PLURAL
    assert Number.of(False) is Number.SINGULAR


def test_copula_inflects_for_number():
    assert _text(Copulas.for_number(Number.SINGULAR).as_fragment()) == "is"
    assert _text(Copulas.for_number(Number.PLURAL).as_fragment()) == "are"


def test_existential_inflects_for_number():
    assert (
        _text(ExistentialPhrase.for_number(Number.SINGULAR).build_phrase("Robot"))
        == "there's a Robot"
    )
    assert (
        _text(ExistentialPhrase.for_number(Number.PLURAL).build_phrase("Robot"))
        == "there are Robots"
    )
