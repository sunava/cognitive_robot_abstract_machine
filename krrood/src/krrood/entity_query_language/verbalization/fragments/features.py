"""
Grammatical **features** carried by fragments — small typed values the morphology pass
reads to inflect a leaf.

Kept dependency-free (no fragment / vocabulary imports) so it can sit *below* both the
fragment IR (:mod:`~krrood.entity_query_language.verbalization.fragments.base`) and the
lexicon (:mod:`~krrood.entity_query_language.verbalization.vocabulary.words`) without a
cycle.
"""

from __future__ import annotations

from enum import Enum


class Number(Enum):
    """
    Grammatical **number** — the morphological feature a planner decides, an assembler
    *tags* onto a fragment, and the morphology pass *applies* (pluralising the leaf's text).
    """

    SINGULAR = "singular"
    PLURAL = "plural"

    @classmethod
    def of(cls, is_plural: bool) -> "Number":
        """``PLURAL`` when *is_plural* else ``SINGULAR`` (bridges boolean plan features)."""
        return cls.PLURAL if is_plural else cls.SINGULAR


class Definiteness(Enum):
    """
    Grammatical **definiteness** of a noun phrase — the determiner-system feature a rule
    *tags* onto a :class:`~krrood.entity_query_language.verbalization.fragments.base.NounPhrase`
    and the determiner phase *realises* (choosing *"a/an"* / *"the"* / no determiner, in
    concord with :class:`Number`).

    :cvar BARE: No determiner (a numbered label *"Robot 2"*, a bare predicate noun).
    :cvar INDEFINITE: First, non-specific mention — *"a/an Robot"* (singular) or a **bare**
        plural *"Robots"* (the indefinite article is inherently singular).
    :cvar DEFINITE: Identifiable / subsequent mention — *"the Robot"* / *"the Robots"*.
    :cvar UNIQUE: A uniqueness-quantified first mention — *"the unique Robot"* (the ``eql.the``
        selection).  A *repeat* mention downgrades to ``DEFINITE`` (*"the Robot"*), so it is a
        referring first-mention form like ``INDEFINITE``, not an invariant determiner.
    """

    BARE = "bare"
    INDEFINITE = "indefinite"
    DEFINITE = "definite"
    UNIQUE = "unique"
