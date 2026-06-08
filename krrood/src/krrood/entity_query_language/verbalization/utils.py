"""
Low-level string utilities for the verbalization subsystem.

Pure helpers — CamelCase splitting, English ordinals, and safe pluralisation.
Fragment flattening lives in
:func:`~krrood.entity_query_language.verbalization.fragments.base.flatten_fragment_to_plain_text`.
"""

from __future__ import annotations

import re

from krrood.entity_query_language.verbalization._inflect import _engine as _inflect_engine


def _camel_to_words(name: str) -> str:
    """
    Convert a CamelCase class name to space-separated lowercase words.

    :param name: CamelCase identifier string.
    :type name: str
    :return: Space-separated lowercase words.
    :rtype: str

    Examples::

        _camel_to_words("HasRole")     # → "has role"
        _camel_to_words("IsReachable") # → "is reachable"
    """
    return re.sub(r"([A-Z])", r" \1", name).strip().lower()


def _ordinal(n: int) -> str:
    """
    Convert a zero-based integer index to an ordinal word (e.g. ``0`` → ``"first"``).

    Delegates to the ``inflect`` library for correct English ordinals.

    :param n: Zero-based integer index.
    :type n: int
    :return: English ordinal word (e.g. ``"first"``, ``"second"``, ``"third"``).
    :rtype: str
    """
    return _inflect_engine.ordinal(_inflect_engine.number_to_words(n + 1))


def _ensure_plural(word: str) -> str:
    """
    Return *word* in plural form without double-pluralising already-plural words.

    Uses ``inflect.singular_noun`` to detect whether *word* is already plural;
    if so returns it unchanged.

    :param word: English noun in singular or plural form.
    :type word: str
    :return: Plural form of *word*.
    :rtype: str
    """
    return word if _inflect_engine.singular_noun(word) else _inflect_engine.plural(word)


