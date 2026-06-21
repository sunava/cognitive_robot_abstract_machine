from __future__ import annotations

import inflect
from lemminflect import getInflection, getLemma

#: The one shared ``inflect`` engine for the whole verbalization subsystem.
_engine = inflect.engine()


def plural(word: str) -> str:
    """
    :param word: An English noun (assumed singular).
    :return: The plural form of *word*, unconditionally (e.g. ``"Robot"`` → ``"Robots"``).
    """
    return _engine.plural(word)


def ensure_plural(word: str) -> str:
    """
    :param word: An English noun in either number.
    :return: The plural form of *word*, without double-pluralising an already-plural word.
    """
    return word if _engine.singular_noun(word) else _engine.plural(word)


def is_plural(word: str) -> bool:
    """
    :param word: An English noun.
    :return: ``True`` when *word* is already in plural form.
    """
    return bool(_engine.singular_noun(word))


def is_past_participle(word: str) -> bool:
    """
    :param word: A single English word.
    :return: ``True`` when *word* is a past-participle verb form — regular (*"assigned"*,
        *"located"*) or irregular (*"sent"*, *"given"*, *"written"*) — and ``False`` for a base form
        (*"assign"*), a noun (*"battery"*), or a past tense that differs from the participle
        (*"sang"* vs *"sung"*).

    Deterministic dictionary + rule lookup (``lemminflect``): the word's verb lemma is taken, and the
    word is checked against that lemma's generated ``VBN`` forms. No statistical model, no data
    download — so it is reproducible and offline, unlike a context-sensitive POS tagger.
    """
    lowered = word.lower()
    return any(
        lowered in {form.lower() for form in getInflection(lemma, tag="VBN")}
        for lemma in getLemma(lowered, upos="VERB")
    )


def indefinite_article(following_word: str) -> str:
    """
    :param following_word: The word the article precedes.
    :return: The indefinite article (``"a"`` / ``"an"``) for *following_word*, chosen
        phonologically (e.g. ``"hour"`` → ``"an"``, ``"robot"`` → ``"a"``).
    """
    return _engine.a(following_word).split()[0]


def ordinal(index: int) -> str:
    """
    :param index: Zero-based integer index.
    :return: The English ordinal word for a zero-based *index* (``0`` → ``"first"``).
    """
    return _engine.ordinal(_engine.number_to_words(index + 1))


def cardinal(n: int) -> str:
    """
    :param n: A positive integer.
    :return: The English cardinal word for *n* (``2`` → ``"two"``).
    """
    return _engine.number_to_words(n)
