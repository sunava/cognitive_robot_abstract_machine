from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole


@dataclass
class VerbFragment:
    """Abstract base for all verbalized output fragments."""


@dataclass
class WordFragment(VerbFragment):
    """Plain neutral text: articles, connectives, punctuation."""
    text: str


@dataclass
class RoleFragment(VerbFragment):
    """Text carrying a semantic role — drives coloring."""
    text: str
    role: SemanticRole


@dataclass
class PhraseFragment(VerbFragment):
    """An inline sequence of fragments joined by a separator."""
    parts: list[VerbFragment]
    separator: str = " "

    @classmethod
    def joined(cls, parts: list[VerbFragment], separator: str = " ") -> "PhraseFragment":
        return cls(parts=parts, separator=separator)

    @classmethod
    def spaced(cls, *parts: VerbFragment) -> "PhraseFragment":
        return cls(parts=list(parts), separator=" ")


@dataclass
class BlockFragment(VerbFragment):
    """
    A named structural block with sub-items.

    ParagraphRenderer flattens this into prose.
    HierarchicalRenderer turns it into an indented bullet list.
    """
    header: Optional[VerbFragment]
    items: list[VerbFragment] = field(default_factory=list)
