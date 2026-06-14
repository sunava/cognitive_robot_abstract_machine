from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List, Optional

from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Call,
    FlatVariable,
    Index,
    MappedVariable,
)
from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)


@dataclass(frozen=True)
class PathStep:
    """
    One hop of a navigation chain — a display name and the source reference it links to.

    Replaces a bare ``(name, source_reference)`` tuple so the two halves are named (``.name`` /
    ``.source_reference``) rather than positional.
    """

    name: str
    """The display text for this hop (e.g. ``"amount"``, ``"handle[0]"``, ``"()"``)."""

    source_reference: Optional[SourceReference] = None
    """The attribute's source reference, or ``None`` for composite / index / call hops."""


def build_path_parts(chain: List[MappedVariable]) -> List[PathStep]:
    """
    Convert a walked chain into :class:`PathStep` hops.

    Merging rules:

    * Consecutive ``Attribute → Index`` pairs are merged into ``"attribute[key]"`` with no source
      reference (composite indexed access has no clean single-symbol anchor).
    * Standalone ``Index`` nodes appear as ``"[key]"`` with no source reference.
    * ``Call`` nodes appear as ``"()"`` with no source reference.
    * ``FlatVariable`` nodes are skipped.

    :param chain: Outermost-first chain list.
    :return: Ordered list of :class:`PathStep`, outermost attribute first.
    """
    parts: List[PathStep] = []
    i = 0
    while i < len(chain):
        node = chain[i]
        if isinstance(node, Attribute):
            name = node._attribute_name_
            owner = node._owner_class_
            reference: Optional[SourceReference] = SourceReference.for_attribute(
                owner, name
            )
            while i + 1 < len(chain) and isinstance(chain[i + 1], Index):
                i += 1
                name += f"[{repr(chain[i]._key_)}]"
                reference = (
                    None  # composite indexed access has no clean single-line anchor
                )
            parts.append(PathStep(name, reference))
        elif isinstance(node, Index):
            parts.append(PathStep(f"[{repr(node._key_)}]", None))
        elif isinstance(node, Call):
            parts.append(PathStep("()", None))
        elif isinstance(node, FlatVariable):
            pass
        i += 1
    return parts
