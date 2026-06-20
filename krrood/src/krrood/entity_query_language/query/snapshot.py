"""
Snapshotting of compiled query expressions for the Entity Query Language.

When a query is embedded as a subquery inside another expression, the parent must capture an
immutable copy of the query's compiled expression. Otherwise a later edit to the original query
would rewire the very node already embedded in the parent. :class:`SpineSnapshot` produces that copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Dict

from krrood.entity_query_language.core.base_expressions import (
    SymbolicExpression,
    Selectable,
)
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.aggregators import Aggregator


@dataclass
class SpineSnapshot:
    """
    Produces an independent copy of a built query's compiled expression spine.

    Structural nodes (quantifiers, ordering, filters, the cartesian-product query node and the
    logical/comparison operators) are cloned into a fresh object graph, while variable leaves are
    shared. Identifiers are preserved: the clone evaluates identically to the original at snapshot
    time (so co-referenced bindings still match), but because it is a separate object graph, later
    edits that rewire the original query cannot reach the already-embedded copy.
    """

    _clones_by_source_id: Dict[int, SymbolicExpression] = field(default_factory=dict)
    """Maps the ``id()`` of an already-cloned source node to its clone, so a shared subtree is cloned
    once and every reference to it within the snapshot points at the same clone."""

    def snapshot(self, root: SymbolicExpression) -> SymbolicExpression:
        """
        :param root: The compiled expression to copy.
        :return: An independent clone of ``root`` whose variable leaves are shared with the original.
        """
        if self._is_shared_leaf_(root):
            return root
        if id(root) in self._clones_by_source_id:
            return self._clones_by_source_id[id(root)]

        clone = self._clone_structural_node_(root)
        self._clones_by_source_id[id(root)] = clone
        for child in root._children_:
            cloned_child = self.snapshot(child)
            clone._replace_child_field_(child, cloned_child)
            cloned_child._parent_ = clone
        self._relink_tracked_variable_(root, clone)
        return clone

    @staticmethod
    def _is_shared_leaf_(node: SymbolicExpression) -> bool:
        """
        :param node: The node to classify.
        :return: Whether ``node`` is a leaf that must be shared rather than cloned.
        """
        return isinstance(node, (Variable, MappedVariable, Aggregator))

    @staticmethod
    def _clone_structural_node_(source: SymbolicExpression) -> SymbolicExpression:
        """
        Shallow-copy ``source`` and reset its graph bookkeeping so it forms a fresh, detached node
        ready to be relinked to its cloned children. The identifier is preserved so the clone
        evaluates identically to the original.

        :param source: The structural node to clone.
        :return: The detached clone sharing the source's identifier.
        """
        # Build the clone via ``__new__`` and a shallow copy of ``__dict__`` rather than
        # ``copy.copy``: these nodes define a catch-all ``__getattr__`` that turns the copy
        # protocol's dunder probing (``__setstate__`` and friends) into infinite recursion.
        clone = source.__class__.__new__(source.__class__)
        clone.__dict__.update(source.__dict__)
        clone._parents_ = []
        clone._parent__ = None
        clone._children_ = []
        clone._expression_ = clone
        clone._expression_id_cache_ = {}
        return clone

    def _relink_tracked_variable_(
        self, source: SymbolicExpression, clone: SymbolicExpression
    ) -> None:
        """
        Repoint a clone's tracked variable (``_var_``) at the corresponding clone.

        Selectable wrappers (for example a result quantifier) delegate variable behaviour to a
        tracked node; after cloning, that delegation must follow the clones rather than the original.

        :param source: The original node.
        :param clone: Its clone, whose ``_var_`` still references the original's tracked node.
        """
        if not isinstance(clone, Selectable):
            return
        tracked = clone._var_
        if tracked is source:
            clone._var_ = clone
        elif tracked is not None and id(tracked) in self._clones_by_source_id:
            clone._var_ = self._clones_by_source_id[id(tracked)]
