"""
Collects the nodes, edges and details of one graph view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Dict, List, Optional

from cramera_protocol.questions import GraphDetail, GraphEdge, GraphEdgeKind, GraphNode
from cramera_bridge.questions.exceptions import DuplicateGraphNode

TITLE_SEPARATOR = "\n"
"""
Separates the lines of a node's tooltip.
"""

# %% builder


@dataclass
class GraphBuilder:
    """
    Collects one view's nodes, each with its details, and the edges between them.
    """

    nodes: List[GraphNode] = field(default_factory=list)
    """
    The nodes in the order they were added.
    """

    edges: List[GraphEdge] = field(default_factory=list)
    """
    The edges in the order they were added.
    """

    details: Dict[str, GraphDetail] = field(default_factory=dict)
    """
    The details of each node by its identity.
    """

    def add_node(
        self,
        identifier: str,
        label: str,
        group: str,
        lines: List[str],
        status: Optional[str] = None,
        kind: Optional[str] = None,
        parent: Optional[str] = None,
    ) -> None:
        """
        :param identifier: The node's identity within the view.
        :param label: The node's label.
        :param group: The colour group it is drawn in.
        :param lines: The lines describing it.
        :param status: Its native status, for plan and statechart nodes.
        :param kind: The class name it was made from, for nodes of a tree.
        :param parent: Its parent, for nodes of a tree.
        :raises DuplicateGraphNode: When the view already holds a node of that identity.
        """
        if identifier in self.details:
            raise DuplicateGraphNode(identifier)
        self.nodes.append(
            GraphNode(
                identifier=identifier,
                label=label,
                group=group,
                title=TITLE_SEPARATOR.join([label, *lines]),
                status=status,
                kind=kind,
                parent=parent,
            )
        )
        self.details[identifier] = GraphDetail(label=label, group=group, lines=lines)

    def add_edge(
        self, source: str, destination: str, kind: GraphEdgeKind, label: str
    ) -> None:
        """
        :param source: The node the edge starts at.
        :param destination: The node it leads to.
        :param kind: How it is drawn.
        :param label: The relation it stands for.
        """
        self.edges.append(
            GraphEdge(source=source, destination=destination, kind=kind, label=label)
        )
