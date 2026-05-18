from __future__ import annotations

from enum import Enum
from typing import Optional


class SemanticRole(Enum):
    KEYWORD = "keyword"        # If, Then, Find, Where, Such that
    VARIABLE = "variable"      # Robot, Employee 1
    AGGREGATION = "aggregation"  # sum of, number of, average of
    OPERATOR = "operator"      # is greater than, equals
    LOGICAL = "logical"        # and, or, not, for all, there exists
    LITERAL = "literal"        # 42, "hello", True
    ATTRIBUTE = "attribute"    # battery, tasks, name
    PLAIN = "plain"            # neutral connecting text


# Hex colours taken directly from QueryGraph.ColorLegend
ROLE_COLORS: dict[SemanticRole, Optional[str]] = {
    SemanticRole.KEYWORD:     "#eded18",        # ConclusionSelector yellow
    SemanticRole.VARIABLE:    "cornflowerblue",
    SemanticRole.AGGREGATION: "#F54927",         # Aggregator red-orange
    SemanticRole.OPERATOR:    "#ff7f0e",         # Comparator orange
    SemanticRole.LOGICAL:     "#2ca02c",         # LogicalOperator green
    SemanticRole.LITERAL:     "#949292",         # Literal gray
    SemanticRole.ATTRIBUTE:   "#8FC7B8",         # MappedVariable teal
    SemanticRole.PLAIN:       None,
}
