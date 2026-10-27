"""
Where the test files are.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DATASET = Path(__file__).parent
"""
The directory of the test files.
"""

# %% files


class QueryFile(StrEnum):
    """
    The query sources the tests run.
    """

    HEAVY_ITEMS = "heavy_items.eql"
    """
    Sets a threshold up, then asks for the items above it.
    """

    BOUND_RESULT = "bound_result.eql"
    """
    Ends in a statement binding ``result``.
    """

    SELECTION = "selection.eql"
    """
    Selects two attributes of each item.
    """

    DELIVERY_TIMES = "delivery_times.eql"
    """
    Selects the moment of each delivery.
    """

    BROKEN = "broken.eql"
    """
    Does not parse.
    """

    UNKNOWN_NAME = "unknown_name.eql"
    """
    Names nothing in reach.
    """


@dataclass(frozen=True)
class QuerySource:
    """
    Reads a query source of the dataset.
    """

    query: QueryFile
    """
    The query.
    """

    def text(self) -> str:
        """
        :return: The query's source.
        """
        return (DATASET / "queries" / self.query).read_text(encoding="utf-8")


class SceneName(StrEnum):
    """
    The bundles of the dataset.
    """

    ANNOTATED = "annotated_run"
    """
    A recording with annotated parts, statecharts, plans and detections.
    """

    LEGACY = "legacy_run"
    """
    One of the earliest recordings, naming its parts without annotations.
    """


SCENES = DATASET / "scenes"
"""
The scenes directory of the dataset.
"""

CLASS_TREE = DATASET / "class_tree" / "mimicked_package"
"""
A package directory whose classes are indexed.
"""

EVENT_TYPES_WITH_QUERY_CODE = DATASET / "events" / "event_types_with_query_code.json"
"""
Detected event types as a crafted bundle could list them: one real type and one that
would close the query's quotes and run code of its own.
"""

TWO_ROBOTS_SCENE = DATASET / "documents" / "two_robots_scene.json"
"""
The scene description of a recording of two robots, the second one selected.
"""

PREFIXED_LINKS_URDF = DATASET / "urdf" / "prefixed_links.urdf"
"""
The annotated recording's robot model with every link named below the robot's prefix, as
CRAM writes models while the part annotations name bare links.
"""
