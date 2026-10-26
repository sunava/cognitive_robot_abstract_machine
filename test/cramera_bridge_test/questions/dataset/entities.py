"""
Entities standing in for the objects a query ranges over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from typing_extensions import List, Optional

from cramera_bridge.questions.entity import NamedEntity
from cramera_bridge.questions.highlights import ValueFormat

# %% entities


@dataclass(unsafe_hash=True)
class ShelfItem(NamedEntity):
    """
    A named item with a size, a note kept out of answers and a class-level default.
    """

    size: float = 0.0
    """
    The item's size.
    """

    shelf: Optional[str] = None
    """
    The name of the shelf the item stands on.
    """

    note: str = field(default="internal", repr=False)
    """
    Bookkeeping that answers leave out.
    """

    stocked: bool = field(default=True, init=False)
    """
    A field left at its default.
    """


@dataclass(unsafe_hash=True)
class Delivery(NamedEntity):
    """
    Something that happened at a moment.
    """

    timestamp: datetime
    """
    When it happened.
    """


@dataclass(frozen=True)
class Coordinates:
    """
    A value whose representation is replaced by a format.
    """

    x: float
    """
    The x coordinate.
    """


@dataclass(frozen=True)
class CoordinatesFormat(ValueFormat):
    """
    Writes coordinates as a short text.
    """

    prefix: str = "at "
    """
    The text written before the coordinate.
    """

    def accepts(self, value: object) -> bool:
        return isinstance(value, Coordinates)

    def text(self, value: Coordinates) -> str:
        return f"{self.prefix}{value.x}"


@dataclass
class GrowingShelf:
    """
    A source of items that grows after a desk was built.
    """

    items: List[ShelfItem] = field(default_factory=list)
    """
    The items so far.
    """

    def __iter__(self):
        return iter(list(self.items))
