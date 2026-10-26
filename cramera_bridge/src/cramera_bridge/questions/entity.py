"""
What every entity of a recorded scene has in common.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from krrood.symbol_graph.symbol_graph import Symbol

# %% entities


@dataclass
class NamedEntity(Symbol, ABC):
    """
    One entity of a recorded scene a query can range over; answers name it, and the
    viewer lights it up, by its name.
    """

    name: str
    """
    The entity's name.
    """
