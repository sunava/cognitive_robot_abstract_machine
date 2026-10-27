"""
Which keys of the viewer an answer lights up, and how answer values read as text.

The answer keeps the native names of what it found; turning them into the keys the
viewer shows them under happens here, at the presentation boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any, Dict, FrozenSet, List

from cramera_bridge.questions.entity import NamedEntity

# %% highlights


class HighlightKeys(ABC):
    """
    Maps answer values to the keys of the objects and links the viewer shows.
    """

    @abstractmethod
    def keys_of(self, value: Any) -> List[str]:
        """
        :param value: An answer value: an entity, a text or anything else.
        :return: The viewer keys it lights up; empty when it names nothing shown.
        """


@dataclass(frozen=True)
class EntityNameKeys(HighlightKeys):
    """
    Lights up a named entity under its name, or under the key its name maps to, and a
    text that equals a known key.
    """

    keys_by_name: Dict[str, str] = field(default_factory=dict)
    """
    The viewer key of each entity name that is shown under another key.
    """

    known_keys: FrozenSet[str] = frozenset()
    """
    Every key the viewer shows, so a text answer naming one lights it up.
    """

    def keys_of(self, value: Any) -> List[str]:
        if isinstance(value, NamedEntity):
            return [self.keys_by_name.get(value.name, value.name)]
        if isinstance(value, str) and value in self.keys_by_name:
            return [self.keys_by_name[value]]
        if isinstance(value, str) and value in self.known_keys:
            return [value]
        return []


# %% value texts


class ValueFormat(ABC):
    """
    Writes values of some type as the text an answer shows, for types whose own
    representation is unreadable.
    """

    @abstractmethod
    def accepts(self, value: Any) -> bool:
        """
        :param value: An answer value.
        :return: Whether this format writes it.
        """

    @abstractmethod
    def text(self, value: Any) -> str:
        """
        :param value: An answer value this format accepts.
        :return: The value as an answer shows it.
        """
