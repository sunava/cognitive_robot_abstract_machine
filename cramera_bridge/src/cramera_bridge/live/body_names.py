"""
Finding a world body by the name a person or a browser wrote.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, List, Optional

from cramera_bridge.exceptions import AmbiguousBody, UnknownBody

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body

# %% lookup


@dataclass(frozen=True)
class BodyNames:
    """
    Resolves names to bodies: a full prefixed name first, else a name without its prefix
    that exactly one body carries.
    """

    world: World
    """
    The world whose bodies are searched.
    """

    def find(self, name: str) -> Optional[Body]:
        """
        :param name: A full or short body name.
        :return: The body, or None when no body carries the name.
        :raises AmbiguousBody: When several bodies carry the short name.
        """
        bodies = list(self.world.bodies)
        for body in bodies:
            if str(body.name) == name:
                return body
        candidates: List[Body] = [body for body in bodies if body.name.name == name]
        if len(candidates) > 1:
            raise AmbiguousBody(name, sorted(str(body.name) for body in candidates))
        return candidates[0] if candidates else None

    def require(self, name: str) -> Body:
        """
        :param name: A full or short body name.
        :return: The body.
        :raises UnknownBody: When no body carries the name.
        :raises AmbiguousBody: When several bodies carry the short name.
        """
        body = self.find(name)
        if body is None:
            raise UnknownBody(name)
        return body
