"""
What a world consists of, read again whenever its model changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Iterable, List, Optional
from uuid import UUID

from cramera_bridge.robots import RobotRoster
from cramera_bridge.world_objects import WorldObjects
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import (
        Body,
        Connection,
    )

# %% model


@dataclass(frozen=True)
class WorldModelView:
    """
    The objects, joints, connections and robots of a world at one model version.
    """

    world: World
    """
    The world.
    """

    objects: List[Body] = field(default_factory=list)
    """
    The free objects.
    """

    joints: List[ActiveConnection1DOF] = field(default_factory=list)
    """
    The connections with one degree of freedom, published as joint values.
    """

    connections: List[Connection] = field(default_factory=list)
    """
    Every connection.
    """

    robots: List[AbstractRobot] = field(default_factory=list)
    """
    The robot annotations.
    """

    @classmethod
    def of_world(cls, world: World, tracked: Iterable[UUID] = ()) -> WorldModelView:
        """
        :param world: The world to read.
        :param tracked: Identities of objects published before.
        :return: The world's current model.
        """
        connections = list(world.connections)
        return cls(
            world=world,
            objects=WorldObjects(world).objects(tracked),
            joints=[
                connection
                for connection in connections
                if isinstance(connection, ActiveConnection1DOF)
            ],
            connections=connections,
            robots=RobotRoster(world).robots(),
        )

    def object_keys(self) -> List[str]:
        """
        :return: The key of every free object: the identity of its body.
        """
        return [ObjectKey.of(body) for body in self.objects]

    def object_with_key(self, key: str) -> Optional[Body]:
        """
        :param key: An object key.
        :return: The object with that key, or None.
        """
        return next((body for body in self.objects if ObjectKey.of(body) == key), None)

    def joint_named(self, name: str) -> Optional[ActiveConnection1DOF]:
        """
        :param name: The prefixed name of a connection.
        :return: The joint of that name, or None.
        """
        return next((joint for joint in self.joints if str(joint.name) == name), None)


@dataclass(frozen=True)
class ObjectKey:
    """
    How the live publisher keys a free object: by the identity of its body.
    """

    @staticmethod
    def of(body: Body) -> str:
        """
        :param body: A free object.
        :return: Its key.
        """
        return str(body.id)
