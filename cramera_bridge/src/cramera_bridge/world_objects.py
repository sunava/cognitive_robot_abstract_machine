"""
The free objects of a world: bodies that move on their own rather than as part of the
robot or the environment model, and bodies a semantic annotation describes as something
a robot takes hold of, wherever they are attached.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Iterable, List, Set
from uuid import UUID

from cramera_bridge.robots import RobotRoster
from semantic_digital_twin.semantic_annotations.mixins import (
    HasGraspCandidates,
    HasMovableJoint,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Furniture,
    Handle,
    Rim,
)
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

MANIPULABLE_KINDS = (HasGraspCandidates,)
"""
The semantic annotations of things a robot takes hold of: food, tableware, tools and
whatever else offers grasps.
"""

PART_KINDS = (Handle, Rim, HasMovableJoint, Furniture)
"""
The annotations among :data:`MANIPULABLE_KINDS` that describe a part of something else,
such as a drawer's handle, a door or a cup's rim, which moves with what it belongs to
rather than on its own.
"""

# %% discovery


@dataclass(frozen=True)
class WorldObjects:
    """
    Selects a world's free objects from its connections, identifying bodies by UUID.

    A free object is a body held by a connection with six degrees of freedom, or the
    root body of an annotation of something a robot takes hold of however it is
    attached, that is neither part of a robot nor carries one, together with the shaped
    bodies fixed below it. An object stays tracked while it is attached to a gripper or
    a support, until it leaves the world.
    """

    world: World
    """
    The world whose current bodies are read.
    """

    def robot_bodies(self) -> Set[UUID]:
        """
        :return: The identities of every body of every robot.
        """
        return {
            body.id
            for robot in RobotRoster(self.world).robots()
            for body in robot.bodies
        }

    def robot_carriers(self) -> Set[UUID]:
        """
        :return: The identities of the bodies above a robot's root, such as the odometry
            frame a robot drives in.
        """
        carriers: Set[UUID] = set()
        for robot in RobotRoster(self.world).robots():
            connection = robot.root.parent_connection
            while connection is not None:
                carriers.add(connection.parent.id)
                connection = connection.parent.parent_connection
        return carriers

    def free_roots(self) -> List[Body]:
        """
        :return: The bodies held by six degrees of freedom outside every robot.
        """
        excluded = self.robot_bodies() | self.robot_carriers()
        return [
            body
            for body in self.world.bodies
            if isinstance(body.parent_connection, Connection6DoF)
            and body.id not in excluded
        ]

    def manipulable_roots(self) -> List[Body]:
        """
        :return: The root bodies of the annotations of things a robot takes hold of,
            outside every robot and without the parts of other things, however they are
            attached, as a demo that fixes its objects to the environment attaches them.
        """
        excluded = self.robot_bodies()
        return [
            annotation.root
            for annotation in self.world.semantic_annotations
            if isinstance(annotation, MANIPULABLE_KINDS)
            and not isinstance(annotation, PART_KINDS)
            and annotation.root.id not in excluded
        ]

    def objects(self, tracked: Iterable[UUID] = ()) -> List[Body]:
        """
        :param tracked: Identities of objects published before, which stay objects
            after a grasp or a placement attached them to another body.
        :return: The free objects and the bodies of things a robot takes hold of, in the
            order the world holds them.
        """
        robot_bodies = self.robot_bodies()
        roots = (
            {body.id for body in self.free_roots()}
            | {body.id for body in self.manipulable_roots()}
            | set(tracked)
        )
        present = [body for body in self.world.bodies if body.id in roots]
        published = {body.id for body in present}
        for root in present:
            for entity in self.world.get_kinematic_structure_entities_of_branch(root):
                if (
                    isinstance(entity, Body)
                    and entity.id not in robot_bodies
                    and (entity.visual.shapes or entity.collision.shapes)
                ):
                    published.add(entity.id)
        return [body for body in self.world.bodies if body.id in published]
