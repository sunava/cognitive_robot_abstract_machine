"""
Building CRAM worlds from a scene bundle: the steps every bundle environment shares.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% scene environment


@dataclass
class SceneEnvironment(ABC):
    """
    Creates independent CRAM worlds holding a scene, with any requested robots.
    """

    def create_world(self, *, robots: Sequence[RobotSpecification] = ()) -> World:
        """
        Create an independent world holding the scene and any requested robots.

        :param robots: Robot descriptions and placements; empty creates only the scene.
        :return: A semantic digital twin usable by ordinary CRAM contexts and plans.
        """
        world = WorldSpecification(robots=list(robots)).to_domain_object()
        self.populate(world)
        return world

    @abstractmethod
    def populate(self, world: World) -> None:
        """
        Add the scene to an existing world.

        :param world: The world receiving the scene at its root origin.
        """

    @staticmethod
    def merge_environment(
        world: World, description: Path, prefix: str
    ) -> KinematicStructureEntity:
        """
        Merge the links and joints of a URDF into a world, its root fixed at the world
        root's origin.

        :param world: The world receiving the links.
        :param description: The URDF file.
        :param prefix: The prefix of the merged link and joint names.
        :return: The URDF's root in the world.
        """
        environment = URDFParser.from_file(str(description), prefix=prefix).parse()
        root_id = environment.root.id
        with world.modify_world():
            connection = FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=environment.root,
                parent_T_connection_expression=HomogeneousTransformationMatrix(),
            )
            world.merge_world(environment, root_connection=connection)
        return world.get_kinematic_structure_entity_by_id(root_id)
