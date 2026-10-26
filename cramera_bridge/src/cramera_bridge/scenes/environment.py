"""
CRAM worlds built around a scene: the steps every scene environment shares.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Optional, Sequence

from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    FileUriResolver,
    PackageUriResolver,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% environments


@dataclass
class SceneEnvironment(ABC):
    """
    Creates independent CRAM worlds holding a scene, with any requested robots.
    """

    def create_world(self, robots: Sequence[RobotSpecification] = ()) -> World:
        """
        :param robots: The robots to place in the world; none creates only the scene.
        :return: A new world holding the robots and the scene.
        """
        world = WorldSpecification(robots=list(robots)).to_domain_object()
        self.populate(world)
        return world

    @abstractmethod
    def populate(self, world: World) -> None:
        """
        Add the scene to an existing world.

        :param world: The world receiving the scene below its root.
        """

    @staticmethod
    def merge_model(
        world: World,
        description: Path,
        prefix: Optional[str],
        root_pose: Optional[HomogeneousTransformationMatrix] = None,
    ) -> KinematicStructureEntity:
        """
        Merge the links and joints of a URDF into a world, its root fixed to the world's
        root. A mesh the URDF names by a relative path is found beside the URDF,
        wherever the process runs.

        :param world: The world receiving the links.
        :param description: The URDF file.
        :param prefix: The prefix of the merged names; an empty prefix keeps the names
            as the URDF writes them.
        :param root_pose: Where the URDF's root is fixed, the world's origin by default.
        :return: The URDF's root in the world.
        """
        resolver = CompositePathResolver(
            [
                FileUriResolver(base_directory=str(description.parent)),
                PackageUriResolver(),
            ]
        )
        model = URDFParser.from_file(
            str(description), prefix=prefix, path_resolver=resolver
        ).parse()
        root_identity = model.root.id
        with world.modify_world():
            connection = FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=model.root,
                parent_T_connection_expression=(
                    root_pose
                    if root_pose is not None
                    else HomogeneousTransformationMatrix()
                ),
            )
            world.merge_world(model, root_connection=connection)
        return world.get_kinematic_structure_entity_by_id(root_identity)
