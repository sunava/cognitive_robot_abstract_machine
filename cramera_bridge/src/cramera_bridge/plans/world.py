"""
The world a plan runs in: its environment, an empty floor when it names none, and the
free objects it places, each a mesh or a library model's URDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Dict, List

from cramera_bridge.scenes.bundle_world import ColorText, PoseText
from cramera_bridge.scenes.environment import SceneEnvironment
from cramera_bridge.plans.objects import ObjectLibraries
from cramera_bridge.plans.workspace import OBJECT_DIRECTORY
from cramera_protocol.plan_builder import PlannedObject
from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    PackageUriResolver,
    SearchPathFileResolver,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

URDF_SUFFIX = ".urdf"
"""
The suffix of a library model's description.
"""

PREFIX_SEPARATOR = "_"
"""
Joins an object's identity to the names of the links its URDF brings.
"""

# %% environment


@dataclass
class EmptyFloor(SceneEnvironment):
    """
    The environment of a plan that names none: nothing but the world's root.
    """

    def populate(self, world: World) -> None:
        """
        Add nothing.

        :param world: The world.
        """


# %% objects


@dataclass(eq=False)
class PlannedGraspable(HasGraspCandidates):
    """
    A free object of a plan, which coraplex's actions take hold of and put down: grasped
    around its own origin, as nothing more is known of its shape.
    """

    @classmethod
    def of(cls, body: Body) -> PlannedGraspable:
        """
        :param body: The body of an object the plan placed.
        :return: The object's annotation.
        """
        (annotation,) = [
            annotation
            for annotation in body._world.get_semantic_annotations_by_type(cls)
            if annotation.root is body
        ]
        return annotation


@dataclass(frozen=True)
class PlannedObjects:
    """
    The free objects of a plan, each a copy of an object mesh or of a library model that
    moves on its own.
    """

    objects: List[PlannedObject]
    """
    The objects.
    """

    directory: Path = field(default=OBJECT_DIRECTORY)
    """
    The directory of the object meshes.
    """

    libraries: ObjectLibraries = field(default_factory=ObjectLibraries.of_environment)
    """
    The object libraries the plan's models may come from.
    """

    def spawn(self, world: World) -> Dict[str, Body]:
        """
        :param world: The world receiving the objects at their start poses, below its
            root, each annotated as graspable.
        :return: Each object's body, a library model's root link, by its identity.
        """
        bodies = {
            planned.identifier: self.spawn_one(world, planned)
            for planned in self.objects
        }
        with world.modify_world():
            for body in bodies.values():
                world.add_semantic_annotation(PlannedGraspable(root=body))
        return bodies

    def spawn_one(self, world: World, planned: PlannedObject) -> Body:
        """
        :param world: The world receiving the object below its root.
        :param planned: The object.
        :return: Its body, free to move: the mesh's, or the root link of the URDF of a
            library model with every link named below the object's identity.
        """
        path = self.libraries.path_of(planned.model, self.directory)
        pose = PoseText(planned.pose).native(world.root)
        if path.suffix.lower() != URDF_SUFFIX:
            return BodySpecification.mesh(
                planned.identifier,
                str(path),
                color=(
                    ColorText(planned.color).native()
                    if planned.color is not None
                    else None
                ),
                connection_specification=Connection6DoFSpecification(),
            ).spawn(world, parent_T_self=pose)
        # a library model names its meshes relative to its URDF, not to the process
        resolver = CompositePathResolver(
            resolvers=[
                SearchPathFileResolver(root_directories=[str(path.parent)]),
                PackageUriResolver(),
            ]
        )
        model = URDFParser.from_file(
            str(path),
            prefix=planned.identifier + PREFIX_SEPARATOR,
            path_resolver=resolver,
        ).parse()
        root = model.root.id
        with world.modify_world():
            connection = Connection6DoF.create_with_dofs(
                world=world, parent=world.root, child=model.root
            )
            world.merge_world(model, root_connection=connection)
        body = world.get_kinematic_structure_entity_by_id(root)
        body.parent_connection.origin = pose
        return body
