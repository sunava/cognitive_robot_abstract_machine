"""
The semantic annotations CRAM's world reasoner infers for an environment that carries
none of its own, such as one loaded from a URDF: its handles, drawers, doors and
surfaces, each named after the body it is rooted in so a plan can name it.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Iterable, List

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)


@dataclass(frozen=True)
class InferredAnnotations:
    """
    Adds the annotations the world reasoner infers for an environment's bodies.
    """

    world: World
    """
    The world holding the environment, and possibly robots.
    """

    def add(self, environment: Iterable[Body]) -> List[SemanticAnnotation]:
        """
        Infer the world's annotations and keep those of the environment: an annotation
        rooted in another body, such as a robot's, is removed again. Each kept
        annotation with a root is named after it, since the reasoner names every
        annotation after its class.

        :param environment: The environment's bodies.
        :return: The kept annotations.
        """
        bodies = {body.id for body in environment}
        inferred = WorldReasoner(self.world).infer_semantic_annotations()
        kept: List[SemanticAnnotation] = []
        with self.world.modify_world():
            for annotation in inferred:
                if not isinstance(annotation, HasRootKinematicStructureEntity):
                    kept.append(annotation)
                    continue
                if annotation.root.id not in bodies:
                    self.world.remove_semantic_annotation(annotation)
                    continue
                annotation.name = PrefixedName(
                    annotation.root.name.name, annotation.name.prefix
                )
                kept.append(annotation)
        return kept
