"""
Questions about a live world: its bodies, objects and semantic annotations as the ready-
made variables of the entity query language, and the ready-made questions the world's
content suggests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any, Callable, Iterator, List, Type

from cramera_bridge.geometry import NumericPose
from cramera_bridge.questions.desk import QuestionDesk
from cramera_bridge.questions.highlights import HighlightKeys, ValueFormat
from cramera_bridge.questions.knowledge import QueryDomain, QueryableKnowledge
from cramera_bridge.questions.presets import EntityListing, Preset, RobotQuestions
from cramera_bridge.questions.workspace_classes import WorkspaceClassIndex
from cramera_protocol.questions import QueryScope
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

ObjectSource = Callable[[], List[Body]]
"""
Answers the world's current free objects.
"""


class WorldVariable(StrEnum):
    """
    The ready-made variables of questions about a live world.
    """

    BODY = "body"
    """
    Every body.
    """

    OBJECT = "scene_object"
    """
    The free objects.
    """

    ANNOTATION = "annotation"
    """
    Every semantic annotation.
    """

    HANDLE = "handle"
    """
    The handles.
    """

    SURFACE = "surface"
    """
    The supporting surfaces, containers included.
    """

    ROBOT = "robot"
    """
    The robots.
    """

    ARM = "arm"
    """
    The arms.
    """

    WORLD = "world"
    """
    The world itself.
    """


class WorldQuestion(StrEnum):
    """
    The ready-made questions about a world's annotations.
    """

    BODIES = "show all scene bodies"
    """
    Lists the bodies.
    """

    ANNOTATIONS = "show all semantic annotations"
    """
    Lists the annotations.
    """

    HANDLES = "show all handles"
    """
    Lists the handles.
    """

    SURFACES = "show all supporting surfaces"
    """
    Lists the supporting surfaces.
    """


LIVE_TITLE = "Live world"
"""
What the live questions are about.
"""

# %% dynamic domains


@dataclass(frozen=True)
class WorldBodies:
    """
    The bodies of a world, read whenever a query iterates them.
    """

    world: World
    """
    The world.
    """

    def __iter__(self) -> Iterator[Body]:
        return iter(list(self.world.bodies))


@dataclass(frozen=True)
class WorldAnnotations:
    """
    The annotations of one type in a world, read whenever a query iterates them.
    """

    world: World
    """
    The world.
    """

    annotation_type: Type[SemanticAnnotation]
    """
    The type of annotation.
    """

    def __iter__(self) -> Iterator[SemanticAnnotation]:
        return iter(
            list(self.world.get_semantic_annotations_by_type(self.annotation_type))
        )


@dataclass(frozen=True)
class CurrentObjects:
    """
    The free objects, read whenever a query iterates them.
    """

    objects: ObjectSource
    """
    Answers the current free objects.
    """

    def __iter__(self) -> Iterator[Body]:
        return iter(self.objects())


# %% knowledge


@dataclass(frozen=True)
class WorldKnowledge:
    """
    What questions about a live world may range over.
    """

    world: World
    """
    The world.
    """

    objects: ObjectSource
    """
    Answers the current free objects.
    """

    def knowledge(self) -> QueryableKnowledge:
        """
        :return: The world's current state as ready-made variables, read afresh by
            every query.
        """
        annotation_domains = [
            QueryDomain(WorldVariable.ANNOTATION, SemanticAnnotation),
            QueryDomain(WorldVariable.HANDLE, Handle),
            QueryDomain(WorldVariable.SURFACE, HasSupportingSurface),
            QueryDomain(WorldVariable.ROBOT, AbstractRobot),
            QueryDomain(WorldVariable.ARM, Arm),
        ]
        return QueryableKnowledge(
            scope=QueryScope.CURRENT_STATE,
            domains=[
                QueryDomain(WorldVariable.BODY, Body, WorldBodies(self.world)),
                QueryDomain(WorldVariable.OBJECT, Body, CurrentObjects(self.objects)),
            ]
            + [
                QueryDomain(
                    domain.name,
                    domain.entity_type,
                    WorldAnnotations(self.world, domain.entity_type),
                )
                for domain in annotation_domains
            ],
            extra_names={WorldVariable.WORLD.value: self.world},
        )

    def presets(self) -> List[Preset]:
        """
        :return: The robot questions suited to the world's robots, then the questions
            about annotations the world holds.
        """
        presets = RobotQuestions(
            len(self.world.get_semantic_annotations_by_type(AbstractRobot))
        ).presets()
        presets.append(
            Preset(WorldQuestion.BODIES, EntityListing(WorldVariable.BODY).every())
        )
        annotation_questions = [
            (WorldQuestion.ANNOTATIONS, WorldVariable.ANNOTATION, SemanticAnnotation),
            (WorldQuestion.HANDLES, WorldVariable.HANDLE, Handle),
            (WorldQuestion.SURFACES, WorldVariable.SURFACE, HasSupportingSurface),
        ]
        presets.extend(
            Preset(question, EntityListing(variable).every())
            for question, variable, annotation_type in annotation_questions
            if self.world.get_semantic_annotations_by_type(annotation_type)
        )
        return presets


# %% presentation


@dataclass(frozen=True)
class WorldHighlightKeys(HighlightKeys):
    """
    Lights up a free object under its key, any other body under its name (the name of
    its link in the bundle), and an annotation through its bodies.
    """

    objects: ObjectSource
    """
    Answers the current free objects.
    """

    def keys_of(self, value: Any) -> List[str]:
        if isinstance(value, Body):
            object_keys = {body.id for body in self.objects()}
            return [str(value.id) if value.id in object_keys else str(value.name)]
        if isinstance(value, SemanticAnnotation):
            return [key for body in value.bodies for key in self.keys_of(body)]
        return []


@dataclass(frozen=True)
class SpatialValueFormat(ValueFormat):
    """
    Writes native poses, transformations and points as their numbers.
    """

    def accepts(self, value: Any) -> bool:
        return isinstance(value, (HomogeneousTransformationMatrix, Pose, Point3))

    def text(self, value: Any) -> str:
        if isinstance(value, Point3):
            return "(%.2f, %.2f, %.2f)" % tuple(value.to_np()[:3])
        return NumericPose.of_matrix(value.to_np()).label


# %% desk


@dataclass
class WorldQuestionDesk:
    """
    Builds the desk answering questions about a live world.
    """

    world: World
    """
    The world.
    """

    objects: ObjectSource
    """
    Answers the current free objects.
    """

    class_index: WorkspaceClassIndex = field(default_factory=WorkspaceClassIndex)
    """
    The workspace classes a query may name.
    """

    def desk(self) -> QuestionDesk:
        """
        :return: A desk over the world's current state with the world's presets.
        """
        knowledge = WorldKnowledge(self.world, self.objects)
        return QuestionDesk(
            title=LIVE_TITLE,
            knowledge=[knowledge.knowledge()],
            presets=knowledge.presets(),
            highlight_keys=WorldHighlightKeys(self.objects),
            class_index=self.class_index,
            value_formats=[SpatialValueFormat()],
        )
