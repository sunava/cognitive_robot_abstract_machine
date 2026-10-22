"""
Load a scene bundle converted from a USD stage by ``cramera-import`` into CRAM worlds,
with the semantic attributes the stage authored on its prims as annotations that EQL
queries can select.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, Self

from cramera import paths
from cramera.knowledge.presets import Preset
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.queryable_knowledge import QueryableKnowledge, QueryScope
from cramera.knowledge.detected_events import SceneField
from cramera.live.bridge import Bridge
from cramera.live.http import DEFAULT_PORT
from cramera.live.query import LiveQuerySource
from cramera.live.visualization import LiveVisualization
from cramera.live.world_query import WorldQuerySource
from cramera.onboard.usd_scene.bundle import BundleKey
from cramera.onboard.usd_scene.bundle_layout import BundleFile
from cramera.onboard.usd_scene.exceptions import (
    SceneBundleNotFound,
    SceneNotFound,
    UnknownPrimLink,
)
from cramera.onboard.usd_scene.scene_description import (
    OntologyAttributeName,
    SemanticAttributeName,
    SemanticNamespace,
)
from cramera.scene_environment import SceneEnvironment
from cramera.scene_presentation import PresentationField, ScenePresentation
from semantic_digital_twin.api import BodySpecification, Connection6DoFSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Desk,
    Fridge,
    Houseplant,
    Shelf,
    Sink,
    Table,
)
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


# %% prim labels
@dataclass(frozen=True)
class PrimAttribute:
    """
    One ``sdt`` attribute as the stage authored it.
    """

    name: str
    """
    The attribute name without its namespace.
    """

    value: Any
    """
    The authored value as JSON: a string, number, boolean or list.
    """


@dataclass(eq=False)
class UsdPrimAnnotation(HasRootBody):
    """
    The semantic attributes of one USD prim, on the body holding the prim's geometry.

    Several prims merged into one link give one label each, told apart by their path.
    """

    prim_path: str = field(kw_only=True)
    """
    The USD path of the prim.
    """

    semantic_class: str | None = field(default=None, kw_only=True)
    """
    The authored ``sdt:class``.
    """

    entity_type: str | None = field(default=None, kw_only=True)
    """
    The authored ``sdt:entityType``.
    """

    canonical_name: str | None = field(default=None, kw_only=True)
    """
    The authored ``sdt:canonicalName``.
    """

    component_id: str | None = field(default=None, kw_only=True)
    """
    The authored ``sdt:componentId``.
    """

    capabilities: list[str] = field(default_factory=list, kw_only=True)
    """
    The authored ``sdt:capabilities``.
    """

    ontology_types: list[str] = field(default_factory=list, kw_only=True)
    """
    The authored ``ont:types``.
    """

    inferred_ontology_types: list[str] = field(default_factory=list, kw_only=True)
    """
    The authored ``ont:inferredTypes``.
    """

    attributes: list[PrimAttribute] = field(default_factory=list, kw_only=True)
    """
    Every ``sdt`` attribute of the prim, including those with a field of their own.
    """

    def __post_init__(self) -> None:
        """
        Name the label after its prim unless a name is given.
        """
        if self.name is None:
            self.name = PrefixedName(self.prim_path)
        super().__post_init__()

    def __hash__(self) -> int:
        """
        Identify the label by its type, body and prim, so labels of prims sharing a
        body are distinct.
        """
        return hash((self.__class__, self.root, self.prim_path))


# %% bundle description
@dataclass(frozen=True)
class AnnotatedPrim:
    """
    One prim entry of ``semantics.json``.
    """

    path: str
    """
    The USD path of the prim.
    """

    link: str
    """
    The name of the link holding the prim's geometry.
    """

    semantic_attributes: dict[str, Any]
    """
    The prim's ``sdt`` attributes by name.
    """

    ontology_attributes: dict[str, Any]
    """
    The prim's ``ont`` attributes by name.
    """

    @classmethod
    def from_json(cls, path: str, entry: dict[str, Any]) -> Self:
        """
        Read one prim entry.

        :param path: The USD path the entry is keyed by.
        :param entry: The entry.
        """
        return cls(
            path=path,
            link=entry[BundleKey.LINK],
            semantic_attributes=entry.get(SemanticNamespace.SEMANTIC_DIGITAL_TWIN, {}),
            ontology_attributes=entry.get(SemanticNamespace.ONTOLOGY, {}),
        )

    @property
    def semantic_class(self) -> str | None:
        """
        The authored ``sdt:class``, or None.
        """
        return self.semantic_attributes.get(SemanticAttributeName.CLASS)

    def label(self, body: Body) -> UsdPrimAnnotation:
        """
        The label of this prim on a body.

        :param body: The body of the prim's link.
        """
        semantic = self.semantic_attributes
        ontology = self.ontology_attributes
        return UsdPrimAnnotation(
            root=body,
            prim_path=self.path,
            semantic_class=self.semantic_class,
            entity_type=semantic.get(SemanticAttributeName.ENTITY_TYPE),
            canonical_name=semantic.get(SemanticAttributeName.CANONICAL_NAME),
            component_id=semantic.get(SemanticAttributeName.COMPONENT_ID),
            capabilities=list(semantic.get(SemanticAttributeName.CAPABILITIES, [])),
            ontology_types=list(ontology.get(OntologyAttributeName.TYPES, [])),
            inferred_ontology_types=list(
                ontology.get(OntologyAttributeName.INFERRED_TYPES, [])
            ),
            attributes=[
                PrimAttribute(name=name, value=value)
                for name, value in semantic.items()
            ],
        )


@dataclass(frozen=True)
class SceneModel:
    """
    One URDF model entry of ``scene.json``.
    """

    description: Path
    """
    The bundle-relative URDF file.
    """

    prefix: str
    """
    The prefix of the model's link and joint names.
    """

    robot: bool
    """
    Whether the model is the recorded robot.
    """

    @classmethod
    def from_json(cls, entry: dict[str, Any]) -> Self:
        """
        Read one model entry.

        :param entry: The entry.
        """
        return cls(
            description=Path(entry[BundleKey.URDF]),
            prefix=entry[BundleKey.PREFIX],
            robot=entry[PresentationField.ROBOT],
        )


@dataclass(frozen=True)
class SceneObject:
    """
    One free object entry of ``scene.json``: a mesh that can be moved on its own.
    """

    key: str
    """
    The object's identifier, used as its body name.
    """

    mesh: Path
    """
    The bundle-relative mesh file.
    """

    pose: Pose
    """
    The initial pose in the scene frame.
    """

    @classmethod
    def from_json(cls, entry: dict[str, Any]) -> Self:
        """
        Read one object entry.

        :param entry: The entry.
        """
        return cls(
            key=entry[PresentationField.KEY],
            mesh=Path(entry[BundleKey.MESH]),
            pose=Pose.from_xyz_quaternion(*entry[BundleKey.SPAWN]),
        )

    def spawn(self, world: World, bundle_directory: Path) -> Body:
        """
        Add the object to a world as a free body at its initial pose.

        :param world: The world receiving the object.
        :param bundle_directory: The bundle the mesh path is relative to.
        :return: The object's body.
        """
        return BodySpecification(
            name=self.key,
            shapes=ShapeCollection(),
            visual_shapes=ShapeCollection(
                [Mesh(filename=str(bundle_directory / self.mesh))]
            ),
            connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent_T_self=self.pose.to_homogeneous_matrix())


@dataclass(frozen=True)
class UsdSceneDescription:
    """
    What a converted scene bundle holds for building a world.
    """

    title: str
    """
    The environment's name.
    """

    models: list[SceneModel]
    """
    The URDF models of the scene.
    """

    objects: list[SceneObject]
    """
    The free objects of the scene.
    """

    prims: list[AnnotatedPrim]
    """
    The prims carrying semantic attributes.
    """

    initial_joint_values: dict[str, float]
    """
    The joint values of the first trajectory frame, keyed as the trajectory keys them.
    """

    @classmethod
    def from_bundle(cls, directory: Path) -> Self:
        """
        Read the scene, semantics and trajectory files of a bundle.

        :param directory: The bundle directory.
        """
        scene = json.loads((directory / BundleFile.SCENE).read_text(encoding="utf-8"))
        semantics = json.loads(
            (directory / BundleFile.SEMANTICS).read_text(encoding="utf-8")
        )
        trajectory = json.loads(
            (directory / scene[BundleKey.TRAJECTORY]).read_text(encoding="utf-8")
        )
        return cls(
            title=scene[SceneField.ENVIRONMENT_NAME],
            models=[
                SceneModel.from_json(entry) for entry in scene[PresentationField.MODELS]
            ],
            objects=[
                SceneObject.from_json(entry)
                for entry in scene[PresentationField.OBJECTS]
            ],
            prims=[
                AnnotatedPrim.from_json(path, entry)
                for path, entry in semantics[BundleKey.PRIMS].items()
            ],
            initial_joint_values=trajectory[BundleKey.FRAMES][0],
        )

    @property
    def environment_models(self) -> list[SceneModel]:
        """
        The models that are not the recorded robot.
        """
        return [model for model in self.models if not model.robot]


# %% semantic class mapping
class TwinObjectClass(StrEnum):
    """
    The ``sdt:class`` values with a matching semantic digital twin annotation.
    """

    LAB_REFRIGERATOR = "lab_refrigerator"
    """
    A laboratory refrigerator.
    """

    STORAGE_SHELVING = "storage_shelving"
    """
    A free-standing shelving rack.
    """

    LAB_SINK_UNIT = "lab_sink_unit"
    """
    A sink unit with its tap.
    """

    POTTED_PLANT = "potted_plant"
    """
    A plant in a planter.
    """

    OPERATOR_DESK = "operator_desk"
    """
    A desk for an operator.
    """

    GAS_CYLINDER_CABINET = "gas_cylinder_cabinet"
    """
    A cabinet holding gas cylinders.
    """

    LAB_BENCH_ISLAND = "lab_bench_island"
    """
    A free-standing laboratory bench.
    """

    LAB_BENCH_WALL = "lab_bench_wall"
    """
    A laboratory bench along a wall.
    """


@dataclass
class SemanticClassMapping:
    """
    Which semantic digital twin annotation an authored ``sdt:class`` stands for.
    """

    annotation_types: dict[str, type[HasRootBody]] = field(
        default_factory=lambda: {
            TwinObjectClass.LAB_REFRIGERATOR: Fridge,
            TwinObjectClass.STORAGE_SHELVING: Shelf,
            TwinObjectClass.LAB_SINK_UNIT: Sink,
            TwinObjectClass.POTTED_PLANT: Houseplant,
            TwinObjectClass.OPERATOR_DESK: Desk,
            TwinObjectClass.GAS_CYLINDER_CABINET: Cabinet,
            TwinObjectClass.LAB_BENCH_ISLAND: Table,
            TwinObjectClass.LAB_BENCH_WALL: Table,
        }
    )
    """
    The annotation type per ``sdt:class`` value.
    """

    def annotation_for(self, prim: AnnotatedPrim, body: Body) -> HasRootBody | None:
        """
        The annotation the prim's class stands for, named after its body, or None
        when the class has no counterpart.

        :param prim: The prim.
        :param body: The body of the prim's link.
        """
        annotation_type = self.annotation_types.get(prim.semantic_class)
        if annotation_type is None:
            return None
        return annotation_type(name=body.name, root=body)


# %% live queries
class UsdSceneQueryName(StrEnum):
    """
    The variables a query about a converted scene may range over besides the world's.
    """

    PRIM = "prim"
    """
    The labels of the annotated prims.
    """


class UsdSceneQuestion(StrEnum):
    """
    The ready-made questions about a converted scene.
    """

    PRIMS = "show all annotated prims"
    """
    List every prim label.
    """

    CAPABILITIES = "which prims offer capabilities?"
    """
    List the labels of prims with authored capabilities.
    """


@dataclass
class UsdSceneQuerySource(LiveQuerySource):
    """
    Answer questions about a world holding a converted scene: the standard world
    variables, and the prim labels.
    """

    world: World
    """
    The world read afresh whenever a question is evaluated.
    """

    scene_title: str
    """
    The name of the converted environment.
    """

    world_source: WorldQuerySource = field(init=False)
    """
    The standard questions about the world.
    """

    def __post_init__(self) -> None:
        """
        Ask the standard world questions of the same world.
        """
        self.world_source = WorldQuerySource(self.world)

    def title(self) -> str:
        """
        Name the converted environment.
        """
        return self.scene_title

    def knowledge(self) -> list[QueryableKnowledge]:
        """
        The world's knowledge, with the prim labels added to its current state.
        """
        return [
            self.with_prims(knowledge) for knowledge in self.world_source.knowledge()
        ]

    def with_prims(self, knowledge: QueryableKnowledge) -> QueryableKnowledge:
        """
        Add the prim labels to the current state; other knowledge stays as it is.

        :param knowledge: One body of the world's knowledge.
        """
        if knowledge.scope is not QueryScope.CURRENT_STATE:
            return knowledge
        return replace(
            knowledge,
            domains=[
                *knowledge.domains,
                QueryDomain(
                    UsdSceneQueryName.PRIM,
                    UsdPrimAnnotation,
                    self.world.get_semantic_annotations_by_type(UsdPrimAnnotation),
                ),
            ],
            extra_names={
                **knowledge.extra_names,
                PrimAttribute.__name__: PrimAttribute,
            },
        )

    def presets(self) -> list[Preset]:
        """
        The world's questions, then questions about the prim labels.
        """
        prim = UsdSceneQueryName.PRIM
        return self.world_source.presets() + [
            Preset(UsdSceneQuestion.PRIMS, f"an(entity({prim}))"),
            Preset(
                UsdSceneQuestion.CAPABILITIES,
                f"an(entity({prim}).where({prim}.capabilities != []))",
            ),
        ]


# %% environment
@dataclass
class UsdSceneEnvironment(SceneEnvironment):
    """
    Create CRAM worlds from a converted scene bundle: its environment links and joints,
    its free objects and its prim labels.

    The recorded robot of a bundle is not loaded; a robot to plan with is passed to
    :meth:`create_world`.
    """

    bundle_directory: Path
    """
    The directory of the converted scene bundle.
    """

    class_mapping: SemanticClassMapping = field(
        default_factory=SemanticClassMapping, kw_only=True
    )
    """
    The annotations added for authored classes.
    """

    description: UsdSceneDescription = field(init=False)
    """
    What the bundle holds.
    """

    def __post_init__(self) -> None:
        """
        Read the bundle.
        """
        self.bundle_directory = Path(self.bundle_directory).expanduser().resolve()
        if not (self.bundle_directory / BundleFile.SCENE).is_file():
            raise SceneBundleNotFound(self.bundle_directory)
        self.description = UsdSceneDescription.from_bundle(self.bundle_directory)

    @classmethod
    def from_scene_name(cls, name: str) -> Self:
        """
        The environment of a bundle in CRAMERA's scene roots.

        :param name: The scene name, as ``cramera-import --name`` wrote it.
        """
        directory = paths.resolve_scene_directory(name)
        if directory is None:
            raise SceneNotFound(name)
        return cls(bundle_directory=directory)

    def populate(self, world: World) -> None:
        """
        Add the scene to an existing world: the environment at the root origin with its
        joints at their authored values, the free objects, and the prim labels.

        :param world: The world receiving the scene.
        """
        links = {}
        for model in self.description.environment_models:
            links.update(self.add_environment(world, model))
        for item in self.description.objects:
            item.spawn(world, self.bundle_directory)
        self.annotate(world, links)

    def add_environment(self, world: World, model: SceneModel) -> dict[str, Body]:
        """
        Merge one environment model below the world root and set its joints to their
        authored values.

        :param world: The world receiving the model.
        :param model: The environment model.
        :return: The model's bodies by link name.
        """
        root = self.merge_environment(
            world, self.bundle_directory / model.description, model.prefix
        )
        bodies = [
            entity
            for entity in world.get_kinematic_structure_entities_of_branch(root)
            if isinstance(entity, Body)
        ]
        for body in bodies:
            self.apply_initial_joint_value(body)
        return {body.name.name: body for body in bodies}

    def apply_initial_joint_value(self, body: Body) -> None:
        """
        Move a body's joint to the value of the first trajectory frame, if it has one.

        :param body: A body of an environment model.
        """
        connection = body.parent_connection
        if not isinstance(connection, ActiveConnection1DOF):
            return
        value = self.description.initial_joint_values.get(str(connection.name))
        if value is None:
            return
        connection.position = value

    def annotate(self, world: World, links: dict[str, Body]) -> None:
        """
        Label every annotated prim on its link's body and add the annotation its class
        stands for.

        :param world: The world holding the bodies.
        :param links: The environment bodies by link name.
        """
        unknown = [prim for prim in self.description.prims if prim.link not in links]
        if unknown:
            raise UnknownPrimLink(unknown[0].path, unknown[0].link)
        with world.modify_world():
            for prim in self.description.prims:
                body = links[prim.link]
                world.add_semantic_annotation(prim.label(body))
                annotation = self.class_mapping.annotation_for(prim, body)
                if annotation is not None:
                    world.add_semantic_annotation(annotation)

    def query_source(self, world: World) -> UsdSceneQuerySource:
        """
        What the live bridge answers questions about a world holding this scene from.

        :param world: The world holding the scene.
        """
        return UsdSceneQuerySource(world=world, scene_title=self.description.title)

    def create_visualization(
        self, world: World, port: int = DEFAULT_PORT
    ) -> LiveVisualization:
        """
        A live visualization of a world holding this scene, showing it as authored and
        answering questions about its prims; call
        :meth:`~cramera.live.visualization.LiveVisualization.start` to serve it.

        :param world: The world holding the scene.
        :param port: The port of the live bridge.
        """
        bridge = Bridge(
            presentation=ScenePresentation.from_bundle(self.bundle_directory)
        )
        bridge.register_query_source(self.query_source(world))
        return LiveVisualization(world, port=port, bridge=bridge)
