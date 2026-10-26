"""
CRAM worlds built from a scene bundle: its environment models with their joints at the
first recorded values, its free objects, and the semantic attributes of a bundle
converted from a USD stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Dict, List

from cramera.scenes import SceneCatalog
from cramera_bridge.documents import DocumentFile
from cramera_bridge.exceptions import BundleNotFound, UnknownPrimLink
from cramera_bridge.geometry import COLOR_CHANNEL_MAXIMUM
from cramera_bridge.scenes.environment import SceneEnvironment
from cramera_bridge.scenes.inferred_annotations import InferredAnnotations
from cramera_bridge.scenes.semantics import (
    SemanticClassMapping,
    Semantics,
    UsdPrimAnnotation,
)
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.bundle.scene import (
    ModelDescription,
    ObjectDescription,
    SceneDescription,
)
from cramera_protocol.bundle.trajectory import Trajectory
from cramera_protocol.geometry import Pose
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Mesh,
    Scale,
    Shape,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

NAME_SEPARATOR = "/"
"""
What separates the prefix of a full body name from its own name.
"""

HEXADECIMAL_BASE = 16
"""
The base of the digits of a CSS hexadecimal colour.
"""

CHANNEL_DIGITS = 2
"""
How many digits of a CSS hexadecimal colour one channel takes.
"""

COLOR_CHANNELS = 3
"""
The red, green and blue channels of a CSS hexadecimal colour.
"""

# %% values


@dataclass(frozen=True)
class ColorText:
    """
    A CSS hexadecimal colour as a native colour.
    """

    text: str
    """
    The colour, such as ``#3366cc``.
    """

    def native(self) -> Color:
        """
        :return: The colour with full opacity.
        """
        digits = self.text.lstrip("#")
        channels = [
            int(
                digits[channel * CHANNEL_DIGITS : (channel + 1) * CHANNEL_DIGITS],
                HEXADECIMAL_BASE,
            )
            / COLOR_CHANNEL_MAXIMUM
            for channel in range(COLOR_CHANNELS)
        ]
        return Color(*channels)


@dataclass(frozen=True)
class PoseText:
    """
    A protocol pose as a native transformation.
    """

    pose: Pose
    """
    The pose.
    """

    def native(self, reference_frame: Body) -> HomogeneousTransformationMatrix:
        """
        :param reference_frame: The frame the pose is relative to.
        :return: The transformation.
        """
        return HomogeneousTransformationMatrix.from_xyz_quaternion(
            self.pose.x,
            self.pose.y,
            self.pose.z,
            self.pose.qx,
            self.pose.qy,
            self.pose.qz,
            self.pose.qw,
            reference_frame=reference_frame,
        )


@dataclass(frozen=True)
class FullName:
    """
    A full body name, such as ``kitchen/milk``, as a native name.
    """

    text: str
    """
    The full name.
    """

    def native(self) -> PrefixedName:
        """
        :return: The name, its prefix everything before the last separator.
        """
        prefix, separator, name = self.text.rpartition(NAME_SEPARATOR)
        return PrefixedName(name, prefix if separator else None)


# %% bundle contents


@dataclass(frozen=True)
class BundleContents:
    """
    What a bundle holds for building a world.
    """

    directory: Path
    """
    The bundle directory.
    """

    scene: SceneDescription
    """
    The scene description.
    """

    initial_joint_values: Dict[str, float]
    """
    The joint values of the first trajectory frame.
    """

    semantics: Semantics
    """
    The semantic attributes of the prims, empty for a bundle without them.
    """

    @classmethod
    def of_directory(cls, directory: Path) -> BundleContents:
        """
        :param directory: A bundle directory.
        :return: Its contents.
        :raises BundleNotFound: When the directory holds no ``scene.json``.
        """
        scene_file = DocumentFile(directory / BundleFile.SCENE)
        if not scene_file.path.is_file():
            raise BundleNotFound(directory)
        scene = scene_file.read(SceneDescription)
        trajectory = (
            DocumentFile(directory / scene.trajectory).read(Trajectory)
            if scene.trajectory is not None
            else None
        )
        semantics = DocumentFile(directory / BundleFile.SEMANTICS).read_if_present(
            Semantics
        )
        return cls(
            directory=directory,
            scene=scene,
            initial_joint_values=(
                dict(trajectory.frames[0])
                if trajectory is not None and trajectory.frames
                else {}
            ),
            semantics=semantics if semantics is not None else Semantics(),
        )

    def environment_models(self) -> List[ModelDescription]:
        """
        :return: The models that are not robots.
        """
        return [model for model in self.scene.models if not model.robot]


# %% environment


@dataclass
class BundleEnvironment(SceneEnvironment):
    """
    Builds CRAM worlds from a scene bundle.

    The bundle's robots are not loaded: the robots to plan with are passed to
    :meth:`~cramera_bridge.scenes.environment.SceneEnvironment.create_world`.
    """

    directory: Path
    """
    The bundle directory.
    """

    class_mapping: SemanticClassMapping = field(
        default_factory=SemanticClassMapping, kw_only=True
    )
    """
    The annotations added for authored prim classes.
    """

    contents: BundleContents = field(init=False)
    """
    What the bundle holds.
    """

    def __post_init__(self) -> None:
        self.directory = Path(self.directory).expanduser().resolve()
        self.contents = BundleContents.of_directory(self.directory)

    @classmethod
    def of_scene(cls, catalog: SceneCatalog, name: str) -> BundleEnvironment:
        """
        :param catalog: The scenes directories the viewer shows.
        :param name: The name of a bundle of the catalog.
        :return: The bundle's environment.
        :raises UnknownScene: When no directory of the catalog holds the bundle.
        """
        return cls(directory=catalog.directory_of(name).path / name)

    def populate(self, world: World) -> None:
        """
        Add the environment models with their joints at the first recorded values, the
        free objects, and the prim annotations; an environment without prims, such as
        one loaded from a URDF, gets the annotations CRAM's reasoner infers instead.

        :param world: The world receiving the scene.
        """
        links: Dict[str, Body] = {}
        for model in self.contents.environment_models():
            links.update(self.add_model(world, model))
        for description in self.contents.scene.objects:
            self.add_object(world, description)
        self.annotate(world, links)
        if not self.contents.semantics.prims:
            InferredAnnotations(world).add(links.values())

    def add_model(self, world: World, model: ModelDescription) -> Dict[str, Body]:
        """
        :param world: The world receiving the model.
        :param model: An environment model.
        :return: The model's bodies by link name.
        """
        root = self.merge_model(
            world,
            self.directory / model.urdf,
            model.prefix,
            PoseText(model.pose).native(world.root) if model.pose else None,
        )
        bodies = [
            entity
            for entity in world.get_kinematic_structure_entities_of_branch(root)
            if isinstance(entity, Body)
        ]
        for body in bodies:
            connection = body.parent_connection
            if not isinstance(connection, ActiveConnection1DOF):
                continue
            value = self.contents.initial_joint_values.get(str(connection.name))
            if value is not None:
                connection.position = value
        return {body.name.name: body for body in bodies}

    def add_object(self, world: World, description: ObjectDescription) -> Body:
        """
        Add a free object at its spawn pose, drawn from its mesh or box; an object
        without either has no shapes.

        :param world: The world receiving the object.
        :param description: The object's description.
        :return: The object's body.
        """
        shapes: List[Shape] = []
        if description.mesh is not None:
            shapes.append(Mesh(filename=str(self.directory / description.mesh)))
        elif description.box is not None:
            box = description.box
            shapes.append(
                Box(
                    scale=Scale(box.x, box.y, box.z),
                    color=(
                        ColorText(description.color).native()
                        if description.color is not None
                        else Color()
                    ),
                )
            )
        body = Body(
            name=FullName(description.identifier).native(),
            visual=ShapeCollection(shapes=shapes),
        )
        with world.modify_world():
            connection = Connection6DoF.create_with_dofs(
                parent=world.root, child=body, world=world
            )
            world.add_connection(connection)
        connection.origin = PoseText(description.spawn).native(world.root)
        return body

    def annotate(self, world: World, links: Dict[str, Body]) -> None:
        """
        Annotate every prim on its link's body, together with the annotation its class
        stands for.

        :param world: The world holding the bodies.
        :param links: The environment's bodies by link name.
        :raises UnknownPrimLink: When a prim names a link the environment lacks.
        """
        prims = self.contents.semantics.prims
        for path, entry in prims.items():
            if entry.link not in links:
                raise UnknownPrimLink(path, entry.link)
        with world.modify_world():
            for path, entry in prims.items():
                body = links[entry.link]
                world.add_semantic_annotation(
                    UsdPrimAnnotation.of_prim(path, entry, body)
                )
                annotation = self.class_mapping.annotation_for(entry, body)
                if annotation is not None:
                    world.add_semantic_annotation(annotation)


# %% summary


@dataclass(frozen=True)
class WorldSummary:
    """
    What a world built from a bundle holds, for a person checking it.
    """

    world: World
    """
    The world.
    """

    def movable_joint_count(self) -> int:
        """
        :return: How many connections move along one degree of freedom.
        """
        return sum(
            isinstance(connection, ActiveConnection1DOF)
            for connection in self.world.connections
        )

    def prim_annotation_count(self) -> int:
        """
        :return: How many prim annotations the world holds.
        """
        return len(self.world.get_semantic_annotations_by_type(UsdPrimAnnotation))

    def text(self) -> str:
        """
        :return: One fact per line.
        """
        prim_annotations = self.prim_annotation_count()
        return "\n".join(
            [
                f"Bodies: {len(self.world.bodies)}",
                f"Movable joints: {self.movable_joint_count()}",
                f"Prim annotations: {prim_annotations}",
                "Class annotations: "
                f"{len(self.world.semantic_annotations) - prim_annotations}",
            ]
        )
