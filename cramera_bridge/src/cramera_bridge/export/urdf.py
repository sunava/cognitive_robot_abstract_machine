"""
URDF documents of a scene bundle: written from the bodies of a world, or copied from an
existing URDF file with every mesh it references.
"""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy
from scipy.spatial.transform import Rotation
from typing_extensions import Dict, Iterable, List, Optional, Sequence
from uuid import UUID

from cramera_bridge.exceptions import UnrepresentableJointLimits, UrdfSourceNotFound
from cramera_bridge.export.assets import BundledAssets, MeshFormat, MeshReference
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Mesh,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.world_entity import Body, Connection

# %% vocabulary


class UrdfElement(StrEnum):
    """
    The URDF element names written and read.
    """

    ROBOT = "robot"
    """
    The document root.
    """

    LINK = "link"
    """
    A rigid body.
    """

    JOINT = "joint"
    """
    A connection between two links.
    """

    VISUAL = "visual"
    """
    Geometry a link is drawn with.
    """

    COLLISION = "collision"
    """
    Geometry a link collides with.
    """

    GEOMETRY = "geometry"
    """
    The geometry of a visual or collision element.
    """

    ORIGIN = "origin"
    """
    A pose relative to the enclosing element's frame.
    """

    BOX = "box"
    """
    A box geometry.
    """

    SPHERE = "sphere"
    """
    A sphere geometry.
    """

    CYLINDER = "cylinder"
    """
    A cylinder geometry.
    """

    MESH = "mesh"
    """
    A mesh file geometry.
    """

    MATERIAL = "material"
    """
    The appearance of a visual element.
    """

    COLOR = "color"
    """
    The colour of a material.
    """

    PARENT = "parent"
    """
    The parent link of a joint.
    """

    CHILD = "child"
    """
    The child link of a joint.
    """

    AXIS = "axis"
    """
    The axis of a joint.
    """

    LIMIT = "limit"
    """
    The limits of a joint.
    """


class UrdfAttribute(StrEnum):
    """
    The URDF attribute names written and read.
    """

    NAME = "name"
    """
    The name of a link, joint or material.
    """

    TYPE = "type"
    """
    The type of a joint.
    """

    LINK = "link"
    """
    The link a parent or child element names.
    """

    TRANSLATION = "xyz"
    """
    A translation or an axis.
    """

    ROTATION = "rpy"
    """
    A fixed-axis roll, pitch and yaw rotation.
    """

    SIZE = "size"
    """
    The extent of a box.
    """

    RADIUS = "radius"
    """
    The radius of a sphere or cylinder.
    """

    LENGTH = "length"
    """
    The length of a cylinder.
    """

    FILENAME = "filename"
    """
    The file of a mesh.
    """

    SCALE = "scale"
    """
    The scale of a mesh.
    """

    RGBA = "rgba"
    """
    The red, green, blue and opacity of a colour.
    """

    LOWER = "lower"
    """
    The lower position limit.
    """

    UPPER = "upper"
    """
    The upper position limit.
    """

    VELOCITY = "velocity"
    """
    The velocity limit.
    """

    EFFORT = "effort"
    """
    The effort limit.
    """


class UrdfJointType(StrEnum):
    """
    The URDF joint types a connection becomes.
    """

    FIXED = "fixed"
    """
    A connection without degrees of freedom.
    """

    REVOLUTE = "revolute"
    """
    A rotation within limits.
    """

    CONTINUOUS = "continuous"
    """
    A rotation without limits.
    """

    PRISMATIC = "prismatic"
    """
    A translation within limits.
    """

    FLOATING = "floating"
    """
    Any other connection; its pose is streamed rather than a joint value.
    """


AXIS_JOINT_TYPES = frozenset(
    {UrdfJointType.REVOLUTE, UrdfJointType.CONTINUOUS, UrdfJointType.PRISMATIC}
)
"""
The joint types that move along an axis and are driven by one value.
"""

LIMITED_JOINT_TYPES = frozenset({UrdfJointType.REVOLUTE, UrdfJointType.PRISMATIC})
"""
The joint types URDF requires limits for.
"""

COORDINATE_PRECISION = 6
"""
Decimal places a written number keeps.
"""

UNSPECIFIED_LIMIT = 0.0
"""
The value written for a velocity or effort limit the world does not state; URDF requires
the attribute, and the viewer does not read it.
"""

ROOT_LINK = "world_root"
"""
The link a document written from part of a world is rooted in.
"""

URDF_SUFFIX = ".urdf"
"""
The suffix of a written URDF file.
"""

XACRO_SUFFIX = ".xacro"
"""
The suffix of a URDF macro file, expanded before bundling.
"""

# %% numbers


@dataclass(frozen=True)
class UrdfNumbers:
    """
    Numbers as URDF attribute values.
    """

    @staticmethod
    def text(values: Iterable[float]) -> str:
        """
        :param values: Numbers.
        :return: The numbers, rounded and separated by spaces.
        """
        return " ".join(
            str(round(float(value), COORDINATE_PRECISION)) for value in values
        )

    @classmethod
    def origin(cls, parent: ElementTree.Element, matrix: numpy.ndarray) -> None:
        """
        Add an ``origin`` element: a translation and a fixed-axis roll, pitch and yaw.

        :param parent: The element the origin belongs to.
        :param matrix: The pose as a 4x4 homogeneous transformation.
        """
        with warnings.catch_warnings():
            # at a pitch of ±90° the decomposition is not unique, but the chosen one
            # still reproduces the rotation exactly
            warnings.filterwarnings("ignore", message="Gimbal lock detected")
            rotation = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")
        ElementTree.SubElement(
            parent,
            UrdfElement.ORIGIN,
            {
                UrdfAttribute.TRANSLATION: cls.text(matrix[:3, 3]),
                UrdfAttribute.ROTATION: cls.text(rotation),
            },
        )


# %% link names


@dataclass
class LinkNames:
    """
    Unique URDF link names for bodies, which may share a name in their world.
    """

    names: Dict[UUID, str] = field(default_factory=dict)
    """
    The link name of every named body by its identity.
    """

    def name_of(self, body: Body) -> str:
        """
        :param body: A body of the document.
        :return: Its full name, or, when another body of the document already carries
            that name, the name followed by its identity.
        """
        if body.id in self.names:
            return self.names[body.id]
        name = str(body.name)
        if name in self.names.values():
            name = f"{name}__{body.id.hex}"
        self.names[body.id] = name
        return name


# %% world to URDF


@dataclass(frozen=True)
class ModelReport:
    """
    What a URDF document written from a world contains.
    """

    name: str
    """
    The model name.
    """

    urdf: str
    """
    The URDF file, relative to the bundle.
    """

    links: List[str]
    """
    The link names in document order, :data:`ROOT_LINK` first.
    """

    movable_joints: List[str]
    """
    The joints driven by one value, in document order.
    """

    posed_joints: List[str]
    """
    The floating joints whose poses are streamed.
    """

    missing: List[str]
    """
    The mesh files that could not be copied.
    """


@dataclass
class UrdfDocument:
    """
    A URDF document assembled from bodies of a world.

    Connections within the written bodies become joints; a body whose parent is not
    written is fixed to :data:`ROOT_LINK` at the pose it holds in the world.
    """

    world: World
    """
    The world holding the bodies.
    """

    assets: BundledAssets
    """
    Copies mesh files into the bundle.
    """

    mesh_directory: str
    """
    The folder below ``meshes`` the copied meshes go to.
    """

    root: ElementTree.Element = field(init=False, repr=False)
    """
    The document's ``robot`` element.
    """

    link_names: LinkNames = field(default_factory=LinkNames)
    """
    The link name of every written body.
    """

    movable_joints: List[str] = field(default_factory=list)
    """
    The joints driven by one value, in document order.
    """

    posed_joints: List[str] = field(default_factory=list)
    """
    The floating joints, in document order.
    """

    def write(
        self,
        bodies: Sequence[Body],
        name: str,
        identity_root: Optional[Body] = None,
    ) -> ModelReport:
        """
        Write the bodies as ``<name>.urdf`` into the bundle.

        :param bodies: The bodies, parents before their children.
        :param name: The model name.
        :param identity_root: A body fixed at the root link's origin instead of at its
            world pose, such as a robot's base whose pose is applied on top.
        :return: What was written.
        """
        self.root = ElementTree.Element(UrdfElement.ROBOT, {UrdfAttribute.NAME: name})
        ElementTree.SubElement(
            self.root, UrdfElement.LINK, {UrdfAttribute.NAME: ROOT_LINK}
        )
        written = {body.id for body in bodies}
        for body in bodies:
            self.add_link(body)
            connection = body.parent_connection
            if body is identity_root:
                self.add_root_joint(body, numpy.eye(4))
            elif connection is not None and connection.parent.id in written:
                self.add_joint(connection)
            else:
                self.add_root_joint(
                    body,
                    self.world.compute_forward_kinematics_np(self.world.root, body),
                )
        urdf = self.assets.bundle_root / f"{name}{URDF_SUFFIX}"
        self.assets.require_inside(urdf)
        ElementTree.indent(self.root)
        ElementTree.ElementTree(self.root).write(
            urdf, encoding="utf-8", xml_declaration=True
        )
        return ModelReport(
            name=name,
            urdf=self.assets.relative(urdf),
            links=[ROOT_LINK] + [self.link_names.name_of(body) for body in bodies],
            movable_joints=list(self.movable_joints),
            posed_joints=list(self.posed_joints),
            missing=list(self.assets.missing),
        )

    # %% links

    def add_link(self, body: Body) -> None:
        """
        Add a link with one visual element per visual shape and one collision element
        per collision shape; a body without shapes gets neither.

        :param body: The body.
        """
        link = ElementTree.SubElement(
            self.root,
            UrdfElement.LINK,
            {UrdfAttribute.NAME: self.link_names.name_of(body)},
        )
        for index, shape in enumerate(body.visual.shapes):
            visual = self.add_shape(link, UrdfElement.VISUAL, shape)
            material = ElementTree.SubElement(
                visual,
                UrdfElement.MATERIAL,
                {UrdfAttribute.NAME: f"{self.link_names.name_of(body)}#{index}"},
            )
            ElementTree.SubElement(
                material,
                UrdfElement.COLOR,
                {UrdfAttribute.RGBA: UrdfNumbers.text(shape.color.to_rgba())},
            )
        for shape in body.collision.shapes:
            self.add_shape(link, UrdfElement.COLLISION, shape)

    def add_shape(
        self, link: ElementTree.Element, kind: UrdfElement, shape: Shape
    ) -> ElementTree.Element:
        """
        :param link: The link element.
        :param kind: :attr:`UrdfElement.VISUAL` or :attr:`UrdfElement.COLLISION`.
        :param shape: The shape, with its pose relative to its body.
        :return: The added element.
        """
        element = ElementTree.SubElement(link, kind)
        UrdfNumbers.origin(element, shape.origin.to_np())
        geometry = ElementTree.SubElement(element, UrdfElement.GEOMETRY)
        if isinstance(shape, Box):
            ElementTree.SubElement(
                geometry,
                UrdfElement.BOX,
                {UrdfAttribute.SIZE: UrdfNumbers.text(shape.scale.to_np()[:3])},
            )
        elif isinstance(shape, Sphere):
            ElementTree.SubElement(
                geometry,
                UrdfElement.SPHERE,
                {UrdfAttribute.RADIUS: UrdfNumbers.text([shape.radius])},
            )
        elif isinstance(shape, Cylinder):
            ElementTree.SubElement(
                geometry,
                UrdfElement.CYLINDER,
                {
                    UrdfAttribute.RADIUS: UrdfNumbers.text([shape.radius]),
                    UrdfAttribute.LENGTH: UrdfNumbers.text([shape.height]),
                },
            )
        else:
            self.add_mesh(geometry, shape)
        return element

    def add_mesh(self, geometry: ElementTree.Element, mesh: Mesh) -> None:
        """
        Copy a mesh file into the bundle and reference it.

        :param geometry: The geometry element.
        :param mesh: The mesh shape.
        """
        source = Path(mesh.filename)
        destination = self.assets.folder_for(source, self.mesh_directory) / source.name
        self.assets.copy_mesh(source, destination)
        ElementTree.SubElement(
            geometry,
            UrdfElement.MESH,
            {
                UrdfAttribute.FILENAME: self.assets.relative(destination),
                UrdfAttribute.SCALE: UrdfNumbers.text(mesh.scale.to_np()[:3]),
            },
        )

    # %% joints

    def add_root_joint(self, body: Body, matrix: numpy.ndarray) -> None:
        """
        Fix a body to :data:`ROOT_LINK`.

        :param body: The body.
        :param matrix: The pose it is fixed at.
        """
        joint = ElementTree.SubElement(
            self.root,
            UrdfElement.JOINT,
            {
                UrdfAttribute.NAME: f"{ROOT_LINK}_to_{self.link_names.name_of(body)}",
                UrdfAttribute.TYPE: UrdfJointType.FIXED,
            },
        )
        ElementTree.SubElement(
            joint, UrdfElement.PARENT, {UrdfAttribute.LINK: ROOT_LINK}
        )
        ElementTree.SubElement(
            joint,
            UrdfElement.CHILD,
            {UrdfAttribute.LINK: self.link_names.name_of(body)},
        )
        UrdfNumbers.origin(joint, matrix)

    def add_joint(self, connection: Connection) -> None:
        """
        Add the joint a connection becomes.

        :param connection: The connection between two written bodies.
        :raises UnrepresentableJointLimits: When a limited joint lacks finite limits.
        """
        joint_type = self.joint_type(connection)
        name = str(connection.name)
        joint = ElementTree.SubElement(
            self.root,
            UrdfElement.JOINT,
            {UrdfAttribute.NAME: name, UrdfAttribute.TYPE: joint_type},
        )
        ElementTree.SubElement(
            joint,
            UrdfElement.PARENT,
            {UrdfAttribute.LINK: self.link_names.name_of(connection.parent)},
        )
        ElementTree.SubElement(
            joint,
            UrdfElement.CHILD,
            {UrdfAttribute.LINK: self.link_names.name_of(connection.child)},
        )
        if joint_type in AXIS_JOINT_TYPES:
            # the joint value is applied on top of the origin, so the origin is the
            # pose at value zero
            UrdfNumbers.origin(joint, connection.parent_T_connection_expression.to_np())
            ElementTree.SubElement(
                joint,
                UrdfElement.AXIS,
                {
                    UrdfAttribute.TRANSLATION: UrdfNumbers.text(
                        connection.axis.to_np()[:3]
                    )
                },
            )
            self.movable_joints.append(name)
        else:
            UrdfNumbers.origin(joint, connection.origin.to_np())
        if joint_type in LIMITED_JOINT_TYPES:
            self.add_limit(joint, connection)
        if joint_type is UrdfJointType.FLOATING:
            self.posed_joints.append(name)

    @staticmethod
    def add_limit(joint: ElementTree.Element, connection: ActiveConnection1DOF) -> None:
        """
        :param joint: The joint element.
        :param connection: Its connection.
        :raises UnrepresentableJointLimits: When a position limit is missing or not
            finite.
        """
        limits = connection.dof.limits
        positions = [limits.lower.position, limits.upper.position]
        if any(
            position is None or not math.isfinite(position) for position in positions
        ):
            raise UnrepresentableJointLimits(str(connection.name))
        velocity = limits.upper.velocity
        ElementTree.SubElement(
            joint,
            UrdfElement.LIMIT,
            {
                UrdfAttribute.LOWER: UrdfNumbers.text([positions[0]]),
                UrdfAttribute.UPPER: UrdfNumbers.text([positions[1]]),
                UrdfAttribute.VELOCITY: UrdfNumbers.text(
                    [
                        (
                            velocity
                            if velocity is not None and math.isfinite(velocity)
                            else UNSPECIFIED_LIMIT
                        )
                    ]
                ),
                UrdfAttribute.EFFORT: UrdfNumbers.text([UNSPECIFIED_LIMIT]),
            },
        )

    @staticmethod
    def joint_type(connection: Connection) -> UrdfJointType:
        """
        :param connection: A connection.
        :return: The joint type it becomes; a revolute connection without position
            limits is continuous, a connection URDF has no type for is floating.
        """
        if isinstance(connection, FixedConnection):
            return UrdfJointType.FIXED
        if isinstance(connection, RevoluteConnection):
            if connection.dof.has_position_limits():
                return UrdfJointType.REVOLUTE
            return UrdfJointType.CONTINUOUS
        if isinstance(connection, PrismaticConnection):
            return UrdfJointType.PRISMATIC
        return UrdfJointType.FLOATING


# %% existing URDF files


@dataclass(frozen=True)
class UrdfBundleReport:
    """
    What bundling one existing URDF wrote.
    """

    urdf: Path
    """
    The written URDF.
    """

    links: List[str]
    """
    The names of its links, in document order.
    """

    movable_joints: List[str]
    """
    The names of its joints that are not fixed.
    """

    missing: List[str]
    """
    The references that resolved to no file.
    """


@dataclass(frozen=True)
class UrdfBundle:
    """
    Makes an existing URDF or xacro file self-contained inside a bundle: every mesh it
    references is copied in and the references are rewritten relative to the bundle.
    """

    output_directory: Path
    """
    The bundle directory.
    """

    mesh_directory: str
    """
    The folder below ``meshes`` the copied meshes go to.
    """

    def write(self, source: Path, name: str) -> UrdfBundleReport:
        """
        :param source: The URDF or xacro file.
        :param name: The model name; the URDF is written as ``<name>.urdf``.
        :return: What was written.
        :raises UrdfSourceNotFound: When the source does not exist.
        """
        if not source.is_file():
            raise UrdfSourceNotFound(str(source))
        if source.suffix == XACRO_SUFFIX:
            text = URDFParser.from_xacro(str(source)).urdf
        else:
            text = source.read_text(encoding="utf-8", errors="replace")
        document = ElementTree.fromstring(text)
        assets = BundledAssets(bundle_root=self.output_directory)
        for mesh in document.iter(UrdfElement.MESH):
            reference = mesh.get(UrdfAttribute.FILENAME, "")
            if MeshFormat.of_path(reference) is None:
                continue
            resolved = MeshReference(reference).resolve(source.parent)
            if resolved is None:
                assets.missing.append(reference)
                continue
            destination = (
                assets.folder_for(resolved, self.mesh_directory) / resolved.name
            )
            assets.copy_mesh(resolved, destination)
            mesh.set(UrdfAttribute.FILENAME, assets.relative(destination))
        urdf = self.output_directory / f"{name}{URDF_SUFFIX}"
        assets.require_inside(urdf)
        ElementTree.ElementTree(document).write(
            urdf, encoding="utf-8", xml_declaration=True
        )
        return UrdfBundleReport(
            urdf=urdf,
            links=[
                link.attrib[UrdfAttribute.NAME]
                for link in document.findall(UrdfElement.LINK)
            ],
            movable_joints=[
                joint.attrib[UrdfAttribute.NAME]
                for joint in document.findall(UrdfElement.JOINT)
                if joint.attrib[UrdfAttribute.TYPE] != UrdfJointType.FIXED
            ],
            missing=assets.missing,
        )
