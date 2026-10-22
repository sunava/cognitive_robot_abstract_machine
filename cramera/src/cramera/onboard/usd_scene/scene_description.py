"""
The neutral description of a converted USD stage: links with their meshes, joints,
materials, lights and semantic attributes, in metres and with Z up.

Nothing here imports ``pxr``: the stage reader fills these dataclasses, and every writer
(glTF, URDF, scene bundle) reads only them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import numpy
from scipy.spatial.transform import Rotation
from typing_extensions import Any, Self

# %% fixed vocabularies


class JointKind(StrEnum):
    """
    How a joint lets its child link move, named as URDF names its joint types.
    """

    FIXED = "fixed"
    """
    The child link does not move relative to its parent.
    """

    REVOLUTE = "revolute"
    """
    The child link rotates about the joint axis within limits.
    """

    CONTINUOUS = "continuous"
    """
    The child link rotates about the joint axis without limits.
    """

    PRISMATIC = "prismatic"
    """
    The child link slides along the joint axis within limits.
    """

    @property
    def movable(self) -> bool:
        """
        Whether a joint of this kind has a value that can change.
        """
        return self is not JointKind.FIXED

    @property
    def rotates(self) -> bool:
        """
        Whether the joint value is an angle in radians rather than a distance in metres.
        """
        return self in (JointKind.REVOLUTE, JointKind.CONTINUOUS)


class MaterialSource(StrEnum):
    """
    The shading model a bound material was authored with.
    """

    OMNI_PBR = "OmniPBR"
    """
    NVIDIA's OmniPBR MDL material.
    """

    OMNI_SURFACE = "OmniSurface"
    """
    NVIDIA's OmniSurface MDL material, a variant of Autodesk Standard Surface.
    """

    OMNI_GLASS = "OmniGlass"
    """
    NVIDIA's OmniGlass MDL material.
    """

    PREVIEW_SURFACE = "UsdPreviewSurface"
    """
    The USD preview surface shader.
    """

    DISPLAY_COLOR = "displayColor"
    """
    No material is bound; the geometry's display colour primvar (or the USD fallback
    grey) shades it.
    """

    UNKNOWN = "unknown"
    """
    A shader none of the mappings understands.
    """


class TextureChannel(StrEnum):
    """
    Which channels of a texture file a shader input reads, named as UsdUVTexture names
    its outputs.
    """

    RGB = "rgb"
    """
    The three colour channels.
    """

    RED = "r"
    """
    The red channel only.
    """

    GREEN = "g"
    """
    The green channel only.
    """

    BLUE = "b"
    """
    The blue channel only.
    """

    ALPHA = "a"
    """
    The alpha channel only.
    """


class ColorSpace(StrEnum):
    """
    How a texture's stored values relate to linear light, named as USD names its
    colour spaces.
    """

    AUTO = "auto"
    """
    The reader decides from the image file.
    """

    SRGB = "sRGB"
    """
    The values are sRGB encoded colours.
    """

    RAW = "raw"
    """
    The values are data, such as roughness, used as stored.
    """


class LightKind(StrEnum):
    """
    The UsdLux light types the converter carries into the lighting description.
    """

    DISTANT = "DistantLight"
    """
    A light infinitely far away, such as the sun.
    """

    DOME = "DomeLight"
    """
    Light arriving from every direction of the surrounding sphere.
    """

    RECTANGLE = "RectLight"
    """
    A rectangular area light emitting along its negative Z axis.
    """

    DISK = "DiskLight"
    """
    A disk-shaped area light emitting along its negative Z axis.
    """

    SPHERE = "SphereLight"
    """
    A spherical light.
    """

    CYLINDER = "CylinderLight"
    """
    A cylindrical light.
    """


class SkipReason(StrEnum):
    """
    Why a prim was left out of the converted visuals.
    """

    GUIDE_PURPOSE = "guide purpose"
    """
    The prim is guide geometry, such as a collision proxy.
    """

    PROXY_PURPOSE = "proxy purpose"
    """
    The prim is a lightweight stand-in for render geometry.
    """

    INVISIBLE = "invisible"
    """
    The prim or one of its ancestors is invisible.
    """

    UNSUPPORTED_GEOMETRY = "unsupported geometry type"
    """
    The prim is renderable geometry of a type the converter cannot tessellate.
    """

    EMPTY_GEOMETRY = "empty geometry"
    """
    The prim has no faces.
    """

    LOOP_CLOSING_JOINT = "loop-closing joint"
    """
    The joint would give a link a second parent; URDF describes trees only.
    """

    JOINT_WITHOUT_CHILD = "joint without child body"
    """
    The joint names no body to move.
    """

    JOINT_TO_EXCLUDED_PRIM = "joint to an excluded prim"
    """
    One of the joint's bodies lies in a subtree the conversion leaves out.
    """


class ExclusionReason(StrEnum):
    """
    Why a prim was left out of the conversion together with its subtree.
    """

    REQUESTED = "requested"
    """
    The prim was named to be left out.
    """

    ISAAC_ROBOT_SCHEMA = "Isaac robot schema"
    """
    The prim is a robot: Isaac Sim's robot schema is applied to it.
    """

    ROBOT_ENTITY_TYPE = "robot entity type"
    """
    The prim is a robot: its semantic entity type says so.
    """


class GeneratedName(StrEnum):
    """
    Names and name parts the converter makes up for things the stage does not name.
    """

    WORLD = "world"
    """
    The root link of a stage without a default prim, standing for the world.
    """

    ATTACHMENT = "attachment"
    """
    The fixed joint holding a link without an authored joint at its authored pose.
    """


class SemanticNamespace(StrEnum):
    """
    The attribute namespaces carried into the semantics sidecar, by which a prim entry
    of ``semantics.json`` groups its attributes.
    """

    SEMANTIC_DIGITAL_TWIN = "sdt"
    """
    The twin's own semantic attributes.
    """

    ONTOLOGY = "ont"
    """
    Ontology types and IRIs.
    """

    def attribute_name(self, base_name: str) -> str:
        """
        The full name of an attribute in this namespace, such as ``sdt:entityType``.

        :param base_name: The attribute name without its namespace.
        """
        return f"{self.value}:{base_name}"


class SemanticAttributeName(StrEnum):
    """
    The ``sdt`` attributes known by name, without their namespace.
    """

    CLASS = "class"
    """
    The kind of object the prim is.
    """

    ENTITY_TYPE = "entityType"
    """
    The kind of entity the prim stands for, for prims that are no physical object kind.
    """

    CANONICAL_NAME = "canonicalName"
    """
    The human-readable name of the prim's object.
    """

    COMPONENT_ID = "componentId"
    """
    The identifier of the component, such as a station number.
    """

    CAPABILITIES = "capabilities"
    """
    The skills the prim's object offers.
    """


class OntologyAttributeName(StrEnum):
    """
    The ``ont`` attributes known by name, without their namespace.
    """

    TYPES = "types"
    """
    The ontology classes asserted for the prim.
    """

    INFERRED_TYPES = "inferredTypes"
    """
    The ontology classes a reasoner inferred for the prim.
    """


# %% names


@dataclass
class UniqueNames:
    """
    Names unique among each other: a name asked for again gets a number appended,
    starting from two.
    """

    used: set[str] = field(default_factory=set)
    """
    The names handed out so far.
    """

    def claim(self, wanted: str) -> str:
        """
        The wanted name, numbered when it is taken already.

        :param wanted: The name asked for.
        """
        name = wanted
        suffix = 2
        while name in self.used:
            name = f"{wanted}_{suffix}"
            suffix += 1
        self.used.add(name)
        return name

    def claim_each(self, wanted: dict[str, str]) -> dict[str, str]:
        """
        A unique name for every key, in the order of the keys.

        :param wanted: The name asked for per key.
        """
        return {key: self.claim(name) for key, name in wanted.items()}


# %% rigid poses


@dataclass(frozen=True, eq=False)
class Pose:
    """
    A rigid transform (rotation and translation, no scale) as a 4x4 matrix acting on
    column vectors.
    """

    matrix: numpy.ndarray = field(default_factory=lambda: numpy.eye(4))
    """
    The homogeneous 4x4 matrix.
    """

    @classmethod
    def rigid_part(cls, matrix: numpy.ndarray) -> Self:
        """
        The rotation and translation of an affine matrix, with scale and shear removed.

        :param matrix: A homogeneous 4x4 matrix acting on column vectors.
        """
        return cls(
            PoseTrack.rigid_part(numpy.asarray(matrix, dtype=float)[None]).matrices[0]
        )

    @classmethod
    def from_translation_quaternion(
        cls, translation: numpy.ndarray, quaternion: numpy.ndarray
    ) -> Self:
        """
        A pose from a translation and a unit quaternion.

        :param translation: The translation in metres.
        :param quaternion: The rotation as ``[x, y, z, w]``.
        """
        result = numpy.eye(4)
        result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        result[:3, 3] = translation
        return cls(result)

    @classmethod
    def from_rotation(cls, rotation: numpy.ndarray) -> Self:
        """
        A pure rotation.

        :param rotation: A 3x3 rotation matrix.
        """
        result = numpy.eye(4)
        result[:3, :3] = rotation
        return cls(result)

    @classmethod
    def from_translation(cls, translation: numpy.ndarray) -> Self:
        """
        A pure translation.

        :param translation: The translation in metres.
        """
        result = numpy.eye(4)
        result[:3, 3] = translation
        return cls(result)

    @property
    def translation(self) -> numpy.ndarray:
        """
        The translation in metres.
        """
        return self.matrix[:3, 3].copy()

    @property
    def rotation(self) -> numpy.ndarray:
        """
        The 3x3 rotation matrix.
        """
        return self.matrix[:3, :3].copy()

    @property
    def quaternion(self) -> numpy.ndarray:
        """
        The rotation as a unit quaternion ``[x, y, z, w]``.
        """
        return Rotation.from_matrix(self.rotation).as_quat()

    @property
    def roll_pitch_yaw(self) -> numpy.ndarray:
        """
        The rotation as fixed-axis roll, pitch and yaw in radians, as URDF origins use.
        """
        return Rotation.from_matrix(self.rotation).as_euler("xyz")

    def translation_quaternion(self) -> list[float]:
        """
        The pose as ``[x, y, z, qx, qy, qz, qw]``, the layout scene files use.
        """
        return PoseTrack.constant(self, 1).translation_quaternions()[0]

    def inverse(self) -> Pose:
        """
        The pose undoing this one.
        """
        return PoseTrack.constant(self, 1).inverse().pose_at(0)

    def __matmul__(self, other: Pose) -> Pose:
        return Pose(self.matrix @ other.matrix)

    def is_close(self, other: Pose, tolerance: float = 1e-6) -> bool:
        """
        Whether two poses agree up to a tolerance on every matrix entry.

        :param other: The pose to compare with.
        :param tolerance: The largest allowed difference of a matrix entry.
        """
        return bool(numpy.allclose(self.matrix, other.matrix, atol=tolerance))


@dataclass(frozen=True, eq=False)
class PoseTrack:
    """
    One rigid pose per frame, as ``(frames, 4, 4)`` matrices acting on column vectors.
    """

    matrices: numpy.ndarray
    """
    The homogeneous matrix of every frame.
    """

    @classmethod
    def constant(cls, pose: Pose, frame_count: int) -> Self:
        """
        The same pose at every frame.

        :param pose: The pose.
        :param frame_count: The number of frames.
        """
        return cls(numpy.broadcast_to(pose.matrix, (frame_count, 4, 4)))

    @classmethod
    def of_parts(cls, rotations: numpy.ndarray, translations: numpy.ndarray) -> Self:
        """
        The poses made of a rotation and a translation per frame.

        :param rotations: ``(frames, 3, 3)`` rotation matrices.
        :param translations: ``(frames, 3)`` translations in metres.
        """
        matrices = numpy.zeros((len(translations), 4, 4))
        matrices[:, :3, :3] = rotations
        matrices[:, :3, 3] = translations
        matrices[:, 3, 3] = 1.0
        return cls(matrices)

    @classmethod
    def of_positions(cls, positions: numpy.ndarray, quaternions: numpy.ndarray) -> Self:
        """
        The poses at positions with orientations.

        :param positions: ``(frames, 3)`` positions in metres.
        :param quaternions: ``(frames, 4)`` orientations as ``[x, y, z, w]``, which
            need not be exactly unit length.
        """
        return cls.of_parts(Rotation.from_quat(quaternions).as_matrix(), positions)

    @classmethod
    def rigid_part(cls, matrices: numpy.ndarray) -> Self:
        """
        The rotations and translations of affine matrices, with scale and shear
        removed; a mirroring matrix keeps the proper rotation nearest to it.

        :param matrices: ``(frames, 4, 4)`` matrices acting on column vectors.
        """
        left, _, right = numpy.linalg.svd(matrices[:, :3, :3])
        mirrored = numpy.linalg.det(left @ right) < 0
        left[mirrored, :, -1] *= -1
        return cls.of_parts(left @ right, matrices[:, :3, 3])

    def __matmul__(self, other: PoseTrack | Pose) -> PoseTrack:
        if isinstance(other, PoseTrack):
            return PoseTrack(self.matrices @ other.matrices)
        return PoseTrack(self.matrices @ other.matrix)

    @property
    def frame_count(self) -> int:
        """
        The number of frames.
        """
        return len(self.matrices)

    @property
    def rotations(self) -> numpy.ndarray:
        """
        The ``(frames, 3, 3)`` rotation matrices.
        """
        return self.matrices[:, :3, :3]

    @property
    def translations(self) -> numpy.ndarray:
        """
        The ``(frames, 3)`` translations in metres.
        """
        return self.matrices[:, :3, 3]

    def inverse(self) -> PoseTrack:
        """
        The poses undoing these, frame by frame.
        """
        rotations = numpy.swapaxes(self.rotations, 1, 2)
        return PoseTrack.of_parts(
            rotations, -(rotations @ self.translations[:, :, None])[:, :, 0]
        )

    def at(self, frames: numpy.ndarray) -> PoseTrack:
        """
        The poses of some frames.

        :param frames: The frame indices.
        """
        return PoseTrack(self.matrices[frames])

    def pose_at(self, frame: int) -> Pose:
        """
        The pose of one frame.

        :param frame: The frame index.
        """
        return Pose(numpy.array(self.matrices[frame]))

    def translation_quaternions(self, decimals: int | None = None) -> list[list[float]]:
        """
        Every pose as ``[x, y, z, qx, qy, qz, qw]``, the layout scene files use.

        :param decimals: The number of decimals kept, or None to keep them all.
        """
        quaternions = Rotation.from_matrix(self.rotations).as_quat()
        values = numpy.concatenate([self.translations, quaternions], axis=1)
        if decimals is not None:
            values = numpy.round(values, decimals)
        return (values + 0.0).tolist()


# %% bounds


@dataclass(frozen=True, eq=False)
class Bounds:
    """
    An axis-aligned box in metres.
    """

    minimum: numpy.ndarray
    """
    The corner with the smallest coordinates.
    """

    maximum: numpy.ndarray
    """
    The corner with the largest coordinates.
    """

    @classmethod
    def of_points(cls, points: numpy.ndarray) -> Self:
        """
        The smallest box holding every point.

        :param points: An ``(N, 3)`` array of positions.
        """
        return cls(points.min(axis=0), points.max(axis=0))

    @classmethod
    def central(cls, points: numpy.ndarray, fraction: float) -> Self:
        """
        The box holding the central fraction of the points along every axis, which
        leaves out sparse outliers such as a building shell around a dense interior.

        :param points: An ``(N, 3)`` array of positions.
        :param fraction: The share of points kept per axis, between 0 and 1.
        """
        cut = (1.0 - fraction) / 2.0 * 100.0
        return cls(
            numpy.percentile(points, cut, axis=0),
            numpy.percentile(points, 100.0 - cut, axis=0),
        )

    @property
    def center(self) -> numpy.ndarray:
        """
        The middle of the box.
        """
        return (self.minimum + self.maximum) / 2.0

    @property
    def diagonal(self) -> float:
        """
        The length of the box's diagonal.
        """
        return float(numpy.linalg.norm(self.maximum - self.minimum))

    def to_json(self) -> list[list[float]]:
        """
        The box as ``[[min x, min y, min z], [max x, max y, max z]]``.
        """
        return [
            [round(float(value), 4) for value in self.minimum],
            [round(float(value), 4) for value in self.maximum],
        ]


# %% materials


@dataclass(frozen=True)
class TextureFile:
    """
    One texture image a shader input reads.
    """

    path: Path
    """
    The resolved image file.
    """

    color_space: ColorSpace = ColorSpace.AUTO
    """
    The colour space the stage declares for the image.
    """

    channel: TextureChannel = TextureChannel.RGB
    """
    The channels the shader input reads.
    """


ShaderInputValue = bool | int | float | str | tuple[float, ...] | TextureFile
"""
The value of one authored shader input: a number, a flag, a token, a colour or vector,
or a texture file.
"""


@dataclass
class ImportedMaterial:
    """
    A material bound on the stage, as the authored inputs of its surface shader.
    """

    name: str
    """
    The material's USD path, or a generated name for display-colour shading.
    """

    source: MaterialSource
    """
    The shading model the inputs belong to.
    """

    inputs: dict[str, ShaderInputValue] = field(default_factory=dict)
    """
    The authored shader inputs by name, without the ``inputs:`` namespace.
    """

    shader_identifier: str = ""
    """
    The shader's identifier on the stage (MDL sub-identifier or ``info:id``), kept for
    the report when the source is unknown.
    """


# %% geometry


@dataclass(eq=False)
class ImportedMesh:
    """
    The triangles of one geometry prim (or one material subset of it), in the frame
    of the link it belongs to.
    """

    path: str
    """
    The USD path of the geometry prim.
    """

    points: numpy.ndarray
    """
    ``(N, 3)`` float32 vertex positions in metres, in the link frame.
    """

    triangles: numpy.ndarray
    """
    ``(M, 3)`` uint32 vertex indices, counter-clockwise when seen from outside.
    """

    material: str
    """
    The name of the material in :attr:`SceneDescription.materials`.
    """

    normals: numpy.ndarray | None = None
    """
    ``(N, 3)`` float32 unit vertex normals in the link frame, or None for flat shading.
    """

    texture_coordinates: numpy.ndarray | None = None
    """
    ``(N, 2)`` float32 texture coordinates as USD authors them (origin bottom left).
    """

    double_sided: bool = False
    """
    Whether both sides of every triangle are visible.
    """


@dataclass(eq=False)
class ImportedLink:
    """
    One rigid part of the converted scene: a rigid body, a jointed part, or the
    static geometry of one asset.
    """

    name: str
    """
    The unique link name, derived from :attr:`path`.
    """

    path: str
    """
    The USD path of the prim the link stands for.
    """

    pose: Pose
    """
    The link frame in the world at the stage's initial time.
    """

    prim_to_link: Pose = field(default_factory=Pose)
    """
    The link frame relative to the rigid part of the prim's own frame, which maps a
    world pose of the prim (from a recorded trace, for example) to the link pose.
    """

    meshes: list[ImportedMesh] = field(default_factory=list)
    """
    The visual geometry, in the link frame.
    """

    free: bool = False
    """
    Whether the link is a rigid body that no joint connects to the world; it is
    attached at its authored pose by a fixed joint.
    """

    @property
    def triangle_count(self) -> int:
        """
        The number of triangles over all meshes of the link.
        """
        return sum(len(mesh.triangles) for mesh in self.meshes)


# %% joints


@dataclass(frozen=True, eq=False)
class AllowedMotion:
    """
    How a joint lets its child link move: the kind of motion, its axis and its limits.
    """

    kind: JointKind
    """
    The kind of motion.
    """

    axis: numpy.ndarray = field(default_factory=lambda: numpy.array([1.0, 0.0, 0.0]))
    """
    The unit motion axis in the joint frame.
    """

    lower: float | None = None
    """
    The lowest joint value (radians or metres), or None when unlimited.
    """

    upper: float | None = None
    """
    The highest joint value (radians or metres), or None when unlimited.
    """

    def reversed(self) -> AllowedMotion:
        """
        The same motion about the opposite axis: measuring the first joint frame
        relative to the second along it gives the values, and so keeps the limits, of
        measuring the second relative to the first along the original axis.
        """
        return replace(self, axis=-self.axis)


@dataclass(eq=False)
class ImportedJoint:
    """
    A joint between two links, in URDF terms: the child link frame is the parent
    link frame, then :attr:`origin`, then the joint motion about or along :attr:`axis`.
    """

    name: str
    """
    The unique joint name, derived from the joint prim's path.
    """

    path: str
    """
    The USD path of the joint prim, or of the attached link for generated joints.
    """

    kind: JointKind
    """
    How the child link moves.
    """

    parent: str
    """
    The name of the parent link.
    """

    child: str
    """
    The name of the child link.
    """

    origin: Pose
    """
    The joint frame in the parent link frame, at joint value zero.
    """

    axis: numpy.ndarray = field(default_factory=lambda: numpy.array([1.0, 0.0, 0.0]))
    """
    The unit motion axis in the joint frame.
    """

    lower: float | None = None
    """
    The lowest joint value (radians or metres), or None when unlimited.
    """

    upper: float | None = None
    """
    The highest joint value (radians or metres), or None when unlimited.
    """

    initial_value: float = 0.0
    """
    The joint value recovered from the authored body poses.
    """

    enabled: bool = True
    """
    Whether the joint takes part in the simulation (``physics:jointEnabled``); a
    disabled joint still describes the mechanism but yields to an enabled joint of the
    same child body.
    """

    parent_body: str | None = None
    """
    The USD path of the body the child body moves relative to (``physics:body0``), or
    None for the world.
    """

    child_body: str | None = None
    """
    The USD path of the body the joint moves (``physics:body1``, or ``physics:body0``
    when the joint names no other body), or None for generated joints.
    """

    parent_body_to_joint: Pose = field(default_factory=Pose)
    """
    The joint frame relative to the rigid frame of the parent body (or the world).
    """

    child_body_to_joint: Pose = field(default_factory=Pose)
    """
    The joint frame relative to the rigid frame of the child body.
    """

    def relative_motion(
        self, parent_body: PoseTrack, child_body: PoseTrack
    ) -> PoseTrack:
        """
        The child joint frame relative to the parent joint frame, per frame.

        :param parent_body: The world poses of the parent body (identity for the world).
        :param child_body: The world poses of the child body.
        """
        return (parent_body @ self.parent_body_to_joint).inverse() @ (
            child_body @ self.child_body_to_joint
        )

    def values_of(self, relative: PoseTrack) -> numpy.ndarray:
        """
        The joint values best explaining the child joint frame relative to the parent
        joint frame, per frame: the distance along, or the angle about, the axis. An
        angle is taken within the joint's limits when they span less than a full turn,
        else without jumps between frames; a fixed joint stays at zero.

        :param relative: The child joint frame in the parent joint frame per frame.
        """
        if not self.kind.movable:
            return numpy.zeros(relative.frame_count)
        if not self.kind.rotates:
            return relative.translations @ self.axis
        perpendicular = numpy.cross(
            self.axis, numpy.eye(3)[numpy.argmin(abs(self.axis))]
        )
        perpendicular /= numpy.linalg.norm(perpendicular)
        turned = relative.rotations @ perpendicular
        angles = numpy.arctan2(
            numpy.cross(perpendicular, turned) @ self.axis, turned @ perpendicular
        )
        if (
            self.kind is JointKind.CONTINUOUS
            or self.lower is None
            or self.upper is None
            or self.upper - self.lower >= 2.0 * numpy.pi
        ):
            return numpy.unwrap(angles)
        middle = (self.lower + self.upper) / 2.0
        return angles - 2.0 * numpy.pi * numpy.round(
            (angles - middle) / (2.0 * numpy.pi)
        )

    def motions(self, values: numpy.ndarray) -> PoseTrack:
        """
        The child joint frame relative to the parent joint frame at joint values.

        :param values: The joint value per frame, in radians or metres.
        """
        frame_count = len(values)
        if not self.kind.movable:
            return PoseTrack.constant(Pose(), frame_count)
        if self.kind.rotates:
            return PoseTrack.of_parts(
                Rotation.from_rotvec(numpy.outer(values, self.axis)).as_matrix(),
                numpy.zeros((frame_count, 3)),
            )
        return PoseTrack.of_parts(
            numpy.broadcast_to(numpy.eye(3), (frame_count, 3, 3)),
            numpy.outer(values, self.axis),
        )

    def motion(self, value: float) -> Pose:
        """
        The child joint frame relative to the parent joint frame at one joint value.

        :param value: The joint value in radians or metres.
        """
        return self.motions(numpy.array([value], dtype=float)).pose_at(0)

    def value_between(self, parent_body_pose: Pose, child_body_pose: Pose) -> float:
        """
        The joint value that best explains two world poses of the joint's bodies, as
        :meth:`values_of` takes it.

        :param parent_body_pose: The world pose of the parent body (identity for the
            world).
        :param child_body_pose: The world pose of the child body.
        """
        relative = self.relative_motion(
            PoseTrack.constant(parent_body_pose, 1),
            PoseTrack.constant(child_body_pose, 1),
        )
        return float(self.values_of(relative)[0])


# %% lights and semantics


@dataclass(eq=False)
class ImportedLight:
    """
    A UsdLux light, in metres with Z up.
    """

    path: str
    """
    The USD path of the light prim.
    """

    kind: LightKind
    """
    The light type.
    """

    pose: Pose
    """
    The light frame in the world; area lights emit along its negative Z axis.
    """

    intensity: float
    """
    The authored intensity scaled by two to the power of the authored exposure.
    """

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """
    The authored linear light colour.
    """

    color_temperature: float | None = None
    """
    The colour temperature in kelvin, or None when the light does not use one.
    """

    width: float = 0.0
    """
    The width of a rectangle light in metres.
    """

    height: float = 0.0
    """
    The height of a rectangle light in metres.
    """

    radius: float = 0.0
    """
    The radius of a disk, sphere or cylinder light in metres.
    """

    angle: float = 0.0
    """
    The angular diameter of a distant light in degrees.
    """

    @property
    def emission_direction(self) -> numpy.ndarray:
        """
        The direction the light travels in, the negative Z axis of its frame.
        """
        return -self.pose.rotation[:, 2]


@dataclass(eq=False)
class ImportedCamera:
    """
    A camera authored on the stage.
    """

    path: str
    """
    The USD path of the camera prim.
    """

    pose: Pose
    """
    The camera frame in the world; the camera looks along its negative Z axis with
    its Y axis up.
    """

    vertical_field_of_view: float
    """
    The vertical field of view in degrees, from the focal length and the vertical
    aperture.
    """

    @property
    def forward(self) -> numpy.ndarray:
        """
        The viewing direction.
        """
        return -self.pose.rotation[:, 2]


@dataclass
class PrimSemantics:
    """
    The semantic attributes of one prim, grouped by attribute namespace.
    """

    path: str
    """
    The USD path of the prim.
    """

    attributes: dict[str, dict[str, Any]]
    """
    Namespace (such as ``sdt``) to attribute name (without namespace) to JSON value.
    """

    link: str = ""
    """
    The name of the link the prim belongs to.
    """


# %% report


@dataclass(frozen=True)
class ExcludedPrim:
    """
    A prim the conversion left out together with its subtree.
    """

    path: str
    """
    The USD path of the prim.
    """

    reason: ExclusionReason
    """
    Why it was left out.
    """


@dataclass
class DetachedBody:
    """
    An environment body whose joint to a parent body the conversion leaves out was
    dropped: it stays at its authored pose, held by the world instead.
    """

    body: str
    """
    The USD path of the environment body.
    """

    parent_body: str
    """
    The USD path of the left-out body it hung from.
    """

    joint: str
    """
    The USD path of the dropped joint.
    """

    movable_joints: list[str] = field(default_factory=list)
    """
    The USD paths of the converted movable joints that move parts hanging from the
    body; a body with any is more likely a part of the robot than an object it held.
    """


@dataclass
class EnvironmentArticulation:
    """
    An articulation root converted as part of the environment, outside every robot the
    stage marks: a door or station, or a robot the stage does not mark.
    """

    path: str
    """
    The USD path of the prim carrying the articulation root.
    """

    movable_joints: list[str] = field(default_factory=list)
    """
    The USD paths of the converted movable joints of the articulation.
    """


@dataclass
class JointGraph:
    """
    The joints read from a stage, as edges from their parent body to their child body.
    """

    joints: list[ImportedJoint]
    """
    The joints, in stage order.
    """

    def movable_joints_below(self, bodies: set[str]) -> list[str]:
        """
        The USD paths of the movable joints that move one of some bodies or a body
        hanging from them, in stage order.

        :param bodies: The USD paths of the bodies.
        """
        reached = set(bodies)
        grown = True
        while grown:
            below = {
                joint.child_body
                for joint in self.joints
                if joint.parent_body in reached and joint.child_body not in reached
            }
            reached |= below
            grown = bool(below)
        return [
            joint.path
            for joint in self.joints
            if joint.kind.movable and joint.child_body in reached
        ]

    def bodies_in(self, path: str) -> set[str]:
        """
        The USD paths of the joint bodies that are a prim or lie below it.

        :param path: The USD path of the prim.
        """
        ends = {
            body
            for joint in self.joints
            for body in (joint.parent_body, joint.child_body)
            if body is not None
        }
        return {body for body in ends if body == path or body.startswith(path + "/")}


class ReportCount(StrEnum):
    """
    The named counts of a conversion report.
    """

    PRIMS = "prims"
    """
    Prims traversed.
    """

    INSTANCE_PROXIES = "instanceProxies"
    """
    Traversed prims that are instance proxies.
    """

    LINKS = "links"
    """
    Links written.
    """

    MESHES = "meshes"
    """
    Meshes written, one per geometry prim and material.
    """

    TRIANGLES = "triangles"
    """
    Triangles written.
    """

    MATERIALS = "materials"
    """
    Distinct materials bound.
    """

    LIGHTS = "lights"
    """
    Lights read.
    """

    SEMANTIC_PRIMS = "semanticPrims"
    """
    Prims carrying semantic attributes.
    """

    JOINTS = "joints"
    """
    Joints written, counted per kind as ``joints.<kind>``.
    """

    DISABLED_JOINTS = "disabledJoints"
    """
    Joints written although disabled for the simulation.
    """

    def of_kind(self, kind: JointKind) -> str:
        """
        The count key of one kind, such as ``joints.revolute``.

        :param kind: The joint kind.
        """
        return f"{self.value}.{kind.value}"


@dataclass
class ConversionReport:
    """
    What the conversion read, wrote and left out.
    """

    counts: Counter = field(default_factory=Counter)
    """
    Named counts of what was read and written.
    """

    skipped: Counter = field(default_factory=Counter)
    """
    The number of prims left out, by :class:`SkipReason`.
    """

    skipped_examples: dict[str, list[str]] = field(default_factory=dict)
    """
    A few prim paths per skip reason.
    """

    excluded: list[ExcludedPrim] = field(default_factory=list)
    """
    The prims left out with their subtrees, in traversal order.
    """

    detached: list[DetachedBody] = field(default_factory=list)
    """
    The environment bodies whose joint to a left-out parent body was dropped, in joint
    path order.
    """

    articulations: list[EnvironmentArticulation] = field(default_factory=list)
    """
    The articulations with movable joints converted outside every marked robot, in
    traversal order.
    """

    unmapped_inputs: dict[str, list[str]] = field(default_factory=dict)
    """
    Material name to the authored shader inputs no mapping carried over.
    """

    missing_textures: list[str] = field(default_factory=list)
    """
    Texture asset paths that did not resolve to a file.
    """

    warnings: list[str] = field(default_factory=list)
    """
    Everything else worth a look.
    """

    example_limit: int = 5
    """
    How many example paths are kept per skip reason.
    """

    def skip(self, reason: SkipReason, path: str) -> None:
        """
        Record that a prim was left out.

        :param reason: Why it was left out.
        :param path: The prim's USD path.
        """
        self.skipped[reason.value] += 1
        examples = self.skipped_examples.setdefault(reason.value, [])
        if len(examples) < self.example_limit:
            examples.append(path)


# %% scene


@dataclass(eq=False)
class SceneDescription:
    """
    Everything the bundle writers need from a converted stage.
    """

    source: Path
    """
    The stage file that was converted.
    """

    root_link: str
    """
    The name of the link every other link hangs from, standing for the world.
    """

    links: list[ImportedLink] = field(default_factory=list)
    """
    Every link, parents before children.
    """

    joints: list[ImportedJoint] = field(default_factory=list)
    """
    One joint per link except the root.
    """

    materials: dict[str, ImportedMaterial] = field(default_factory=dict)
    """
    The materials meshes refer to, by name.
    """

    lights: list[ImportedLight] = field(default_factory=list)
    """
    The lights of the selected prims.
    """

    semantics: list[PrimSemantics] = field(default_factory=list)
    """
    The prims carrying semantic attributes.
    """

    cameras: list[ImportedCamera] = field(default_factory=list)
    """
    The cameras of the selected prims, in traversal order.
    """

    bounds: Bounds | None = None
    """
    The box holding all visual geometry, or None for a scene without any.
    """

    central_bounds: Bounds | None = None
    """
    The box holding the dense core of the visual geometry, or None without any.
    """

    used_files: list[Path] = field(default_factory=list)
    """
    The layers and textures the conversion read, for change detection.
    """

    report: ConversionReport = field(default_factory=ConversionReport)
    """
    What the conversion read, wrote and left out.
    """

    @property
    def movable_joints(self) -> list[ImportedJoint]:
        """
        The joints whose value can change, in tree order.
        """
        return [joint for joint in self.joints if joint.kind.movable]
