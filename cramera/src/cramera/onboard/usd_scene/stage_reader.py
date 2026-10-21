"""
Reading a composed USD stage into the neutral :class:`SceneDescription`: the only
module of the converter that imports ``pxr``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from typing_extensions import Any

from cramera.onboard.usd_scene.animation import BodyTrace
from cramera.onboard.usd_scene.exceptions import (
    PrimNotFound,
    RobotExcluded,
    SelectedPrimLeftOut,
    StageNotFound,
    StageUnreadable,
    VariantNotFound,
)
from cramera.onboard.usd_scene.scene_description import (
    AllowedMotion,
    Bounds,
    ColorSpace,
    ConversionReport,
    DetachedBody,
    EnvironmentArticulation,
    ExcludedPrim,
    ExclusionReason,
    GeneratedName,
    ImportedCamera,
    ImportedJoint,
    ImportedLight,
    ImportedLink,
    ImportedMaterial,
    ImportedMesh,
    JointGraph,
    JointKind,
    LightKind,
    MaterialSource,
    Pose,
    PoseTrack,
    PrimSemantics,
    ReportCount,
    SceneDescription,
    SemanticAttributeName,
    SemanticNamespace,
    ShaderInputValue,
    SkipReason,
    TextureChannel,
    TextureFile,
    UniqueNames,
)
from cramera.onboard.usd_scene.tessellation import (
    Axis,
    Interpolation,
    PolygonMesh,
    PrimvarData,
    ShapeTessellation,
    TriangleMesh,
)

# %% stage vocabulary


class Purpose(StrEnum):
    """
    The imageable purposes that are not rendered.
    """

    GUIDE = "guide"
    """
    Guide geometry such as collision proxies.
    """

    PROXY = "proxy"
    """
    Lightweight stand-ins for render geometry.
    """


class PrimvarName(StrEnum):
    """
    The geometry primvars the reader looks up by name.
    """

    TEXTURE_COORDINATES = "st"
    """
    The conventional texture coordinate set.
    """

    NORMALS = "normals"
    """
    Normals authored as a primvar, which take precedence over the normals attribute.
    """

    DISPLAY_COLOR = "displayColor"
    """
    The colour of geometry without a material.
    """

    DISPLAY_OPACITY = "displayOpacity"
    """
    The opacity of geometry without a material.
    """


class ShaderIdentifier(StrEnum):
    """
    Shader identifiers the reader recognises in ``info:id``.
    """

    PREVIEW_SURFACE = "UsdPreviewSurface"
    """
    The USD preview surface.
    """

    TEXTURE = "UsdUVTexture"
    """
    The USD texture reader feeding a preview surface input.
    """


class RenderContext(StrEnum):
    """
    The material render contexts the reader asks for a surface shader.
    """

    MATERIAL_DEFINITION_LANGUAGE = "mdl"
    """
    NVIDIA MDL, which Isaac Sim's Omni materials use.
    """


class TextureInput(StrEnum):
    """
    The UsdUVTexture inputs the reader follows.
    """

    FILE = "file"
    """
    The image file.
    """

    COLOR_SPACE = "sourceColorSpace"
    """
    The declared colour space of the image.
    """


class RobotMarker(StrEnum):
    """
    How a stage marks the root prim of a robot, spelled as the stage spells it.
    """

    ISAAC_ROBOT_SCHEMA = "IsaacRobotAPI"
    """
    The API schema of Isaac Sim's robot schema (``isaacsim.robot.schema``), applied to
    the root prim.
    """

    ROBOT_ENTITY_TYPE = "Robot"
    """
    The semantic entity type (:attr:`SemanticAttributeName.ENTITY_TYPE`) of a robot's
    root prim; only this exact value marks a robot.
    """


# %% stage frame


@dataclass
class StageFrame:
    """
    The composed stage at its initial time, with the change into metres and Z up.
    """

    stage: Usd.Stage
    """
    The composed stage.
    """

    time: Usd.TimeCode
    """
    The time the stage is read at: its start time when it has a time range.
    """

    correction: numpy.ndarray
    """
    The 4x4 matrix taking stage coordinates into metres with Z up.
    """

    meters_per_unit: float
    """
    The stage's length unit in metres.
    """

    transforms: UsdGeom.XformCache
    """
    The cache of world transforms at :attr:`time`.
    """

    @classmethod
    def of_stage(cls, stage: Usd.Stage) -> StageFrame:
        """
        The frame of a composed stage.

        :param stage: The stage.
        """
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        correction = numpy.eye(4) * meters_per_unit
        correction[3, 3] = 1.0
        if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
            correction[:3, :3] = (
                numpy.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
                * meters_per_unit
            )
        time = (
            Usd.TimeCode(stage.GetStartTimeCode())
            if stage.HasAuthoredTimeCodeRange()
            else Usd.TimeCode.Default()
        )
        return cls(
            stage=stage,
            time=time,
            correction=correction,
            meters_per_unit=meters_per_unit,
            transforms=UsdGeom.XformCache(time),
        )

    def world_matrix(self, prim: Usd.Prim) -> numpy.ndarray:
        """
        A prim's local-to-world matrix in metres with Z up, acting on column vectors.

        :param prim: The prim.
        """
        local_to_world = numpy.array(self.transforms.GetLocalToWorldTransform(prim)).T
        return self.correction @ local_to_world

    def world_pose(self, prim: Usd.Prim) -> Pose:
        """
        The rigid part of a prim's world transform.

        :param prim: The prim.
        """
        return Pose.rigid_part(self.world_matrix(prim))


# %% names


@dataclass
class NameRegistry:
    """
    Unique, path-derived names: the default prim's prefix is dropped, path separators
    become double underscores, and a repeated name gets a numeric suffix.
    """

    default_prim: str
    """
    The path of the stage's default prim, or an empty string without one.
    """

    names: UniqueNames = field(default_factory=UniqueNames)
    """
    The names handed out so far.
    """

    by_path: dict[str, str] = field(default_factory=dict)
    """
    The name handed out per path.
    """

    def name_of(self, path: str) -> str:
        """
        The unique name of a prim path; the same path always gets the same name.

        :param path: The absolute prim path.
        """
        if path in self.by_path:
            return self.by_path[path]
        relative = path
        if self.default_prim and path.startswith(self.default_prim + "/"):
            relative = path[len(self.default_prim) + 1 :]
        name = self.names.claim(
            relative.strip("/").replace("/", "__") or GeneratedName.WORLD.value
        )
        self.by_path[path] = name
        return name


# %% links


@dataclass
class LinkPartition:
    """
    Decides which prim each piece of geometry belongs to: the nearest rigid body,
    joint body or referenced asset above it, else the outermost prim whose subtree
    holds no other link.
    """

    root_path: str
    """
    The prim standing for the world (the default prim).
    """

    bodies: set[str]
    """
    Rigid bodies and joint bodies; each is a link.
    """

    assets: set[str]
    """
    Outermost referenced or component-kind prims; each holding geometry is a link.
    """

    occupied: set[str] = field(default_factory=set)
    """
    The strict ancestors of every body and asset.
    """

    def __post_init__(self) -> None:
        for path in self.bodies | self.assets:
            parent = Sdf.Path(path).GetParentPath()
            while not parent.IsAbsoluteRootPath() and not parent.isEmpty:
                self.occupied.add(parent.pathString)
                parent = parent.GetParentPath()

    def link_path_of(self, path: str) -> str:
        """
        The path of the prim whose link holds a prim.

        :param path: The prim path.
        """
        current = Sdf.Path(path)
        while not current.IsAbsoluteRootPath():
            if current.pathString in self.bodies or current.pathString in self.assets:
                return current.pathString
            current = current.GetParentPath()
        root = Sdf.Path(self.root_path)
        for prefix in Sdf.Path(path).GetPrefixes():
            if not root.HasPrefix(prefix) and prefix.pathString not in self.occupied:
                return prefix.pathString
        return self.root_path


@dataclass
class LinkRecord:
    """
    A link while the kinematic tree is being built.
    """

    path: str
    """
    The prim the link stands for.
    """

    name: str
    """
    The unique link name.
    """

    prim_pose: Pose
    """
    The rigid world pose of the prim.
    """

    pose: Pose
    """
    The link frame in the world.
    """

    parent: str | None = None
    """
    The path of the parent link, None for the root and before attachment.
    """

    joint: ImportedJoint | None = None
    """
    The joint connecting the link to its parent.
    """

    joint_frame: Pose | None = None
    """
    The world pose of a movable joint's frame at joint value zero.
    """

    is_body: bool = False
    """
    Whether the prim is a rigid or joint body.
    """

    meshes: list[ImportedMesh] = field(default_factory=list)
    """
    The visual geometry in the link frame.
    """


# %% materials


@dataclass
class MaterialReader:
    """
    Reads the surface shader of bound materials into :class:`ImportedMaterial`.
    """

    frame: StageFrame
    """
    The stage frame, for the read time.
    """

    report: ConversionReport
    """
    The report collecting unresolved textures.
    """

    materials: dict[str, ImportedMaterial] = field(default_factory=dict)
    """
    The materials read so far, by name.
    """

    def material_of(self, prim: Usd.Prim) -> str:
        """
        The name of the material shading a prim (or a material subset), reading it on
        first use.

        :param prim: A geometry prim or a geometry subset prim.
        """
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            return self.display_color_material(prim)
        name = material.GetPath().pathString
        if name not in self.materials:
            self.materials[name] = self.read(material)
        return name

    def display_color_material(self, prim: Usd.Prim) -> str:
        """
        The material standing for a prim's display colour and opacity.

        :param prim: A geometry prim (or subset, whose parent carries the primvars).
        """
        geometry = prim if prim.IsA(UsdGeom.Gprim) else prim.GetParent()
        primvars = UsdGeom.PrimvarsAPI(geometry)
        inputs: dict[str, ShaderInputValue] = {}
        for primvar_name in (PrimvarName.DISPLAY_COLOR, PrimvarName.DISPLAY_OPACITY):
            primvar = primvars.GetPrimvar(primvar_name)
            values = primvar.ComputeFlattened(self.frame.time) if primvar else None
            if values is not None and len(values) > 0:
                first = values[0]
                inputs[primvar_name] = (
                    tuple(float(component) for component in first)
                    if primvar_name is PrimvarName.DISPLAY_COLOR
                    else float(first)
                )
        name = "/".join(
            [PrimvarName.DISPLAY_COLOR]
            + [f"{key}={value}" for key, value in sorted(inputs.items())]
        )
        self.materials.setdefault(
            name, ImportedMaterial(name, MaterialSource.DISPLAY_COLOR, inputs)
        )
        return name

    def read(self, material: UsdShade.Material) -> ImportedMaterial:
        """
        The surface shader inputs of a material.

        :param material: The material.
        """
        name = material.GetPath().pathString
        shader, _, _ = material.ComputeSurfaceSource(
            RenderContext.MATERIAL_DEFINITION_LANGUAGE
        )
        if not shader:
            shader, _, _ = material.ComputeSurfaceSource()
        if not shader:
            return ImportedMaterial(name, MaterialSource.UNKNOWN)
        identifier = self.identifier_of(shader)
        source = next(
            (member for member in MaterialSource if member.value == identifier),
            MaterialSource.UNKNOWN,
        )
        inputs = {
            shader_input.GetBaseName(): value
            for shader_input in shader.GetInputs()
            if (value := self.input_value(shader_input)) is not None
        }
        return ImportedMaterial(name, source, inputs, identifier)

    @staticmethod
    def identifier_of(shader: UsdShade.Shader) -> str:
        """
        The shader's identifier: its MDL sub-identifier (or MDL file stem), else its
        ``info:id``.

        :param shader: The surface shader.
        """
        context = RenderContext.MATERIAL_DEFINITION_LANGUAGE
        sub_identifier = shader.GetSourceAssetSubIdentifier(context)
        if sub_identifier:
            return sub_identifier
        asset = shader.GetSourceAsset(context)
        if asset and asset.path:
            return Path(asset.path).stem
        return shader.GetIdAttr().Get() or ""

    def input_value(self, shader_input: UsdShade.Input) -> ShaderInputValue | None:
        """
        The value of a shader input, following connections to material interface
        inputs and to UsdUVTexture readers.

        :param shader_input: The input.
        """
        attributes = shader_input.GetValueProducingAttributes()
        if not attributes:
            return None
        attribute = attributes[0]
        if UsdShade.Output.IsOutput(attribute):
            return self.connected_texture(attribute)
        value = attribute.Get(self.frame.time)
        if isinstance(value, Sdf.AssetPath):
            return self.texture_file(
                value, attribute.GetColorSpace(), TextureChannel.RGB
            )
        return StageValue(value).shader_value()

    def connected_texture(self, output: Usd.Attribute) -> TextureFile | None:
        """
        The texture a UsdUVTexture output provides.

        :param output: The connected shader output attribute.
        """
        shader = UsdShade.Shader(output.GetPrim())
        if shader.GetIdAttr().Get() != ShaderIdentifier.TEXTURE:
            return None
        file_input = shader.GetInput(TextureInput.FILE)
        color_space_input = shader.GetInput(TextureInput.COLOR_SPACE)
        asset = file_input.Get(self.frame.time) if file_input else None
        if not isinstance(asset, Sdf.AssetPath):
            return None
        channel_name = UsdShade.Output(output).GetBaseName()
        channel = next(
            (member for member in TextureChannel if member.value == channel_name),
            TextureChannel.RGB,
        )
        color_space = color_space_input.Get() if color_space_input else ""
        return self.texture_file(asset, color_space or "", channel)

    def texture_file(
        self, asset: Sdf.AssetPath, color_space: str, channel: TextureChannel
    ) -> TextureFile | None:
        """
        A resolved texture file, or None (reported) when the asset does not resolve.

        :param asset: The authored asset path.
        :param color_space: The declared colour space, possibly empty.
        :param channel: The channels read.
        """
        if not asset.path:
            return None
        if not asset.resolvedPath or not Path(asset.resolvedPath).is_file():
            if asset.path not in self.report.missing_textures:
                self.report.missing_textures.append(asset.path)
            return None
        return TextureFile(
            Path(asset.resolvedPath),
            next(
                (member for member in ColorSpace if member.value == color_space),
                ColorSpace.AUTO,
            ),
            channel,
        )


@dataclass
class StageValue:
    """
    Converts values read from a stage into plain Python values.
    """

    value: Any
    """
    The value as ``pxr`` returns it.
    """

    def shader_value(self) -> ShaderInputValue | None:
        """
        The value as a shader input: numbers, flags, tokens and tuples of numbers.
        """
        if isinstance(self.value, (bool, int, float, str)):
            return self.value
        if self.value is None:
            return None
        converted = self.json_value()
        if isinstance(converted, list) and all(
            isinstance(item, (int, float)) for item in converted
        ):
            return tuple(float(item) for item in converted)
        return None

    def json_value(self) -> Any:
        """
        The value as JSON: arrays and vectors become lists, asset paths their path.
        """
        value = self.value
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        if isinstance(value, Sdf.AssetPath):
            return value.path
        if isinstance(value, (Gf.Quatd, Gf.Quatf, Gf.Quath)):
            return [
                float(value.GetReal()),
                *(float(component) for component in value.GetImaginary()),
            ]
        if isinstance(value, Iterable):
            return [StageValue(item).json_value() for item in value]
        return numpy.asarray(value).tolist()


# %% geometry


@dataclass
class GeometryReader:
    """
    Reads a renderable geometry prim as triangle meshes in its local frame, one per
    bound material (material subsets split a mesh).
    """

    frame: StageFrame
    """
    The stage frame, for the read time.
    """

    materials: MaterialReader
    """
    The material reader.
    """

    tessellation: ShapeTessellation = field(default_factory=ShapeTessellation)
    """
    The resolution of implicit shapes.
    """

    def pieces(self, prim: Usd.Prim) -> list[tuple[TriangleMesh, str]] | None:
        """
        The triangle meshes of a geometry prim with their material names, or None
        for an unsupported geometry type.

        :param prim: The geometry prim.
        """
        mesh = self.triangle_mesh(prim)
        if mesh is None:
            return None
        material = self.materials.material_of(prim)
        if not prim.IsA(UsdGeom.Mesh) or mesh.faces is None:
            return [(mesh, material)]
        pieces = []
        remaining = numpy.ones(len(mesh.triangles), dtype=bool)
        for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
            faces = numpy.asarray(subset.GetIndicesAttr().Get(self.frame.time) or [])
            selected = numpy.isin(mesh.faces, faces) & remaining
            if selected.any():
                pieces.append(
                    (
                        mesh.subset(selected),
                        self.materials.material_of(subset.GetPrim()),
                    )
                )
            remaining &= ~selected
        if remaining.all():
            return [(mesh, material)]
        if remaining.any():
            pieces.append((mesh.subset(remaining), material))
        return pieces

    def triangle_mesh(self, prim: Usd.Prim) -> TriangleMesh | None:
        """
        The triangles of one geometry prim, or None for an unsupported type.

        :param prim: The geometry prim.
        """
        time = self.frame.time
        if prim.IsA(UsdGeom.Mesh):
            return self.polygon_mesh(UsdGeom.Mesh(prim)).triangulate()
        if prim.IsA(UsdGeom.Cube):
            return self.tessellation.box(UsdGeom.Cube(prim).GetSizeAttr().Get(time))
        if prim.IsA(UsdGeom.Sphere):
            return self.tessellation.sphere(
                UsdGeom.Sphere(prim).GetRadiusAttr().Get(time)
            )
        if prim.IsA(UsdGeom.Cylinder):
            shape = UsdGeom.Cylinder(prim)
            return self.tessellation.cylinder(
                shape.GetRadiusAttr().Get(time),
                shape.GetHeightAttr().Get(time),
                Axis(shape.GetAxisAttr().Get(time)),
            )
        if prim.IsA(UsdGeom.Cone):
            shape = UsdGeom.Cone(prim)
            return self.tessellation.cone(
                shape.GetRadiusAttr().Get(time),
                shape.GetHeightAttr().Get(time),
                Axis(shape.GetAxisAttr().Get(time)),
            )
        if prim.IsA(UsdGeom.Capsule):
            shape = UsdGeom.Capsule(prim)
            return self.tessellation.capsule(
                shape.GetRadiusAttr().Get(time),
                shape.GetHeightAttr().Get(time),
                Axis(shape.GetAxisAttr().Get(time)),
            )
        return None

    def polygon_mesh(self, mesh: UsdGeom.Mesh) -> PolygonMesh:
        """
        The polygons and primvars of a mesh prim.

        :param mesh: The mesh.
        """
        time = self.frame.time
        primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        normals = None
        normals_primvar = primvars.GetPrimvar(PrimvarName.NORMALS)
        if normals_primvar and normals_primvar.HasValue():
            normals = self.primvar_data(normals_primvar)
        elif mesh.GetNormalsAttr().HasValue():
            normals = PrimvarData(
                numpy.asarray(mesh.GetNormalsAttr().Get(time), dtype=numpy.float64),
                Interpolation(mesh.GetNormalsInterpolation()),
            )
        coordinates = primvars.GetPrimvar(PrimvarName.TEXTURE_COORDINATES)
        if not (coordinates and coordinates.HasValue()):
            coordinates = next(
                (
                    primvar
                    for primvar in primvars.GetPrimvarsWithValues()
                    if primvar.GetTypeName().role
                    == Sdf.ValueRoleNames.TextureCoordinate
                ),
                None,
            )
        return PolygonMesh(
            points=numpy.asarray(
                mesh.GetPointsAttr().Get(time) or [], dtype=numpy.float64
            ).reshape(-1, 3),
            face_vertex_counts=numpy.asarray(
                mesh.GetFaceVertexCountsAttr().Get(time) or [], dtype=numpy.int64
            ),
            face_vertex_indices=numpy.asarray(
                mesh.GetFaceVertexIndicesAttr().Get(time) or [], dtype=numpy.int64
            ),
            normals=normals,
            texture_coordinates=(
                self.primvar_data(coordinates) if coordinates is not None else None
            ),
            left_handed=mesh.GetOrientationAttr().Get(time)
            == UsdGeom.Tokens.leftHanded,
        )

    def primvar_data(self, primvar: UsdGeom.Primvar) -> PrimvarData:
        """
        A primvar's flattened values and interpolation.

        :param primvar: The primvar.
        """
        return PrimvarData(
            numpy.asarray(
                primvar.ComputeFlattened(self.frame.time), dtype=numpy.float64
            ),
            Interpolation(primvar.GetInterpolation()),
        )


# %% joints


@dataclass
class JointReader:
    """
    Reads UsdPhysics joints: their kind, frames, axis, limits and the joint value the
    authored body poses imply.
    """

    frame: StageFrame
    """
    The stage frame.
    """

    report: ConversionReport
    """
    The report collecting unsupported joint types.
    """

    exclusion: PrimExclusion
    """
    The prims the conversion leaves out together with their subtrees.
    """

    def read(self, prim: Usd.Prim, name: str) -> ImportedJoint | None:
        """
        A joint without its links filled in, or None (reported) when it names no body
        or a body the conversion leaves out.

        :param prim: The joint prim.
        :param name: The joint's unique name.
        """
        joint = UsdPhysics.Joint(prim)
        path = prim.GetPath().pathString
        if self.joins_excluded_prim(joint):
            self.report.skip(SkipReason.JOINT_TO_EXCLUDED_PRIM, path)
            return None
        first = self.body_of(joint.GetBody0Rel())
        second = self.body_of(joint.GetBody1Rel())
        first_local = self.local_frame(
            joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()
        )
        second_local = self.local_frame(
            joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()
        )
        only_first = second is None and first is not None
        if only_first:
            first, second = second, first
            first_local, second_local = second_local, first_local
        if second is None:
            self.report.skip(SkipReason.JOINT_WITHOUT_CHILD, path)
            return None
        motion = self.motion_of(prim)
        if only_first:
            motion = motion.reversed()
        first_matrix = (
            self.frame.correction if first is None else self.frame.world_matrix(first)
        )
        parent_body_to_joint = self.body_to_joint(first_matrix, first_local)
        if first is None:
            parent_body_to_joint = Pose.rigid_part(first_matrix) @ parent_body_to_joint
        child_body_to_joint = self.body_to_joint(
            self.frame.world_matrix(second), second_local
        )
        imported = ImportedJoint(
            name=name,
            path=path,
            kind=motion.kind,
            parent="",
            child="",
            origin=Pose(),
            axis=motion.axis,
            lower=motion.lower,
            upper=motion.upper,
            parent_body=None if first is None else first.GetPath().pathString,
            child_body=second.GetPath().pathString,
            parent_body_to_joint=parent_body_to_joint,
            child_body_to_joint=child_body_to_joint,
            enabled=joint.GetJointEnabledAttr().Get(self.frame.time) is not False,
        )
        imported.initial_value = imported.value_between(
            Pose() if first is None else Pose.rigid_part(first_matrix),
            self.frame.world_pose(second),
        )
        return imported

    def joins_excluded_prim(self, joint: UsdPhysics.Joint) -> bool:
        """
        Whether a body of a joint lies in a subtree the conversion leaves out, inside
        the converted prims or beside them.

        :param joint: The joint.
        """
        targets = [*joint.GetBody0Rel().GetTargets(), *joint.GetBody1Rel().GetTargets()]
        return any(self.left_out(target) is not None for target in targets)

    def left_out(self, path: Sdf.Path) -> ExcludedPrim | None:
        """
        The outermost prim holding a body that the conversion leaves out, or None.

        :param path: The USD path of the body.
        """
        return self.exclusion.left_out_holding(self.frame.stage, path, self.frame.time)

    def detached_body(self, prim: Usd.Prim, graph: JointGraph) -> DetachedBody | None:
        """
        The environment body a joint hangs from a left-out parent body, or None when
        the joint holds no such body.

        :param prim: The joint prim.
        :param graph: The converted joints, to find the movable joints below the body.
        """
        joint = UsdPhysics.Joint(prim)
        parent = self.body_of(joint.GetBody0Rel())
        child = self.body_of(joint.GetBody1Rel())
        if parent is None or child is None:
            return None
        if self.left_out(child.GetPath()) is not None:
            return None
        if self.left_out(parent.GetPath()) is None:
            return None
        body = child.GetPath().pathString
        return DetachedBody(
            body,
            parent.GetPath().pathString,
            prim.GetPath().pathString,
            graph.movable_joints_below({body}),
        )

    def body_of(self, relationship: Usd.Relationship) -> Usd.Prim | None:
        """
        The prim a joint body relationship targets, or None.

        :param relationship: ``physics:body0`` or ``physics:body1``.
        """
        targets = relationship.GetTargets()
        if not targets:
            return None
        prim = self.frame.stage.GetPrimAtPath(targets[0])
        return prim if prim and prim.IsActive() else None

    def local_frame(
        self, position: Usd.Attribute, rotation: Usd.Attribute
    ) -> numpy.ndarray:
        """
        A joint frame in its body's (scaled) local coordinates, as a 4x4 matrix.

        :param position: ``physics:localPos0`` or ``physics:localPos1``.
        :param rotation: ``physics:localRot0`` or ``physics:localRot1``.
        """
        matrix = numpy.eye(4)
        quaternion = rotation.Get(self.frame.time)
        if quaternion is not None:
            matrix[:3, :3] = numpy.array(
                Gf.Matrix3d(Gf.Rotation(Gf.Quatd(quaternion)))
            ).T
        translation = position.Get(self.frame.time)
        if translation is not None:
            matrix[:3, 3] = numpy.asarray(translation, dtype=float)
        return matrix

    @staticmethod
    def body_to_joint(body_matrix: numpy.ndarray, local: numpy.ndarray) -> Pose:
        """
        The joint frame relative to the rigid frame of its body: the position goes
        through the body's full (possibly scaled) transform, the rotation through its
        rigid part.

        :param body_matrix: The body's world matrix (or the stage correction for the
            world).
        :param local: The joint frame in the body's local coordinates.
        """
        body = Pose.rigid_part(body_matrix)
        world = numpy.eye(4)
        world[:3, 3] = (body_matrix @ local[:, 3])[:3]
        world[:3, :3] = body.rotation @ local[:3, :3]
        return body.inverse() @ Pose(world)

    def motion_of(self, prim: Usd.Prim) -> AllowedMotion:
        """
        The motion a joint prim allows, with its limits in radians or metres.

        :param prim: The joint prim.
        """
        unit_vectors = {
            UsdPhysics.Tokens.x: numpy.array([1.0, 0.0, 0.0]),
            UsdPhysics.Tokens.y: numpy.array([0.0, 1.0, 0.0]),
            UsdPhysics.Tokens.z: numpy.array([0.0, 0.0, 1.0]),
        }
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            radians_per_degree = float(numpy.radians(1.0))
            lower = self.limit(joint.GetLowerLimitAttr(), radians_per_degree)
            upper = self.limit(joint.GetUpperLimitAttr(), radians_per_degree)
            kind = (
                JointKind.CONTINUOUS
                if lower is None and upper is None
                else JointKind.REVOLUTE
            )
            return AllowedMotion(
                kind, unit_vectors[joint.GetAxisAttr().Get()], lower, upper
            )
        if prim.IsA(UsdPhysics.PrismaticJoint):
            joint = UsdPhysics.PrismaticJoint(prim)
            scale = self.frame.meters_per_unit
            return AllowedMotion(
                JointKind.PRISMATIC,
                unit_vectors[joint.GetAxisAttr().Get()],
                self.limit(joint.GetLowerLimitAttr(), scale),
                self.limit(joint.GetUpperLimitAttr(), scale),
            )
        if not prim.IsA(UsdPhysics.FixedJoint):
            self.report.warnings.append(
                f"{prim.GetPath()}: {prim.GetTypeName()} is written as a fixed joint"
            )
        return AllowedMotion(JointKind.FIXED, unit_vectors[UsdPhysics.Tokens.x])

    def limit(self, attribute: Usd.Attribute, scale: float) -> float | None:
        """
        A finite limit in radians or metres, or None when unset or infinite.

        :param attribute: The limit attribute.
        :param scale: Degrees to radians, or stage units to metres.
        """
        value = attribute.Get(self.frame.time)
        if value is None or not numpy.isfinite(value):
            return None
        return float(value) * scale


# %% lights


@dataclass
class LightReader:
    """
    Reads UsdLux lights.
    """

    frame: StageFrame
    """
    The stage frame.
    """

    def read(self, prim: Usd.Prim) -> ImportedLight | None:
        """
        A light, or None for a light type the lighting description has no use for.

        :param prim: The light prim.
        """
        kind = next(
            (member for member in LightKind if member.value == prim.GetTypeName()), None
        )
        if kind is None:
            return None
        time = self.frame.time
        light = UsdLux.LightAPI(prim)
        matrix = self.frame.world_matrix(prim)
        scales = numpy.linalg.norm(matrix[:3, :3], axis=0)
        intensity = float(light.GetIntensityAttr().Get(time) or 0.0)
        exposure = float(light.GetExposureAttr().Get(time) or 0.0)
        color = light.GetColorAttr().Get(time)
        imported = ImportedLight(
            path=prim.GetPath().pathString,
            kind=kind,
            pose=Pose.rigid_part(matrix),
            intensity=intensity * 2.0**exposure,
            color=(
                tuple(float(component) for component in color)
                if color is not None
                else (1.0, 1.0, 1.0)
            ),
            color_temperature=(
                float(light.GetColorTemperatureAttr().Get(time))
                if light.GetEnableColorTemperatureAttr().Get(time)
                else None
            ),
        )
        if kind is LightKind.RECTANGLE:
            rectangle = UsdLux.RectLight(prim)
            imported.width = float(rectangle.GetWidthAttr().Get(time)) * scales[0]
            imported.height = float(rectangle.GetHeightAttr().Get(time)) * scales[1]
        if kind in (LightKind.DISK, LightKind.SPHERE, LightKind.CYLINDER):
            schema = {
                LightKind.DISK: UsdLux.DiskLight,
                LightKind.SPHERE: UsdLux.SphereLight,
                LightKind.CYLINDER: UsdLux.CylinderLight,
            }[kind]
            radius = schema(prim).GetRadiusAttr().Get(time)
            imported.radius = float(radius or 0.0) * float(scales.mean())
        if kind is LightKind.DISTANT:
            imported.angle = float(UsdLux.DistantLight(prim).GetAngleAttr().Get(time))
        return imported


# %% motion


@dataclass
class StageMotion:
    """
    Samples the world poses of prims whose transforms change over the stage's time
    samples.
    """

    frame: StageFrame
    """
    The stage frame.
    """

    source: Path
    """
    The stage file, named as the source of the sampled poses.
    """

    def trace(self, paths: list[str], frames_per_second: float) -> BodyTrace:
        """
        The rigid world poses of the moving prims among some, sampled at a rate over
        the time samples of their transforms (one frame when none of them moves).

        :param paths: The paths of the prims to sample.
        :param frames_per_second: The sample rate.
        """
        stage = self.frame.stage
        prims = [
            prim
            for path in paths
            if (prim := stage.GetPrimAtPath(path)) and self.transforms_of(prim)
        ]
        if not prims:
            return BodyTrace(frames_per_second, source=str(self.source))
        sample_times = sorted(
            {
                time
                for prim in prims
                for transform in self.transforms_of(prim)
                for time in transform.GetTimeSamples()
            }
        )
        rate = stage.GetTimeCodesPerSecond()
        count = (
            int(
                numpy.floor(
                    (sample_times[-1] - sample_times[0]) / rate * frames_per_second
                    + 1e-9
                )
            )
            + 1
        )
        matrices = numpy.zeros((len(prims), count, 4, 4))
        for index in range(count):
            transforms = UsdGeom.XformCache(
                Usd.TimeCode(sample_times[0] + index * rate / frames_per_second)
            )
            for prim_index, prim in enumerate(prims):
                matrices[prim_index, index] = (
                    self.frame.correction
                    @ numpy.array(transforms.GetLocalToWorldTransform(prim)).T
                )
        return BodyTrace(
            frames_per_second,
            {
                prim.GetPath().pathString: PoseTrack.rigid_part(matrices[prim_index])
                for prim_index, prim in enumerate(prims)
            },
            count,
            str(self.source),
        )

    @staticmethod
    def transforms_of(prim: Usd.Prim) -> list[UsdGeom.Xformable]:
        """
        The transforms of a prim and its ancestors that might change over time.

        :param prim: The prim.
        """
        transforms = []
        while prim and not prim.IsPseudoRoot():
            transform = UsdGeom.Xformable(prim)
            if transform and transform.TransformMightBeTimeVarying():
                transforms.append(transform)
            prim = prim.GetParent()
        return transforms


# %% excluded prims


@dataclass
class PrimExclusion:
    """
    Which prims a conversion leaves out together with their subtrees: the excluded
    prims, and the robots the stage marks unless they are kept.
    """

    excluded_paths: list[str] = field(default_factory=list)
    """
    The prims left out on request.
    """

    keep_robots: bool = False
    """
    Whether the robots the stage marks are converted with the environment.
    """

    robot_path: str | None = None
    """
    The prim of the robot converted as a model of its own, or None; a marked robot
    holding this prim or lying below it is kept.
    """

    def validate(
        self, stage: Usd.Stage, roots: list[Usd.Prim], time: Usd.TimeCode
    ) -> None:
        """
        Refuse an excluded prim the stage does not have, a kept robot inside an
        excluded prim, and a selected prim inside a prim the conversion leaves out.

        :param stage: The composed stage.
        :param roots: The prims whose subtrees are converted.
        :param time: The time the stage is read at.
        """
        for path in self.excluded_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsActive():
                raise PrimNotFound(path)
            if self.robot_path is not None and Sdf.Path(self.robot_path).HasPrefix(
                Sdf.Path(path)
            ):
                raise RobotExcluded(self.robot_path, path)
        for root in roots:
            if root.IsPseudoRoot():
                continue
            left_out = self.left_out_holding(stage, root.GetPath(), time)
            if left_out is not None:
                raise SelectedPrimLeftOut(root.GetPath().pathString, left_out)

    def left_out_holding(
        self, stage: Usd.Stage, path: Sdf.Path, time: Usd.TimeCode
    ) -> ExcludedPrim | None:
        """
        The outermost of a prim and its ancestors that the conversion leaves out, and
        why, or None when the prim is converted.

        :param stage: The composed stage.
        :param path: The USD path of the prim; only the ancestors the stage has count.
        :param time: The time the stage is read at.
        """
        for prefix in path.GetPrefixes():
            prim = stage.GetPrimAtPath(prefix)
            if not prim:
                return None
            reason = self.reason_to_leave_out(prim, time)
            if reason is not None:
                return ExcludedPrim(prefix.pathString, reason)
        return None

    def belongs_to_robot(
        self, stage: Usd.Stage, path: Sdf.Path, time: Usd.TimeCode
    ) -> bool:
        """
        Whether a prim is or lies in a robot the stage marks, or is, holds or lies in
        the kept robot.

        :param stage: The composed stage.
        :param path: The USD path of the prim.
        :param time: The time the stage is read at.
        """
        if self.holds_kept_robot(path):
            return True
        return any(
            self.robot_mark(stage.GetPrimAtPath(prefix), time) is not None
            for prefix in path.GetPrefixes()
        )

    def reason_to_leave_out(
        self, prim: Usd.Prim, time: Usd.TimeCode
    ) -> ExclusionReason | None:
        """
        Why a prim is left out together with its subtree, or None when it is converted.

        :param prim: The prim.
        :param time: The time the stage is read at.
        """
        path = prim.GetPath()
        if any(path.HasPrefix(Sdf.Path(excluded)) for excluded in self.excluded_paths):
            return ExclusionReason.REQUESTED
        if self.keep_robots or self.holds_kept_robot(path):
            return None
        return self.robot_mark(prim, time)

    def holds_kept_robot(self, path: Sdf.Path) -> bool:
        """
        Whether a prim is the kept robot's prim, lies above it or lies below it.

        :param path: The prim path.
        """
        if self.robot_path is None:
            return False
        robot = Sdf.Path(self.robot_path)
        return robot.HasPrefix(path) or path.HasPrefix(robot)

    @staticmethod
    def robot_mark(prim: Usd.Prim, time: Usd.TimeCode) -> ExclusionReason | None:
        """
        The mark by which the stage declares a prim the root of a robot, or None.

        .. note:: The authored schema list is read because USD leaves a schema it has
            no definition for, such as Isaac Sim's robot schema in usd-core, out of the
            applied schemas.

        :param prim: The prim.
        :param time: The time the stage is read at.
        """
        schemas = prim.GetPrimTypeInfo().GetAppliedAPISchemas()
        if RobotMarker.ISAAC_ROBOT_SCHEMA in schemas:
            return ExclusionReason.ISAAC_ROBOT_SCHEMA
        entity_type = prim.GetAttribute(
            SemanticNamespace.SEMANTIC_DIGITAL_TWIN.attribute_name(
                SemanticAttributeName.ENTITY_TYPE
            )
        )
        if entity_type and entity_type.Get(time) == RobotMarker.ROBOT_ENTITY_TYPE:
            return ExclusionReason.ROBOT_ENTITY_TYPE
        return None


# %% stage


@dataclass(frozen=True)
class VariantSelection:
    """
    A variant to select on the composed stage, in its session layer only.
    """

    prim_path: str
    """
    The prim holding the variant set.
    """

    variant_set: str
    """
    The name of the variant set.
    """

    variant: str
    """
    The variant to select.
    """

    def apply(self, stage: Usd.Stage) -> None:
        """
        Select the variant in the stage's current edit target.

        :param stage: The composed stage.
        """
        prim = stage.GetPrimAtPath(self.prim_path)
        variant_sets = prim.GetVariantSets() if prim else None
        if (
            variant_sets is None
            or not variant_sets.HasVariantSet(self.variant_set)
            or self.variant
            not in variant_sets.GetVariantSet(self.variant_set).GetVariantNames()
        ):
            raise VariantNotFound(self.prim_path, self.variant_set, self.variant)
        variant_sets.GetVariantSet(self.variant_set).SetVariantSelection(self.variant)


@dataclass
class StageReader:
    """
    Reads a composed USD stage into a :class:`SceneDescription`.
    """

    stage_path: Path
    """
    The stage file.
    """

    prim_paths: list[str] = field(default_factory=list)
    """
    The prims whose subtrees are converted; empty for the whole stage.
    """

    variant_selections: list[VariantSelection] = field(default_factory=list)
    """
    The variants selected in the stage's in-memory session layer before reading; the
    stage's files are never changed.
    """

    exclusion: PrimExclusion = field(default_factory=PrimExclusion)
    """
    The prims left out with their subtrees; by default the robots the stage marks.
    """

    composed: Usd.Stage | None = field(default=None, init=False, repr=False)
    """
    The composed stage once it has been opened.
    """

    def open(self) -> Usd.Stage:
        """
        The composed stage with all payloads loaded and the variants selected, opened
        on first use.
        """
        if self.composed is not None:
            return self.composed
        if not self.stage_path.is_file():
            raise StageNotFound(self.stage_path)
        stage = Usd.Stage.Open(str(self.stage_path), Usd.Stage.LoadAll)
        if stage is None:
            raise StageUnreadable(self.stage_path)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for selection in self.variant_selections:
                selection.apply(stage)
        self.composed = stage
        return stage

    def read(self) -> SceneDescription:
        """
        The neutral description of the selected part of the stage.
        """
        stage = self.open()
        frame = StageFrame.of_stage(stage)
        roots = self.selected_roots(stage)
        self.exclusion.validate(stage, roots, frame.time)
        return StageConversion(frame, roots, self.stage_path, self.exclusion).run()

    def motion(self, paths: list[str], frames_per_second: float) -> BodyTrace:
        """
        The world poses of the moving prims among some, sampled from the stage's time
        samples.

        :param paths: The paths of the prims to sample.
        :param frames_per_second: The sample rate.
        """
        stage = self.open()
        return StageMotion(StageFrame.of_stage(stage), self.stage_path).trace(
            paths, frames_per_second
        )

    def selected_roots(self, stage: Usd.Stage) -> list[Usd.Prim]:
        """
        The prims whose subtrees are converted; a prim inside another selected prim's
        subtree is already covered by it.

        :param stage: The composed stage.
        """
        if not self.prim_paths:
            return [stage.GetPseudoRoot()]
        roots = []
        for path in self.prim_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsActive():
                raise PrimNotFound(path)
            roots.append(prim)
        return [
            prim
            for prim in roots
            if not any(
                other != prim and prim.GetPath().HasPrefix(other.GetPath())
                for other in roots
            )
        ]


@dataclass
class StageConversion:
    """
    One conversion of a composed stage: traversal, link partition, kinematic tree,
    geometry, lights and semantics.
    """

    frame: StageFrame
    """
    The stage frame.
    """

    roots: list[Usd.Prim]
    """
    The prims whose subtrees are converted.
    """

    source: Path
    """
    The stage file.
    """

    exclusion: PrimExclusion = field(default_factory=PrimExclusion)
    """
    The prims left out with their subtrees.
    """

    report: ConversionReport = field(default_factory=ConversionReport)
    """
    What the conversion read, wrote and left out.
    """

    def run(self) -> SceneDescription:
        """
        Convert the selected prims.
        """
        stage = self.frame.stage
        default_prim = stage.GetDefaultPrim()
        root_path = default_prim.GetPath().pathString if default_prim else "/"
        link_names = NameRegistry(root_path if default_prim else "")
        joint_names = NameRegistry(root_path if default_prim else "")
        prims = self.traverse()
        materials = MaterialReader(self.frame, self.report)
        joint_reader = JointReader(self.frame, self.report, self.exclusion)
        joints = [
            joint
            for prim in prims
            if prim.IsA(UsdPhysics.Joint)
            and (
                joint := joint_reader.read(
                    prim, joint_names.name_of(prim.GetPath().pathString)
                )
            )
            is not None
        ]
        graph = JointGraph(joints)
        self.report.detached = self.detached_bodies(prims, joint_reader, graph)
        self.report.articulations = self.environment_articulations(prims, graph)
        bodies = {
            prim.GetPath().pathString
            for prim in prims
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        }
        bodies |= {joint.child_body for joint in joints}
        bodies |= {
            joint.parent_body for joint in joints if joint.parent_body is not None
        }
        partition = LinkPartition(root_path, bodies, self.asset_roots(prims, root_path))
        geometry = [prim for prim in prims if self.renderable(prim)]
        tree = KinematicTree(
            self.frame, partition, link_names, joint_names, self.report
        )
        tree.build(
            joints,
            [partition.link_path_of(prim.GetPath().pathString) for prim in geometry],
        )
        reader = GeometryReader(self.frame, materials)
        world_points = [numpy.zeros((0, 3))]
        for prim in geometry:
            world_points += self.add_geometry(prim, reader, tree)
        description = SceneDescription(
            source=self.source,
            root_link=tree.links[tree.root_path].name,
            links=tree.imported_links(),
            joints=tree.imported_joints(),
            materials=materials.materials,
            lights=self.lights(prims),
            semantics=self.semantics(prims, tree),
            cameras=self.cameras(prims),
            used_files=self.used_files(materials),
            report=self.report,
        )
        points = numpy.concatenate(world_points)
        if len(points):
            description.bounds = Bounds.of_points(points)
            description.central_bounds = Bounds.central(points, 0.98)
        self.count(description, prims)
        return description

    def traverse(self) -> list[Usd.Prim]:
        """
        Every active prim under the selected roots, instance proxies included, except
        the subtrees the exclusion leaves out.
        """
        prims = []
        for root in self.roots:
            prims += self.subtree(root)
        return prims

    def subtree(self, root: Usd.Prim) -> list[Usd.Prim]:
        """
        The active prims of one selected subtree, parents before children; a prim the
        exclusion leaves out is reported and its subtree is not entered.

        :param root: The selected prim.
        """
        prims = []
        iterator = iter(
            Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
        )
        for prim in iterator:
            if prim.IsPseudoRoot():
                continue
            reason = self.exclusion.reason_to_leave_out(prim, self.frame.time)
            if reason is not None:
                self.report.excluded.append(
                    ExcludedPrim(prim.GetPath().pathString, reason)
                )
                iterator.PruneChildren()
                continue
            prims.append(prim)
        return prims

    def detached_bodies(
        self, prims: list[Usd.Prim], joint_reader: JointReader, graph: JointGraph
    ) -> list[DetachedBody]:
        """
        The converted bodies that hung from a left-out body, by a joint among the
        converted prims or inside a left-out prim, in joint path order.

        :param prims: The traversed prims.
        :param joint_reader: The joint reader of the conversion.
        :param graph: The converted joints.
        """
        left_out_joints = [
            prim
            for excluded in self.report.excluded
            for prim in Usd.PrimRange(
                self.frame.stage.GetPrimAtPath(excluded.path),
                Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate),
            )
            if prim.IsA(UsdPhysics.Joint)
        ]
        joints = [prim for prim in prims if prim.IsA(UsdPhysics.Joint)]
        detached = [
            body
            for prim in joints + left_out_joints
            if (body := joint_reader.detached_body(prim, graph)) is not None
            and self.converts(Sdf.Path(body.body))
        ]
        return sorted(detached, key=lambda body: body.joint)

    def converts(self, path: Sdf.Path) -> bool:
        """
        Whether a prim lies in one of the selected subtrees.

        :param path: The USD path of the prim.
        """
        return any(path.HasPrefix(root.GetPath()) for root in self.roots)

    def environment_articulations(
        self, prims: list[Usd.Prim], graph: JointGraph
    ) -> list[EnvironmentArticulation]:
        """
        The converted articulation roots outside every robot, with the movable joints
        of each; an articulation without any is left out of the list.

        :param prims: The traversed prims.
        :param graph: The converted joints.
        """
        articulations = []
        for prim in prims:
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue
            path = prim.GetPath()
            if self.exclusion.belongs_to_robot(self.frame.stage, path, self.frame.time):
                continue
            movable = graph.movable_joints_below(graph.bodies_in(path.pathString))
            if movable:
                articulations.append(EnvironmentArticulation(path.pathString, movable))
        return articulations

    @staticmethod
    def asset_roots(prims: list[Usd.Prim], root_path: str) -> set[str]:
        """
        The outermost prims below the default prim that reference or load another
        layer, or are component models.

        :param prims: The traversed prims, parents before children.
        :param root_path: The default prim's path.
        """
        assets: set[str] = set()
        for prim in prims:
            path = prim.GetPath()
            if path.pathElementCount < 2 or path.pathString == root_path:
                continue
            if not (
                prim.HasAuthoredReferences()
                or prim.HasAuthoredPayloads()
                or Usd.ModelAPI(prim).IsKind(
                    Kind.Tokens.component, Usd.ModelAPI.KindValidationNone
                )
            ):
                continue
            if any(prefix.pathString in assets for prefix in path.GetPrefixes()[:-1]):
                continue
            assets.add(path.pathString)
        return assets

    def renderable(self, prim: Usd.Prim) -> bool:
        """
        Whether a prim is visible render geometry; guide, proxy and invisible
        geometry is reported as skipped.

        :param prim: The prim.
        """
        if not prim.IsA(UsdGeom.Gprim):
            return False
        path = prim.GetPath().pathString
        imageable = UsdGeom.Imageable(prim)
        purpose = imageable.ComputePurpose()
        if purpose == Purpose.GUIDE:
            self.report.skip(SkipReason.GUIDE_PURPOSE, path)
            return False
        if purpose == Purpose.PROXY:
            self.report.skip(SkipReason.PROXY_PURPOSE, path)
            return False
        if imageable.ComputeVisibility(self.frame.time) == UsdGeom.Tokens.invisible:
            self.report.skip(SkipReason.INVISIBLE, path)
            return False
        return True

    def add_geometry(
        self, prim: Usd.Prim, reader: GeometryReader, tree: KinematicTree
    ) -> list[numpy.ndarray]:
        """
        Read one geometry prim into its link and return its world points.

        :param prim: The geometry prim.
        :param reader: The geometry reader.
        :param tree: The kinematic tree holding the links.
        """
        path = prim.GetPath().pathString
        pieces = reader.pieces(prim)
        if pieces is None:
            self.report.skip(SkipReason.UNSUPPORTED_GEOMETRY, path)
            return []
        link = tree.links[tree.partition.link_path_of(path)]
        to_link = link.pose.inverse().matrix @ self.frame.world_matrix(prim)
        double_sided = bool(UsdGeom.Gprim(prim).GetDoubleSidedAttr().Get())
        world_points = []
        for mesh, material in pieces:
            if len(mesh.triangles) == 0:
                self.report.skip(SkipReason.EMPTY_GEOMETRY, path)
                continue
            local = mesh.transformed(to_link)
            link.meshes.append(
                ImportedMesh(
                    path=path,
                    points=local.points.astype(numpy.float32),
                    triangles=local.triangles.astype(numpy.uint32),
                    material=material,
                    normals=(
                        None
                        if local.normals is None
                        else local.normals.astype(numpy.float32)
                    ),
                    texture_coordinates=(
                        None
                        if local.texture_coordinates is None
                        else local.texture_coordinates.astype(numpy.float32)
                    ),
                    double_sided=double_sided,
                )
            )
            world_points.append(
                local.points @ link.pose.rotation.T + link.pose.translation
            )
        return world_points

    def lights(self, prims: list[Usd.Prim]) -> list[ImportedLight]:
        """
        The visible lights.

        :param prims: The traversed prims.
        """
        reader = LightReader(self.frame)
        lights = []
        for prim in prims:
            if not prim.HasAPI(UsdLux.LightAPI):
                continue
            if (
                UsdGeom.Imageable(prim).ComputeVisibility(self.frame.time)
                == UsdGeom.Tokens.invisible
            ):
                continue
            light = reader.read(prim)
            if light is not None:
                lights.append(light)
        return lights

    def cameras(self, prims: list[Usd.Prim]) -> list[ImportedCamera]:
        """
        The cameras, with their vertical field of view.

        :param prims: The traversed prims.
        """
        cameras = []
        for prim in prims:
            if not prim.IsA(UsdGeom.Camera):
                continue
            camera = UsdGeom.Camera(prim)
            focal_length = camera.GetFocalLengthAttr().Get(self.frame.time)
            aperture = camera.GetVerticalApertureAttr().Get(self.frame.time)
            cameras.append(
                ImportedCamera(
                    prim.GetPath().pathString,
                    self.frame.world_pose(prim),
                    float(
                        numpy.degrees(2.0 * numpy.arctan(aperture / 2.0 / focal_length))
                    ),
                )
            )
        return cameras

    def semantics(
        self, prims: list[Usd.Prim], tree: KinematicTree
    ) -> list[PrimSemantics]:
        """
        The semantic attributes of every prim carrying any.

        :param prims: The traversed prims.
        :param tree: The kinematic tree, to name each prim's link.
        """
        namespaces = {member.value for member in SemanticNamespace}
        result = []
        for prim in prims:
            attributes: dict[str, dict[str, Any]] = {}
            for attribute in prim.GetAuthoredAttributes():
                namespace, _, base_name = attribute.GetName().partition(":")
                if namespace not in namespaces or not base_name:
                    continue
                value = attribute.Get(self.frame.time)
                attributes.setdefault(namespace, {})[base_name] = StageValue(
                    value
                ).json_value()
            if attributes:
                path = prim.GetPath().pathString
                result.append(PrimSemantics(path, attributes, tree.link_name_of(path)))
        return result

    def used_files(self, materials: MaterialReader) -> list[Path]:
        """
        The layer files the stage composed and the textures the materials read.

        :param materials: The material reader.
        """
        files = {
            Path(layer.realPath)
            for layer in self.frame.stage.GetUsedLayers()
            if layer.realPath and Path(layer.realPath).is_file()
        }
        for material in materials.materials.values():
            files |= {
                value.path
                for value in material.inputs.values()
                if isinstance(value, TextureFile)
            }
        return sorted(files)

    def count(self, description: SceneDescription, prims: list[Usd.Prim]) -> None:
        """
        Fill the report's counts.

        :param description: The finished description.
        :param prims: The traversed prims.
        """
        counts = self.report.counts
        counts[ReportCount.PRIMS] = len(prims)
        counts[ReportCount.INSTANCE_PROXIES] = sum(
            1 for prim in prims if prim.IsInstanceProxy()
        )
        counts[ReportCount.LINKS] = len(description.links)
        counts[ReportCount.MESHES] = sum(len(link.meshes) for link in description.links)
        counts[ReportCount.TRIANGLES] = sum(
            link.triangle_count for link in description.links
        )
        counts[ReportCount.MATERIALS] = len(description.materials)
        counts[ReportCount.LIGHTS] = len(description.lights)
        counts[ReportCount.SEMANTIC_PRIMS] = len(description.semantics)
        for joint in description.joints:
            counts[ReportCount.JOINTS.of_kind(joint.kind)] += 1
            counts[ReportCount.DISABLED_JOINTS] += not joint.enabled


# %% kinematic tree


@dataclass
class KinematicTree:
    """
    The links of a conversion and the joints connecting them into one tree rooted in
    the world.
    """

    frame: StageFrame
    """
    The stage frame.
    """

    partition: LinkPartition
    """
    The partition of prims into links.
    """

    link_names: NameRegistry
    """
    The registry naming links.
    """

    joint_names: NameRegistry
    """
    The registry naming joints.
    """

    report: ConversionReport
    """
    The report collecting skipped joints.
    """

    links: dict[str, LinkRecord] = field(default_factory=dict)
    """
    The links by prim path.
    """

    @property
    def root_path(self) -> str:
        """
        The path of the root link's prim.
        """
        return self.partition.root_path

    def build(self, joints: list[ImportedJoint], geometry_links: list[str]) -> None:
        """
        Create the links (bodies, assets, and groups holding geometry), connect joint
        children to their parents, attach every other link to the nearest link above
        it, and place each link frame.

        :param joints: The joints read from the stage.
        :param geometry_links: The link path of every renderable geometry prim.
        """
        self.add_link(self.root_path, False)
        for path in sorted(self.partition.bodies):
            self.add_link(path, True)
        for path in sorted(self.partition.assets | set(geometry_links)):
            self.add_link(path, False)
        for joint in sorted(joints, key=lambda joint: (not joint.enabled, joint.path)):
            self.connect(joint)
        for path in sorted(self.links):
            if path != self.root_path and self.links[path].parent is None:
                self.attach(path)
        for record in self.links.values():
            if record.joint is not None:
                self.place(record)

    def add_link(self, path: str, is_body: bool) -> None:
        """
        Create a link for a prim once, framed at the prim's rigid world pose.

        :param path: The prim path.
        :param is_body: Whether the prim is a rigid or joint body.
        """
        if path in self.links:
            return
        prim = self.frame.stage.GetPrimAtPath(path)
        pose = (
            self.frame.world_pose(prim) if prim and not prim.IsPseudoRoot() else Pose()
        )
        if path == self.root_path:
            pose = Pose()
        self.links[path] = LinkRecord(
            path, self.link_names.name_of(path), pose, pose, is_body=is_body
        )

    def ancestors(self, path: str) -> list[str]:
        """
        The chain of parent links above a link, as far as it is connected yet.

        :param path: The link path.
        """
        chain = []
        parent = self.links[path].parent
        while parent is not None and parent not in chain:
            chain.append(parent)
            parent = self.links[parent].parent
        return chain

    def connect(self, joint: ImportedJoint) -> None:
        """
        Make a joint's child body a child link of the link holding its parent body
        (for a joint to the world, of the static asset enclosing the child), unless
        that would give the child a second parent or close a loop.

        :param joint: The joint.
        """
        child = joint.child_body
        parent = (
            self.static_asset_above(child)
            if joint.parent_body is None
            else self.partition.link_path_of(joint.parent_body)
        )
        if (
            child not in self.links
            or child == self.root_path
            or self.links[child].parent is not None
            or parent == child
            or child in self.ancestors(parent)
        ):
            self.report.skip(SkipReason.LOOP_CLOSING_JOINT, joint.path)
            return
        record = self.links[child]
        parent_body_pose = (
            Pose()
            if joint.parent_body is None
            else self.frame.world_pose(
                self.frame.stage.GetPrimAtPath(joint.parent_body)
            )
        )
        if joint.kind.movable:
            record.joint_frame = parent_body_pose @ joint.parent_body_to_joint
            record.pose = record.joint_frame @ joint.motion(joint.initial_value)
        else:
            record.pose = record.prim_pose @ joint.child_body_to_joint
        joint.parent = self.links[parent].name
        joint.child = record.name
        record.parent = parent
        record.joint = joint

    def static_asset_above(self, path: str) -> str:
        """
        The asset link enclosing a prim when no rigid body lies between it and the
        world, else the root: the link a joint to the world can hang from without
        changing what it is fixed to.

        :param path: The prim path.
        """
        prefixes = [prefix.pathString for prefix in Sdf.Path(path).GetPrefixes()[:-1]]
        if any(prefix in self.partition.bodies for prefix in prefixes):
            return self.root_path
        return next(
            (
                prefix
                for prefix in reversed(prefixes)
                if prefix in self.partition.assets
            ),
            self.root_path,
        )

    def attach(self, path: str) -> None:
        """
        Attach a link without a joint to the nearest link above its prim with a fixed
        joint at its authored pose (to the root if that would close a loop).

        :param path: The link path.
        """
        record = self.links[path]
        parent = self.nearest_link_above(path)
        if path in self.ancestors(parent) or parent == path:
            parent = self.root_path
        record.parent = parent
        record.joint = ImportedJoint(
            name=self.joint_names.name_of(f"{path}/{GeneratedName.ATTACHMENT}"),
            path=path,
            kind=JointKind.FIXED,
            parent=self.links[parent].name,
            child=record.name,
            origin=Pose(),
        )

    def place(self, record: LinkRecord) -> None:
        """
        Set a joint's origin from the final link frames: the joint frame (movable
        joints) or the child link frame (fixed joints) in the parent link frame.

        :param record: The child link of the joint.
        """
        parent_pose = self.links[record.parent].pose
        placed = record.joint_frame if record.joint_frame is not None else record.pose
        record.joint.origin = parent_pose.inverse() @ placed

    def nearest_link_above(self, path: str) -> str:
        """
        The link of the closest ancestor prim that has one, else the root.

        :param path: The link path.
        """
        current = Sdf.Path(path).GetParentPath()
        while not current.IsAbsoluteRootPath() and not current.isEmpty:
            if current.pathString in self.links:
                return current.pathString
            current = current.GetParentPath()
        return self.root_path

    def link_name_of(self, path: str) -> str:
        """
        The name of the link a prim belongs to: the nearest existing link above it.

        :param path: The prim path.
        """
        current = Sdf.Path(path)
        while not current.IsAbsoluteRootPath() and not current.isEmpty:
            if current.pathString in self.links:
                return self.links[current.pathString].name
            current = current.GetParentPath()
        return self.links[self.root_path].name

    def order(self) -> list[str]:
        """
        Link paths with every parent before its children.
        """
        children: dict[str, list[str]] = {}
        for path, record in self.links.items():
            if record.parent is not None:
                children.setdefault(record.parent, []).append(path)
        ordered = [self.root_path]
        for path in ordered:
            ordered += sorted(children.get(path, []))
        return ordered

    def imported_links(self) -> list[ImportedLink]:
        """
        The links, parents before children.
        """
        links = []
        for path in self.order():
            record = self.links[path]
            links.append(
                ImportedLink(
                    name=record.name,
                    path=path,
                    pose=record.pose,
                    prim_to_link=record.prim_pose.inverse() @ record.pose,
                    meshes=record.meshes,
                    free=self.free(path),
                )
            )
        return links

    def free(self, path: str) -> bool:
        """
        Whether a link moves with a rigid body that no authored joint connects to the
        world: the link itself, or a link above it, is a body attached by a generated
        fixed joint.

        :param path: The link path.
        """
        for current in [path, *self.ancestors(path)]:
            record = self.links[current]
            if (
                record.is_body
                and record.joint is not None
                and record.joint.child_body is None
            ):
                return True
        return False

    def imported_joints(self) -> list[ImportedJoint]:
        """
        The joints, in the order of their child links.
        """
        return [
            self.links[path].joint
            for path in self.order()
            if self.links[path].joint is not None
        ]
