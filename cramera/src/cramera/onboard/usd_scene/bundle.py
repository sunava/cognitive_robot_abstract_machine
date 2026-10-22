"""
Writing a converted stage as a self-contained CRAMERA scene bundle: ``scene.json``,
``trajectory.json``, ``environment.urdf``, one GLB per link, the semantics, lighting
and conversion report sidecars, and the refreshed scenes index.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import numpy
from typing_extensions import Any, Self

from cramera._version import __version__
from cramera.knowledge.detected_events import SceneField
from cramera.mesh_format import MeshFormat
from cramera.onboard.scene_index import validate_scene_name, write_scene_index
from cramera.onboard.usd_scene.animation import (
    AnimatedObject,
    KinematicModel,
    LinkPlacement,
    RobotModel,
    SceneAnimation,
)
from cramera.onboard.usd_scene.bundle_layout import BundleFile, ModelName
from cramera.onboard.usd_scene.gltf import LinkGltf
from cramera.onboard.usd_scene.materials import MaterialMapper, PhysicallyBasedMaterial
from cramera.onboard.usd_scene.robot_description import RobotDescription
from cramera.onboard.usd_scene.scene_description import (
    Bounds,
    ImportedCamera,
    ImportedLight,
    LightKind,
    Pose,
    SceneDescription,
)
from cramera.scene_presentation import PresentationField

# %% bundle vocabulary


class BundleKey(StrEnum):
    """
    The JSON keys the converter writes.
    """

    NAME = "name"
    """
    A scene's or model's name.
    """

    FRAMES_PER_SECOND = "framesPerSecond"
    """
    The playback rate.
    """

    TRAJECTORY = "trajectory"
    """
    The bundle-relative trajectory file.
    """

    AMBIENT_OCCLUSION = "ambientOcclusion"
    """
    Whether screen-space ambient occlusion is on.
    """

    EXPOSURE = "exposure"
    """
    The tone mapping exposure.
    """

    GROUND_PLANE = "groundPlane"
    """
    Whether the viewer adds its own floor.
    """

    SHADOW_BOUNDS = "shadowBounds"
    """
    The box the key light's shadows cover.
    """

    POSITION = "position"
    """
    Where the camera stands.
    """

    TARGET = "target"
    """
    The point the camera looks at.
    """

    FIELD_OF_VIEW = "fov"
    """
    The vertical field of view in degrees.
    """

    FAR = "far"
    """
    The far clipping distance.
    """

    URDF = "urdf"
    """
    A model's bundle-relative URDF file.
    """

    PREFIX = "prefix"
    """
    The prefix of a model's joint names in the trajectory.
    """

    LINKS = "links"
    """
    The number of links of a model.
    """

    MOVABLE_JOINTS = "movableJoints"
    """
    The joints that can move: a model's joint names, or in the conversion report the
    USD paths of the joint prims.
    """

    SEGMENTS = "segments"
    """
    The labelled time spans of a recording.
    """

    MISSING_ASSETS = "missingAssets"
    """
    Files the scene refers to that were not found.
    """

    SOURCE = "source"
    """
    Where the scene came from.
    """

    USD = "usd"
    """
    The converted stage file.
    """

    CONVERTER = "converter"
    """
    The version of the converter.
    """

    LAYERS = "layers"
    """
    The digest of the layers and textures read.
    """

    FRAMES = "frames"
    """
    The joint values per frame.
    """

    BASE = "base"
    """
    The robot base pose per frame.
    """

    UNITS = "units"
    """
    The length unit.
    """

    UP_AXIS = "upAxis"
    """
    The axis pointing up.
    """

    PRIMS = "prims"
    """
    The semantic attributes per prim path.
    """

    LINK = "link"
    """
    The link a prim belongs to.
    """

    SUN = "sun"
    """
    The sun light.
    """

    DOME = "dome"
    """
    The dome light.
    """

    PANELS = "panels"
    """
    The rectangular area lights.
    """

    DIRECTION = "direction"
    """
    The way the sunlight travels.
    """

    INTENSITY = "intensity"
    """
    A light's intensity as authored, scaled by its exposure.
    """

    COLOR = "color"
    """
    A light's linear colour.
    """

    COLOR_TEMPERATURE = "colorTemperature"
    """
    A light's colour temperature in kelvin, or null.
    """

    CENTER = "center"
    """
    The middle of a panel.
    """

    NORMAL = "normal"
    """
    The way a panel emits.
    """

    WIDTH = "width"
    """
    A panel's width.
    """

    HEIGHT = "height"
    """
    A panel's height, or an object's extent along its Z axis.
    """

    COUNTS = "counts"
    """
    The named counts of the conversion.
    """

    SKIPPED = "skipped"
    """
    The number of prims left out, per reason.
    """

    SKIPPED_EXAMPLES = "skippedExamples"
    """
    A few prims left out, per reason.
    """

    EXCLUDED = "excluded"
    """
    The prims left out together with their subtrees.
    """

    PATH = "path"
    """
    A prim's USD path.
    """

    REASON = "reason"
    """
    Why a prim was left out.
    """

    DETACHED = "detached"
    """
    The bodies whose joint to a left-out parent body was dropped.
    """

    BODY = "body"
    """
    The USD path of a body.
    """

    PARENT_BODY = "parentBody"
    """
    The USD path of the body another body hung from.
    """

    JOINT = "joint"
    """
    The USD path of a joint prim.
    """

    ARTICULATIONS = "articulations"
    """
    The articulations with movable joints converted outside every marked robot.
    """

    UNMAPPED_INPUTS = "unmappedInputs"
    """
    The shader inputs no mapping carried over, per material.
    """

    MISSING_TEXTURES = "missingTextures"
    """
    Texture paths that did not resolve.
    """

    WARNINGS = "warnings"
    """
    Everything else worth a look.
    """

    READ_SECONDS = "readSeconds"
    """
    How long reading the stage took.
    """

    WRITE_SECONDS = "writeSeconds"
    """
    How long writing the bundle took.
    """

    BUNDLE_BYTES = "bundleBytes"
    """
    The size of the bundle on disk.
    """

    BASE_BODY = "baseBody"
    """
    The robot's root link, which the base pose places.
    """

    PARTS = "parts"
    """
    The robot's links grouped into parts.
    """

    PART_ANNOTATIONS = "partAnnotations"
    """
    The semantic annotations of the robot's parts.
    """

    ID = "id"
    """
    An object's identifier.
    """

    MESH = "mesh"
    """
    An object's bundle-relative mesh file.
    """

    SPAWN = "spawn"
    """
    An object's pose at the first frame.
    """

    STEP = "step"
    """
    What happened in a segment.
    """

    ACTION = "action"
    """
    The plan action a segment performed.
    """

    ARM = "arm"
    """
    The arm a segment used.
    """

    START = "start"
    """
    A segment's first frame.
    """

    END = "end"
    """
    The frame a segment ends at.
    """

    MODEL = "model"
    """
    The model or object a prim's link belongs to.
    """

    ANIMATION = "animation"
    """
    The summary of the playback in the conversion report.
    """

    FRAME_COUNT = "frameCount"
    """
    The number of playback frames.
    """

    LARGEST_BODY_DISTANCE = "largestBodyDistanceMillimetres"
    """
    The largest position error of a replayed body.
    """

    LARGEST_BODY_ANGLE = "largestBodyAngleDegrees"
    """
    The largest orientation error of a replayed body.
    """

    BEYOND_TOLERANCE = "beyondTolerance"
    """
    The bodies and joints whose residuals exceed the tolerance.
    """

    UNTRACKED_BODIES = "untrackedBodies"
    """
    Recorded bodies no link stands for.
    """


class SceneUnit(StrEnum):
    """
    The units and axes every bundle is written in.
    """

    METRES = "metres"
    """
    Lengths are in metres.
    """

    UP_AXIS = "Z"
    """
    Z points up.
    """


# %% presentation


@dataclass
class SceneCamera:
    """
    The viewer's initial camera.
    """

    position: numpy.ndarray
    """
    Where the camera stands, in metres.
    """

    target: numpy.ndarray
    """
    The point the camera looks at, in metres.
    """

    field_of_view: float = 50.0
    """
    The vertical field of view in degrees.
    """

    far: float | None = None
    """
    The far clipping distance in metres, or None for twice the scene's diagonal.
    """

    @classmethod
    def framing(
        cls,
        focus: Bounds,
        scene: Bounds,
        view_from: tuple[float, float, float] = (-0.55, -0.75, 0.45),
        minimum_far: float = 50.0,
    ) -> Self:
        """
        A camera seeing the whole focus box from the front left and above, with a far
        plane beyond the whole scene.

        :param focus: The box to frame.
        :param scene: The box holding all geometry.
        :param view_from: The direction from the target towards the camera.
        :param minimum_far: The smallest far clipping distance in metres.
        """
        camera = cls(
            focus.center, focus.center, far=max(2.0 * scene.diagonal, minimum_far)
        )
        direction = numpy.asarray(view_from) / numpy.linalg.norm(view_from)
        half_angle = numpy.radians(camera.field_of_view) / 2.0
        camera.position = focus.center + direction * (focus.diagonal / 2.0) / numpy.tan(
            half_angle
        )
        return camera

    @classmethod
    def of_camera(cls, camera: ImportedCamera, focus: Bounds) -> Self:
        """
        The view of a camera authored on the stage, looking at the point of its
        viewing ray closest to the centre of the focus box.

        :param camera: The authored camera.
        :param focus: The box the scene's geometry is densest in.
        """
        position = camera.pose.translation
        distance = max(float((focus.center - position) @ camera.forward), 1.0)
        return cls(
            position,
            position + camera.forward * distance,
            camera.vertical_field_of_view,
        )

    def to_json(self) -> dict[str, Any]:
        """
        The ``camera`` entry of ``scene.json``.
        """
        return {
            BundleKey.POSITION: [round(float(value), 4) for value in self.position],
            BundleKey.TARGET: [round(float(value), 4) for value in self.target],
            BundleKey.FIELD_OF_VIEW: round(float(self.field_of_view), 3),
            BundleKey.FAR: round(float(self.far), 2),
        }


@dataclass
class SceneRendering:
    """
    The renderer settings of ``scene.json``.
    """

    shadow_bounds: Bounds
    """
    The box the key light's shadows must cover.
    """

    exposure: float
    """
    The tone mapping exposure.
    """

    ambient_occlusion: bool = False
    """
    Whether the viewer adds screen-space ambient occlusion.
    """

    ground_plane: bool = False
    """
    Whether the viewer adds its own floor; a converted stage brings its own.
    """

    def to_json(self) -> dict[str, Any]:
        """
        The ``rendering`` entry of ``scene.json``.
        """
        return {
            BundleKey.AMBIENT_OCCLUSION: self.ambient_occlusion,
            BundleKey.EXPOSURE: self.exposure,
            BundleKey.GROUND_PLANE: self.ground_plane,
            BundleKey.SHADOW_BOUNDS: self.shadow_bounds.to_json(),
            PresentationField.ENVIRONMENT: BundleFile.LIGHTING.value,
        }


@dataclass
class LightingDescription:
    """
    The lights of the stage as ``lighting.json`` describes them: the brightest sun,
    the brightest dome and every rectangle light as a panel. Directions are the way
    the light travels, in the bundle frame.
    """

    lights: list[ImportedLight]
    """
    The stage's lights.
    """

    def brightest(self, kind: LightKind) -> ImportedLight | None:
        """
        The brightest light of a kind, or None.

        :param kind: The light kind.
        """
        candidates = [light for light in self.lights if light.kind is kind]
        return max(candidates, key=lambda light: light.intensity, default=None)

    def to_json(self) -> dict[str, Any]:
        """
        The content of ``lighting.json``.
        """
        sun = self.brightest(LightKind.DISTANT)
        dome = self.brightest(LightKind.DOME)
        return {
            BundleKey.SUN: (
                None
                if sun is None
                else {
                    BundleKey.DIRECTION: self.rounded(sun.emission_direction),
                    BundleKey.INTENSITY: sun.intensity,
                    BundleKey.COLOR_TEMPERATURE: sun.color_temperature,
                    BundleKey.COLOR: list(sun.color),
                }
            ),
            BundleKey.DOME: (
                None
                if dome is None
                else {
                    BundleKey.COLOR: list(dome.color),
                    BundleKey.INTENSITY: dome.intensity,
                }
            ),
            BundleKey.PANELS: [
                {
                    BundleKey.CENTER: self.rounded(light.pose.translation),
                    BundleKey.NORMAL: self.rounded(light.emission_direction),
                    BundleKey.WIDTH: round(light.width, 4),
                    BundleKey.HEIGHT: round(light.height, 4),
                    BundleKey.INTENSITY: light.intensity,
                    BundleKey.COLOR_TEMPERATURE: light.color_temperature,
                    BundleKey.COLOR: list(light.color),
                }
                for light in self.lights
                if light.kind is LightKind.RECTANGLE
            ],
        }

    @staticmethod
    def rounded(vector: numpy.ndarray) -> list[float]:
        """
        A vector rounded to a tenth of a millimetre (or radian), without negative zeros.

        :param vector: The vector.
        """
        return [round(float(value), 4) + 0.0 for value in vector]


# %% trajectory


@dataclass
class Trajectory:
    """
    The content of ``trajectory.json``; all lists have one entry per frame.
    """

    frames_per_second: float
    """
    The playback rate.
    """

    frames: list[dict[str, float]]
    """
    Joint values by joint name, per frame.
    """

    base: list[list[float]]
    """
    The robot base pose ``[x, y, z, qx, qy, qz, qw]`` per frame.
    """

    objects: list[dict[str, list[float]]]
    """
    Free object poses by object key, per frame.
    """

    @classmethod
    def static(
        cls, joint_values: dict[str, float], frames_per_second: float = 1.0
    ) -> Self:
        """
        A single frame holding joint values.

        :param joint_values: The value of every movable joint.
        :param frames_per_second: The playback rate.
        """
        return cls(
            frames_per_second, [joint_values], [Pose().translation_quaternion()], [{}]
        )

    @classmethod
    def of_animation(cls, animation: SceneAnimation, decimals: int = 6) -> Self:
        """
        The frames of a recorded playback; without a robot the base stays at the
        origin.

        :param animation: The playback.
        :param decimals: The decimals kept of poses.
        """
        keys = list(animation.joint_values)
        columns = [animation.joint_values[key].tolist() for key in keys]
        frames = (
            [dict(zip(keys, row)) for row in zip(*columns)]
            if keys
            else [{} for _ in range(animation.frame_count)]
        )
        base = (
            [Pose().translation_quaternion()] * animation.frame_count
            if animation.base is None
            else animation.base.translation_quaternions(decimals)
        )
        tracks = {
            moving.key: moving.poses.translation_quaternions(decimals)
            for moving in animation.objects
        }
        objects = [
            {key: poses[frame] for key, poses in tracks.items()}
            for frame in range(animation.frame_count)
        ]
        return cls(animation.frames_per_second, frames, base, objects)

    def to_json(self) -> dict[str, Any]:
        """
        The content of ``trajectory.json``.
        """
        return {
            BundleKey.FRAMES_PER_SECOND: self.frames_per_second,
            BundleKey.FRAMES: self.frames,
            BundleKey.BASE: self.base,
            PresentationField.OBJECTS: self.objects,
        }


# %% bundle


@dataclass
class SceneBundle:
    """
    Writes one converted stage as a scene bundle and registers it in the scenes index.
    """

    description: SceneDescription
    """
    The converted stage.
    """

    name: str
    """
    The bundle's directory name and ``?scene=`` value.
    """

    scenes_directory: Path
    """
    The directory holding the bundles and ``index.json``.
    """

    environment_name: str = ""
    """
    The environment's name in the viewer's picker; the stage file's name when empty.
    """

    task: str = ""
    """
    The scene's description in the viewer's picker; the stage file's name when empty.
    """

    camera: SceneCamera | None = None
    """
    The initial camera, or None for the stage's first authored camera (else a view
    framing the dense core of the geometry).
    """

    exposure: float = 0.72
    """
    The tone mapping exposure. The viewer scales every light of ``lighting.json`` so
    that the lit ceiling shows at one, which leaves the exposure alone to set the
    brightness; this default matches Isaac Sim's renders of the lab twin.
    """

    trajectory: Trajectory | None = None
    """
    The frames to play back, or None for one frame of the initial joint values.
    """

    read_seconds: float = 0.0
    """
    How long reading the stage took, for the report.
    """

    mapper: MaterialMapper = field(default_factory=MaterialMapper)
    """
    The mapping of authored materials onto glTF materials.
    """

    environment_model: str = ModelName.ENVIRONMENT.value
    """
    The name of the environment model in ``scene.json``.
    """

    animation: SceneAnimation | None = None
    """
    The recorded playback with its robot and objects, or None for a static scene of
    one environment model.
    """

    object_color: str = "#ffffff"
    """
    The colour of the objects' labels in the viewer.
    """

    @property
    def directory(self) -> Path:
        """
        The bundle directory.
        """
        return self.scenes_directory / self.name

    @property
    def size(self) -> int:
        """
        The number of bytes the bundle's files take on disk.
        """
        return sum(
            path.stat().st_size for path in self.directory.rglob("*") if path.is_file()
        )

    def write(self) -> Path:
        """
        Write every file of the bundle and refresh the scenes index.
        """
        validate_scene_name(self.name)
        started = time.perf_counter()
        (self.directory / BundleFile.MESHES).mkdir(parents=True, exist_ok=True)
        self.write_meshes(self.mapped_materials())
        for model in self.models():
            self.robot_description(model).write(self.directory / self.urdf_file(model))
        self.write_json(
            BundleFile.TRAJECTORY, self.current_trajectory().to_json(), compact=True
        )
        if self.animation is not None:
            self.animation.write_residuals(self.directory / BundleFile.RESIDUALS)
        self.write_json(BundleFile.SEMANTICS, self.semantics())
        self.write_json(
            BundleFile.LIGHTING, LightingDescription(self.description.lights).to_json()
        )
        self.write_json(BundleFile.SCENE, self.scene())
        self.write_json(BundleFile.REPORT, self.report(time.perf_counter() - started))
        write_scene_index(self.scenes_directory / BundleFile.INDEX, self.name)
        return self.directory

    def mapped_materials(self) -> dict[str, PhysicallyBasedMaterial]:
        """
        The glTF material of every authored material, recording unmapped inputs.
        """
        mapped = {
            name: self.mapper.map(material)
            for name, material in self.description.materials.items()
        }
        self.description.report.unmapped_inputs = {
            name: list(material.unmapped_inputs)
            for name, material in sorted(mapped.items())
            if material.unmapped_inputs
        }
        return mapped

    def write_meshes(self, materials: dict[str, PhysicallyBasedMaterial]) -> None:
        """
        Write one GLB per link with geometry of every model and per object, and remove
        GLBs a previous conversion left for links that no longer exist.

        :param materials: The glTF material of every authored material.
        """
        semantics = {item.path: item.attributes for item in self.description.semantics}
        written = set()
        for model in self.models():
            description = self.robot_description(model)
            for link in model.links:
                if not link.meshes:
                    continue
                path = self.directory / description.mesh_filename(link)
                written.add(path)
                LinkGltf(link, materials, semantics).write(path)
        for moving in self.objects():
            path = self.directory / self.object_mesh(moving.key)
            written.add(path)
            LinkGltf(moving.link, materials, semantics).write(path)
        meshes = self.directory / BundleFile.MESHES
        for stale in set(meshes.rglob(f"*{MeshFormat.GLB.value}")):
            if stale not in written:
                stale.unlink()

    def models(self) -> list[KinematicModel]:
        """
        The URDF models of the bundle: the animation's, else one environment model of
        every link.
        """
        if self.animation is not None:
            return self.animation.models
        return [
            KinematicModel(
                self.environment_model,
                self.description.links,
                self.description.joints,
            )
        ]

    def robot_description(self, model: KinematicModel) -> RobotDescription:
        """
        The URDF of a model, which also names the GLB of each of its links.

        :param model: The model.
        """
        return RobotDescription(
            model.name if model.robot else self.name,
            model.links,
            model.joints,
            mesh_directory=self.mesh_directory(model),
        )

    def objects(self) -> list[AnimatedObject]:
        """
        The objects played back by their poses.
        """
        return [] if self.animation is None else self.animation.objects

    def urdf_file(self, model: KinematicModel) -> str:
        """
        The bundle-relative URDF file of a model.

        :param model: The model.
        """
        return f"{model.name}.urdf" if model.robot else BundleFile.ENVIRONMENT.value

    def mesh_directory(self, model: KinematicModel) -> str:
        """
        The bundle-relative directory of a model's GLBs.

        :param model: The model.
        """
        if model.robot:
            return f"{BundleFile.MESHES.value}/{model.name}"
        return BundleFile.MESHES.value

    @staticmethod
    def object_mesh(key: str) -> str:
        """
        The bundle-relative GLB of an object.

        :param key: The object's key.
        """
        return f"{BundleFile.OBJECT_MESHES.value}/{key}{MeshFormat.GLB.value}"

    def current_trajectory(self) -> Trajectory:
        """
        The trajectory to write: the given one, else the animation's frames, else one
        frame of initial values.
        """
        if self.trajectory is not None:
            return self.trajectory
        if self.animation is not None:
            return Trajectory.of_animation(self.animation)
        return Trajectory.static(
            {
                joint.name: joint.initial_value
                for joint in self.description.movable_joints
            }
        )

    def scene_camera(self) -> SceneCamera:
        """
        The given camera, else the stage's first authored camera, else one framing the
        dense core of the geometry; the far plane reaches beyond the whole scene unless
        given.
        """
        framing = SceneCamera.framing(
            self.focus_bounds(), self.description.bounds or self.focus_bounds()
        )
        camera = self.camera
        if camera is None and self.description.cameras:
            camera = SceneCamera.of_camera(
                self.description.cameras[0], self.focus_bounds()
            )
        if camera is None:
            return framing
        if camera.far is None:
            return replace(camera, far=framing.far)
        return camera

    def focus_bounds(self) -> Bounds:
        """
        The dense core of the geometry, or a unit box at the origin without any.
        """
        if self.description.central_bounds is not None:
            return self.description.central_bounds
        return Bounds(numpy.full(3, -0.5), numpy.full(3, 0.5))

    def scene(self) -> dict[str, Any]:
        """
        The content of ``scene.json``.
        """
        stage_name = self.description.source.name
        trajectory = self.current_trajectory()
        return {
            BundleKey.NAME: self.name,
            SceneField.ENVIRONMENT_NAME: self.environment_name or stage_name,
            SceneField.TASK: self.task or stage_name,
            BundleKey.FRAMES_PER_SECOND: trajectory.frames_per_second,
            BundleKey.TRAJECTORY: BundleFile.TRAJECTORY.value,
            PresentationField.RENDERING: SceneRendering(
                self.focus_bounds(), self.exposure
            ).to_json(),
            PresentationField.CAMERA: self.scene_camera().to_json(),
            PresentationField.MODELS: [
                self.model_entry(model) for model in self.models()
            ],
            PresentationField.ROBOT: self.robot_entry(),
            PresentationField.OBJECTS: [
                self.object_entry(moving, trajectory) for moving in self.objects()
            ],
            BundleKey.SEGMENTS: [
                {
                    BundleKey.STEP: segment.step,
                    BundleKey.ACTION: None,
                    BundleKey.ARM: None,
                    BundleKey.START: segment.start,
                    BundleKey.END: segment.end,
                }
                for segment in (
                    [] if self.animation is None else self.animation.segments
                )
            ],
            BundleKey.MISSING_ASSETS: list(self.description.report.missing_textures),
            BundleKey.SOURCE: {
                BundleKey.USD: str(self.description.source),
                BundleKey.CONVERTER: __version__,
                BundleKey.LAYERS: self.source_digest(),
            },
        }

    def model_entry(self, model: KinematicModel) -> dict[str, Any]:
        """
        The ``models`` entry of one model.

        :param model: The model.
        """
        return {
            BundleKey.NAME: model.name,
            BundleKey.URDF: self.urdf_file(model),
            BundleKey.PREFIX: model.prefix,
            PresentationField.ROBOT: model.robot,
            BundleKey.LINKS: len(model.links),
            BundleKey.MOVABLE_JOINTS: [joint.name for joint in model.movable_joints],
            PresentationField.PRESERVE_MATERIALS: True,
        }

    def robot_entry(self) -> dict[str, Any] | None:
        """
        The ``robot`` entry: the robot model's name, prefix, base link and parts, or
        None without a robot.
        """
        robot: RobotModel | None = (
            None if self.animation is None else self.animation.robot
        )
        if robot is None:
            return None
        return {
            BundleKey.NAME: robot.name,
            BundleKey.PREFIX: robot.prefix,
            BundleKey.BASE_BODY: robot.base_link,
            BundleKey.PARTS: robot.parts,
            BundleKey.PART_ANNOTATIONS: [],
        }

    def object_entry(
        self, moving: AnimatedObject, trajectory: Trajectory
    ) -> dict[str, Any]:
        """
        The ``objects`` entry of one object, spawned at its pose in the first frame.

        :param moving: The object.
        :param trajectory: The trajectory holding the object's poses.
        """
        return {
            BundleKey.ID: moving.key,
            PresentationField.KEY: moving.key,
            BundleKey.MESH: self.object_mesh(moving.key),
            BundleKey.SPAWN: trajectory.objects[0][moving.key],
            BundleKey.COLOR: self.object_color,
            BundleKey.HEIGHT: round(moving.height, 4),
            PresentationField.PRESERVE_MATERIALS: True,
        }

    def source_digest(self) -> str:
        """
        The SHA-256 over the names and contents of every layer and texture read.
        """
        digest = hashlib.sha256()
        for path in sorted(self.description.used_files):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def semantics(self) -> dict[str, Any]:
        """
        The content of ``semantics.json``.
        """
        return {
            BundleKey.UNITS: SceneUnit.METRES.value,
            BundleKey.UP_AXIS: SceneUnit.UP_AXIS.value,
            BundleKey.SOURCE: str(self.description.source),
            BundleKey.PRIMS: {
                item.path: {**self.prim_placement(item.link), **item.attributes}
                for item in self.description.semantics
            },
        }

    def prim_placement(self, link: str) -> dict[str, str]:
        """
        The link a prim belongs to, and with an animation the model (or object)
        holding that link.

        :param link: The converted name of the prim's link.
        """
        if self.animation is None:
            return {BundleKey.LINK: link}
        placement = self.animation.placements.get(link, LinkPlacement("", link))
        return {BundleKey.MODEL: placement.model, BundleKey.LINK: placement.link}

    def report(self, write_seconds: float) -> dict[str, Any]:
        """
        The content of ``conversion_report.json``.

        :param write_seconds: How long writing the bundle took.
        """
        report = self.description.report
        return {
            BundleKey.SOURCE: str(self.description.source),
            BundleKey.CONVERTER: __version__,
            BundleKey.READ_SECONDS: round(self.read_seconds, 3),
            BundleKey.WRITE_SECONDS: round(write_seconds, 3),
            BundleKey.BUNDLE_BYTES: self.size,
            BundleKey.COUNTS: dict(sorted(report.counts.items())),
            BundleKey.SKIPPED: dict(sorted(report.skipped.items())),
            BundleKey.SKIPPED_EXAMPLES: report.skipped_examples,
            BundleKey.EXCLUDED: [
                {BundleKey.PATH: excluded.path, BundleKey.REASON: excluded.reason}
                for excluded in report.excluded
            ],
            BundleKey.DETACHED: [
                {
                    BundleKey.BODY: detached.body,
                    BundleKey.PARENT_BODY: detached.parent_body,
                    BundleKey.JOINT: detached.joint,
                    BundleKey.MOVABLE_JOINTS: detached.movable_joints,
                }
                for detached in report.detached
            ],
            BundleKey.ARTICULATIONS: [
                {
                    BundleKey.PATH: articulation.path,
                    BundleKey.MOVABLE_JOINTS: articulation.movable_joints,
                }
                for articulation in report.articulations
            ],
            BundleKey.UNMAPPED_INPUTS: report.unmapped_inputs,
            BundleKey.MISSING_TEXTURES: report.missing_textures,
            BundleKey.WARNINGS: report.warnings,
            BundleKey.ANIMATION: self.animation_summary(),
        }

    def animation_summary(self) -> dict[str, Any] | None:
        """
        The conversion report's summary of the playback, or None without an animation.
        """
        animation = self.animation
        if animation is None:
            return None
        return {
            BundleKey.SOURCE: animation.source,
            BundleKey.FRAME_COUNT: animation.frame_count,
            BundleKey.FRAMES_PER_SECOND: animation.frames_per_second,
            BundleKey.LARGEST_BODY_DISTANCE: round(
                1e3
                * max(
                    (
                        residual.largest_distance
                        for residual in animation.body_residuals
                    ),
                    default=0.0,
                ),
                4,
            ),
            BundleKey.LARGEST_BODY_ANGLE: round(
                float(
                    numpy.degrees(
                        max(
                            (
                                residual.largest_angle
                                for residual in animation.body_residuals
                            ),
                            default=0.0,
                        )
                    )
                ),
                4,
            ),
            BundleKey.BEYOND_TOLERANCE: [
                residual.path for residual in animation.beyond_tolerance()
            ],
            PresentationField.OBJECTS: [moving.key for moving in animation.objects],
            BundleKey.UNTRACKED_BODIES: animation.untracked_bodies,
        }

    def write_json(
        self, file: BundleFile, payload: dict[str, Any], compact: bool = False
    ) -> None:
        """
        Write one JSON file of the bundle.

        :param file: The bundle file.
        :param payload: The content.
        :param compact: Whether to leave out all optional whitespace, for large files.
        """
        text = (
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            if compact
            else json.dumps(payload, indent=1, ensure_ascii=False)
        )
        (self.directory / file).write_text(text + "\n", encoding="utf-8")
