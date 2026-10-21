"""
Playback of recorded motion in a scene bundle: joint values recovered from the world
poses of jointed bodies, the robot as a model of its own with its base pose, moving free
bodies as objects, the recorded steps as segments, and the forward kinematics check that
replaying all of it reproduces every recorded body.

Nothing here imports ``pxr``: a trace file is read with numpy, and the stage reader
samples a stage's time samples into the same :class:`BodyTrace`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
from itertools import takewhile
from pathlib import Path

import numpy
from scipy.spatial.transform import Rotation
from typing_extensions import Any, Self

from cramera.onboard.usd_scene.bundle_layout import ModelName
from cramera.onboard.usd_scene.exceptions import (
    RobotRootNotFound,
    TraceIncomplete,
    TraceNotFound,
)
from cramera.onboard.usd_scene.scene_description import (
    GeneratedName,
    ImportedJoint,
    ImportedLink,
    Pose,
    PoseTrack,
    SceneDescription,
    UniqueNames,
)

# %% input vocabularies


class TraceArray(StrEnum):
    """
    The arrays of a body trace file, as the twin's recorder names them.
    """

    BODIES = "bodies"
    """
    The USD paths of the recorded bodies.
    """

    POSITIONS = "pos"
    """
    ``(frames, bodies, 3)`` world positions in metres.
    """

    QUATERNIONS = "quat"
    """
    ``(frames, bodies, 4)`` world orientations as quaternions, scalar first.
    """

    SAMPLE_RATE = "sample_hz"
    """
    The number of frames per second.
    """


class RecordKey(StrEnum):
    """
    The keys of a run record that the segments are read from.
    """

    STEPS = "steps"
    """
    The completed steps, in order.
    """

    STEP = "step"
    """
    A step's name.
    """

    TIME = "t"
    """
    When a step completed, in seconds on the recorder's clock.
    """

    START = "t0_rig_s"
    """
    The recorder's clock time of the trace's first frame.
    """


# %% pose deviations


@dataclass(frozen=True)
class ResidualTolerance:
    """
    How far a replayed pose may stray from a recorded one.
    """

    distance: float = 0.001
    """
    The largest position error in metres.
    """

    angle: float = float(numpy.radians(0.1))
    """
    The largest orientation error in radians.
    """


@dataclass(frozen=True, eq=False)
class PoseDeviation:
    """
    How far one pose track is from another, frame by frame.
    """

    distances: numpy.ndarray
    """
    The distance between the two origins per frame, in metres.
    """

    angles: numpy.ndarray
    """
    The angle of the rotation between the two orientations per frame, in radians.
    """

    @classmethod
    def between(cls, expected: PoseTrack, actual: PoseTrack) -> Self:
        """
        The deviation of one track from another.

        :param expected: The reference poses.
        :param actual: The poses compared with them.
        """
        relative = expected.inverse() @ actual
        return cls(
            numpy.linalg.norm(relative.translations, axis=1),
            Rotation.from_matrix(relative.rotations).magnitude(),
        )

    @property
    def largest_distance(self) -> float:
        """
        The largest distance over all frames.
        """
        return float(self.distances.max(initial=0.0))

    @property
    def largest_angle(self) -> float:
        """
        The largest angle over all frames.
        """
        return float(self.angles.max(initial=0.0))

    def worst_frame(self, tolerance: ResidualTolerance) -> int:
        """
        The frame deviating most, distances and angles each measured in units of
        their tolerance.

        :param tolerance: The tolerance scaling distances and angles.
        """
        return int(
            numpy.argmax(
                numpy.maximum(
                    self.distances / tolerance.distance, self.angles / tolerance.angle
                )
            )
        )

    def within(self, tolerance: ResidualTolerance) -> bool:
        """
        Whether every frame lies within a tolerance.

        :param tolerance: The tolerance.
        """
        return (
            self.largest_distance <= tolerance.distance
            and self.largest_angle <= tolerance.angle
        )


# %% recorded motion


@dataclass
class BodyTrace:
    """
    World poses of rigid bodies at a fixed rate, by USD path.
    """

    frames_per_second: float
    """
    The number of frames per second.
    """

    poses: dict[str, PoseTrack] = field(default_factory=dict)
    """
    The pose track of every recorded body.
    """

    frame_count: int = 1
    """
    The number of frames.
    """

    source: str = ""
    """
    Where the poses were recorded or sampled.
    """

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """
        A trace as the twin's recorder writes it: a numpy archive of body paths,
        positions, scalar-first quaternions and the sample rate.

        :param path: The ``.npz`` file.
        """
        if not path.is_file():
            raise TraceNotFound(path)
        arrays = numpy.load(path, allow_pickle=False)
        missing = [name.value for name in TraceArray if name.value not in arrays.files]
        if missing:
            raise TraceIncomplete(path, missing)
        positions = arrays[TraceArray.POSITIONS.value]
        quaternions = arrays[TraceArray.QUATERNIONS.value][..., [1, 2, 3, 0]]
        return cls(
            float(arrays[TraceArray.SAMPLE_RATE.value]),
            {
                str(body): PoseTrack.of_positions(
                    positions[:, index], quaternions[:, index]
                )
                for index, body in enumerate(arrays[TraceArray.BODIES.value])
            },
            len(positions),
            str(path),
        )

    def resampled(self, frames_per_second: float) -> Self:
        """
        The trace at another rate, each new frame taking the nearest recorded frame
        and the duration kept.

        :param frames_per_second: The new rate.
        """
        duration = (self.frame_count - 1) / self.frames_per_second
        count = int(numpy.floor(duration * frames_per_second + 1e-9)) + 1
        frames = numpy.minimum(
            numpy.rint(
                numpy.arange(count) * self.frames_per_second / frames_per_second
            ).astype(int),
            self.frame_count - 1,
        )
        return replace(
            self,
            frames_per_second=frames_per_second,
            poses={path: track.at(frames) for path, track in self.poses.items()},
            frame_count=count,
        )


@dataclass(frozen=True)
class Segment:
    """
    One labelled stretch of the playback, as the viewer's timeline shows it.
    """

    step: str
    """
    What happened in the stretch.
    """

    start: int
    """
    The first frame of the stretch.
    """

    end: int
    """
    The frame the stretch ends at, where the next one starts.
    """


@dataclass(frozen=True)
class RecordedStep:
    """
    One completed step of a recorded run.
    """

    name: str
    """
    What the step did.
    """

    end: float
    """
    When the step completed, in seconds on the recorder's clock.
    """


@dataclass
class RecordedRun:
    """
    The steps of a recorded run, read from the run record the twin's recorder writes
    beside the trace.
    """

    steps: list[RecordedStep]
    """
    The steps in the order they completed.
    """

    start: float = 0.0
    """
    The recorder's clock time of the trace's first frame.
    """

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        """
        The run a record describes.

        :param payload: The parsed run record.
        """
        return cls(
            [
                RecordedStep(step[RecordKey.STEP], float(step[RecordKey.TIME]))
                for step in payload[RecordKey.STEPS]
            ],
            float(payload.get(RecordKey.START, 0.0)),
        )

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """
        The run a record file describes.

        :param path: The JSON run record.
        """
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))

    def segments(self, frames_per_second: float, frame_count: int) -> list[Segment]:
        """
        One segment per step completed after the trace began, from the completion of
        the step before it (or the first frame) to its own completion.

        :param frames_per_second: The playback rate.
        :param frame_count: The number of playback frames.
        """
        segments = []
        previous = 0.0
        for step in self.steps:
            end = step.end - self.start
            if end < 0.0:
                continue
            segments.append(
                Segment(
                    step.name,
                    self.frame_at(previous, frames_per_second, frame_count),
                    self.frame_at(end, frames_per_second, frame_count),
                )
            )
            previous = max(previous, end)
        return segments

    @staticmethod
    def frame_at(seconds: float, frames_per_second: float, frame_count: int) -> int:
        """
        The playback frame nearest to a time since the trace began.

        :param seconds: The time since the first frame.
        :param frames_per_second: The playback rate.
        :param frame_count: The number of playback frames.
        """
        return min(max(round(seconds * frames_per_second), 0), frame_count - 1)


# %% kinematic tree


@dataclass
class LinkTree:
    """
    The links of a converted scene as the tree their joints form.
    """

    description: SceneDescription
    """
    The converted scene.
    """

    @cached_property
    def links(self) -> dict[str, ImportedLink]:
        """
        Every link by name.
        """
        return {link.name: link for link in self.description.links}

    @cached_property
    def links_by_path(self) -> dict[str, ImportedLink]:
        """
        Every link by the path of the prim it stands for.
        """
        return {link.path: link for link in self.description.links}

    @cached_property
    def parent_joints(self) -> dict[str, ImportedJoint]:
        """
        The joint holding each link except the root, by child link name.
        """
        return {joint.child: joint for joint in self.description.joints}

    @cached_property
    def children(self) -> dict[str, list[str]]:
        """
        The names of each link's child links, in tree order.
        """
        children: dict[str, list[str]] = {link: [] for link in self.links}
        for link in self.description.links:
            joint = self.parent_joints.get(link.name)
            if joint is not None:
                children[joint.parent].append(link.name)
        return children

    def subtree(self, name: str) -> list[str]:
        """
        A link and every link below it, parents before children.

        :param name: The link's name.
        """
        names = [name]
        for current in names:
            names += self.children[current]
        return names


@dataclass
class BodyPoses:
    """
    The world pose track of any prim a link stands for: recorded, carried along by a
    recorded ancestor prim, or at rest at its initial pose.
    """

    trace: BodyTrace
    """
    The recorded body poses.
    """

    initial: dict[str, Pose]
    """
    The initial rigid world pose of every link's prim, by path.
    """

    @classmethod
    def of_description(cls, trace: BodyTrace, description: SceneDescription) -> Self:
        """
        The body poses of a converted scene under a trace.

        :param trace: The recorded body poses.
        :param description: The converted scene.
        """
        return cls(
            trace,
            {
                link.path: link.pose @ link.prim_to_link.inverse()
                for link in description.links
            },
        )

    def of(self, path: str | None) -> PoseTrack:
        """
        The world pose track of a prim.

        :param path: The prim's path, or None for the world.
        """
        frame_count = self.trace.frame_count
        if path is None:
            return PoseTrack.constant(Pose(), frame_count)
        if path in self.trace.poses:
            return self.trace.poses[path]
        carrier = self.carrier_of(path)
        if carrier is None:
            return PoseTrack.constant(self.initial[path], frame_count)
        return self.trace.poses[carrier] @ (
            self.initial[carrier].inverse() @ self.initial[path]
        )

    def carrier_of(self, path: str) -> str | None:
        """
        The nearest recorded ancestor of a prim, or None.

        :param path: The prim's path.
        """
        ancestors = [
            recorded
            for recorded in self.trace.poses
            if path.startswith(recorded + "/") and recorded in self.initial
        ]
        return max(ancestors, key=len, default=None)


@dataclass(eq=False)
class JointMotion:
    """
    The values a movable joint takes over a trace, recovered from the world poses of
    its two bodies.
    """

    joint: ImportedJoint
    """
    The joint.
    """

    values: numpy.ndarray
    """
    The joint value per frame, in radians or metres; revolute values are kept within
    the joint's limits and continuous values free of jumps.
    """

    off_axis: PoseDeviation
    """
    The part of the bodies' relative motion the joint cannot make, per frame.
    """

    @classmethod
    def recover(
        cls, joint: ImportedJoint, parent_body: PoseTrack, child_body: PoseTrack
    ) -> Self:
        """
        The motion of a joint between two body pose tracks.

        :param joint: A movable joint.
        :param parent_body: The world poses of the joint's parent body.
        :param child_body: The world poses of the joint's child body.
        """
        relative = joint.relative_motion(parent_body, child_body)
        values = joint.values_of(relative)
        return cls(
            joint, values, PoseDeviation.between(joint.motions(values), relative)
        )


# %% bundle models


@dataclass
class KinematicModel:
    """
    One URDF model of a bundle: links and the joints between them.
    """

    name: str
    """
    The model's name in ``scene.json``.
    """

    links: list[ImportedLink]
    """
    The links, the root first and parents before children.
    """

    joints: list[ImportedJoint]
    """
    One joint per link except the root.
    """

    prefix: str = ""
    """
    What the trajectory puts before the model's joint names.
    """

    @property
    def robot(self) -> bool:
        """
        Whether the model is the robot.
        """
        return False

    @property
    def movable_joints(self) -> list[ImportedJoint]:
        """
        The joints whose value can change, in tree order.
        """
        return [joint for joint in self.joints if joint.kind.movable]

    def key_of(self, joint: ImportedJoint) -> str:
        """
        The trajectory key of one of the model's joints.

        :param joint: The joint.
        """
        if not self.prefix:
            return joint.name
        return f"{self.prefix}/{joint.name}"


@dataclass
class RobotModel(KinematicModel):
    """
    The robot's model, placed in the world by the trajectory's base pose.
    """

    parts: dict[str, list[str]] = field(default_factory=dict)
    """
    The names of the robot's links, grouped by the prim they were assembled under.
    """

    @property
    def robot(self) -> bool:
        return True

    @property
    def base_link(self) -> str:
        """
        The name of the root link, whose world pose the base track holds.
        """
        return self.links[0].name


@dataclass(frozen=True)
class LinkPlacement:
    """
    Where a converted link ended up in the bundle.
    """

    model: str
    """
    The name of the model, or the key of the object, holding the link.
    """

    link: str
    """
    The link's name in that model.
    """


@dataclass(eq=False)
class AnimatedObject:
    """
    A link that moves on its own, played back by its world pose per frame.
    """

    key: str
    """
    The object's key in ``scene.json`` and the trajectory.
    """

    link: ImportedLink
    """
    The link, its geometry in the object frame.
    """

    poses: PoseTrack
    """
    The world pose of the object frame per frame.
    """

    @property
    def height(self) -> float:
        """
        The extent of the geometry along the object frame's Z axis.
        """
        heights = [numpy.ptp(mesh.points[:, 2]) for mesh in self.link.meshes]
        return float(max(heights, default=0.0))


@dataclass(eq=False)
class RobotNaming:
    """
    Short, readable names for a robot's links and joints: the names of their prims,
    numbered where they repeat.
    """

    tree: LinkTree
    """
    The links of the converted scene.
    """

    links: list[str]
    """
    The names of the robot's links, the root first.
    """

    @cached_property
    def link_names(self) -> dict[str, str]:
        """
        The robot's name of each link, by converted name.
        """
        return UniqueNames().claim_each(
            {name: self.tree.links[name].path.rsplit("/", 1)[-1] for name in self.links}
        )

    @cached_property
    def joint_names(self) -> dict[str, str]:
        """
        The robot's name of each joint, by converted name; a joint the converter
        generated is named after its child link.
        """
        return UniqueNames().claim_each(
            {
                joint.name: (
                    joint.path.rsplit("/", 1)[-1]
                    if joint.child_body is not None
                    else f"{self.link_names[joint.child]}_{GeneratedName.ATTACHMENT}"
                )
                for joint in self.joints
            }
        )

    @cached_property
    def joints(self) -> list[ImportedJoint]:
        """
        The joints inside the robot, in tree order.
        """
        return [self.tree.parent_joints[name] for name in self.links[1:]]

    def model(self) -> RobotModel:
        """
        The robot's model under its own names, named after its root prim.
        """
        name = self.link_names[self.links[0]].lower()
        links = [
            replace(self.tree.links[link], name=self.link_names[link])
            for link in self.links
        ]
        joints = [
            replace(
                joint,
                name=self.joint_names[joint.name],
                parent=self.link_names[joint.parent],
                child=self.link_names[joint.child],
            )
            for joint in self.joints
        ]
        return RobotModel(name, links, joints, prefix=name, parts=self.parts(name))

    def parts(self, robot_name: str) -> dict[str, list[str]]:
        """
        The robot's links grouped by the prim right below the prim all of them share;
        links standing for that shared prim itself form a part named after the robot.

        :param robot_name: The robot model's name.
        """
        paths = {name: self.tree.links[name].path.split("/") for name in self.links}
        shared = len(
            list(
                takewhile(
                    lambda elements: len(set(elements)) == 1, zip(*paths.values())
                )
            )
        )
        parts: dict[str, list[str]] = {}
        for name, elements in paths.items():
            part = elements[shared] if len(elements) > shared else robot_name
            parts.setdefault(part, []).append(self.link_names[name])
        return parts


# %% residuals


class ResidualKind(StrEnum):
    """
    What a residual measures.
    """

    BODY = "body"
    """
    How far a replayed body is from its recorded pose.
    """

    JOINT = "joint"
    """
    How much of the recorded relative motion of a joint's bodies the joint cannot make.
    """


class ResidualColumn(StrEnum):
    """
    The columns of the residuals table.
    """

    KIND = "kind"
    """
    The :class:`ResidualKind`.
    """

    PATH = "path"
    """
    The USD path of the body or joint.
    """

    NAME = "name"
    """
    The link or joint name in its model.
    """

    MODEL = "model"
    """
    The model or object holding the link or joint.
    """

    LARGEST_DISTANCE = "largest_distance_mm"
    """
    The largest position error in millimetres.
    """

    LARGEST_ANGLE = "largest_angle_deg"
    """
    The largest orientation error in degrees.
    """

    FRAME = "frame"
    """
    The playback frame deviating most.
    """

    WITHIN_TOLERANCE = "within_tolerance"
    """
    Whether every frame lies within the tolerance.
    """


@dataclass(eq=False)
class Residual:
    """
    How far the playback strays from the recording for one body or joint.
    """

    kind: ResidualKind
    """
    What is measured.
    """

    path: str
    """
    The USD path of the body or joint.
    """

    placement: LinkPlacement
    """
    The model and name of the link or joint in the bundle.
    """

    deviation: PoseDeviation
    """
    The error per frame.
    """

    tolerance: ResidualTolerance
    """
    The tolerance that picks the worst frame.
    """

    @property
    def largest_distance(self) -> float:
        """
        The largest position error in metres.
        """
        return self.deviation.largest_distance

    @property
    def largest_angle(self) -> float:
        """
        The largest orientation error in radians.
        """
        return self.deviation.largest_angle

    @property
    def frame(self) -> int:
        """
        The frame deviating most.
        """
        return self.deviation.worst_frame(self.tolerance)

    def within(self, tolerance: ResidualTolerance) -> bool:
        """
        Whether every frame lies within a tolerance.

        :param tolerance: The tolerance.
        """
        return self.deviation.within(tolerance)

    def row(self) -> dict[str, Any]:
        """
        The residual as a row of the residuals table.
        """
        return {
            ResidualColumn.KIND: self.kind.value,
            ResidualColumn.PATH: self.path,
            ResidualColumn.NAME: self.placement.link,
            ResidualColumn.MODEL: self.placement.model,
            ResidualColumn.LARGEST_DISTANCE: round(self.largest_distance * 1e3, 4),
            ResidualColumn.LARGEST_ANGLE: round(
                float(numpy.degrees(self.largest_angle)), 4
            ),
            ResidualColumn.FRAME: self.frame,
            ResidualColumn.WITHIN_TOLERANCE: self.within(self.tolerance),
        }


# %% scene animation


@dataclass(eq=False)
class SceneAnimation:
    """
    Everything a bundle needs to play a recording back, and how well it does.
    """

    frames_per_second: float
    """
    The playback rate.
    """

    frame_count: int
    """
    The number of playback frames.
    """

    models: list[KinematicModel]
    """
    The environment model, then the robot model if there is one.
    """

    joint_values: dict[str, numpy.ndarray]
    """
    The value per frame of every movable joint of the models, by trajectory key.
    """

    base: PoseTrack | None
    """
    The world pose of the robot's root link per frame, or None without a robot.
    """

    objects: list[AnimatedObject]
    """
    The links played back by their world poses.
    """

    segments: list[Segment]
    """
    The labelled stretches of the playback.
    """

    body_residuals: list[Residual]
    """
    How far each recorded body is replayed from where it was recorded.
    """

    joint_residuals: list[Residual]
    """
    How much recorded motion each movable joint cannot make.
    """

    placements: dict[str, LinkPlacement]
    """
    Where each converted link ended up, by converted link name; links without
    geometry that were turned into objects have none.
    """

    untracked_bodies: list[str] = field(default_factory=list)
    """
    Recorded bodies no link stands for.
    """

    tolerance: ResidualTolerance = field(default_factory=ResidualTolerance)
    """
    The tolerance the residuals are judged by.
    """

    source: str = ""
    """
    Where the body poses were recorded or sampled.
    """

    @property
    def robot(self) -> RobotModel | None:
        """
        The robot model, or None.
        """
        return next(
            (model for model in self.models if isinstance(model, RobotModel)), None
        )

    @property
    def residuals(self) -> list[Residual]:
        """
        The body residuals, then the joint residuals.
        """
        return self.body_residuals + self.joint_residuals

    def beyond_tolerance(self) -> list[Residual]:
        """
        The residuals exceeding the tolerance.
        """
        return [
            residual
            for residual in self.residuals
            if not residual.within(self.tolerance)
        ]

    def write_residuals(self, path: Path) -> None:
        """
        Write the residuals table as CSV.

        :param path: The destination file.
        """
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=[column for column in ResidualColumn]
            )
            writer.writeheader()
            for residual in self.residuals:
                writer.writerow(residual.row())


@dataclass
class Replay:
    """
    The world pose of every link when the recovered joint values are played back.
    """

    link_poses: dict[str, PoseTrack] = field(default_factory=dict)
    """
    The world pose of each link frame per frame, by converted link name.
    """

    detached: list[str] = field(default_factory=list)
    """
    The links whose recorded motion the kinematic tree does not reproduce, played back
    by their world poses instead, parents before children.
    """


@dataclass
class SceneAnimator:
    """
    Turns recorded body poses into the playback of a converted scene: joint values,
    the robot's base pose, objects and segments, checked by replaying them.
    """

    description: SceneDescription
    """
    The converted scene.
    """

    trace: BodyTrace
    """
    The recorded body poses, at the playback rate.
    """

    robot_path: str | None = None
    """
    The prim of the robot's root link (or a prim holding exactly one topmost link),
    or None for a scene without a robot.
    """

    run: RecordedRun | None = None
    """
    The recorded steps to label the playback with, or None.
    """

    tolerance: ResidualTolerance = field(default_factory=ResidualTolerance)
    """
    How far the replay may stray from the recording.
    """

    environment_name: str = ModelName.ENVIRONMENT.value
    """
    The environment model's name.
    """

    decimals: int = 6
    """
    The decimals kept of joint values and poses.
    """

    @cached_property
    def tree(self) -> LinkTree:
        """
        The links of the converted scene as a tree.
        """
        return LinkTree(self.description)

    @cached_property
    def body_poses(self) -> BodyPoses:
        """
        The world pose track of every link's prim.
        """
        return BodyPoses.of_description(self.trace, self.description)

    @cached_property
    def motions(self) -> dict[str, JointMotion]:
        """
        The recovered motion of every movable joint, values rounded to
        :attr:`decimals`, by converted joint name.
        """
        motions = {}
        for joint in self.description.movable_joints:
            motion = JointMotion.recover(
                joint,
                self.body_poses.of(joint.parent_body),
                self.body_poses.of(joint.child_body),
            )
            motions[joint.name] = replace(
                motion, values=numpy.round(motion.values, self.decimals) + 0.0
            )
        return motions

    @cached_property
    def robot_root(self) -> ImportedLink | None:
        """
        The robot's root link, or None without a robot.
        """
        if self.robot_path is None:
            return None
        exact = self.tree.links_by_path.get(self.robot_path)
        if exact is not None and exact.name in self.tree.parent_joints:
            return exact
        below = [
            link
            for link in self.description.links
            if link.path.startswith(self.robot_path + "/")
        ]
        names = {link.name for link in below}
        topmost = [
            link
            for link in below
            if self.tree.parent_joints[link.name].parent not in names
        ]
        if len(topmost) != 1:
            raise RobotRootNotFound(self.robot_path, [link.name for link in topmost])
        return topmost[0]

    @cached_property
    def robot_links(self) -> list[str]:
        """
        The names of the robot's links, the root first.
        """
        if self.robot_root is None:
            return []
        return self.tree.subtree(self.robot_root.name)

    @cached_property
    def robot_naming(self) -> RobotNaming | None:
        """
        The robot's own names of its links and joints, or None without a robot.
        """
        if self.robot_root is None:
            return None
        return RobotNaming(self.tree, self.robot_links)

    def animate(self) -> SceneAnimation:
        """
        Recover, replay and check the playback.
        """
        replay = self.replay()
        robot = None if self.robot_naming is None else self.robot_naming.model()
        objects = self.objects(replay)
        placements = self.placements(robot, objects, replay)
        environment = self.environment(replay)
        frame_count = self.trace.frame_count
        return SceneAnimation(
            frames_per_second=self.trace.frames_per_second,
            frame_count=frame_count,
            models=[environment] if robot is None else [environment, robot],
            joint_values=self.joint_values(environment, robot),
            base=None if robot is None else replay.link_poses[self.robot_root.name],
            objects=objects,
            segments=(
                []
                if self.run is None
                else self.run.segments(self.trace.frames_per_second, frame_count)
            ),
            body_residuals=self.body_residuals(replay, placements),
            joint_residuals=self.joint_residuals(placements),
            placements=placements,
            untracked_bodies=[
                path for path in self.trace.poses if path not in self.tree.links_by_path
            ],
            tolerance=self.tolerance,
            source=self.trace.source,
        )

    def replay(self) -> Replay:
        """
        Play the recovered joint values back from the root, the robot's root link from
        its base pose; an environment link whose recorded pose the tree misses, and
        everything below it, is detached and follows its recording.
        """
        replay = Replay()
        for link in self.description.links:
            joint = self.tree.parent_joints.get(link.name)
            if joint is None:
                replay.link_poses[link.name] = PoseTrack.constant(
                    Pose(), self.trace.frame_count
                )
                continue
            if self.robot_root is not None and link.name == self.robot_root.name:
                replay.link_poses[link.name] = self.base_poses(link, joint, replay)
                continue
            replay.link_poses[link.name] = self.child_poses(joint, replay)
            if link.name in self.robot_links:
                continue
            recorded = self.recorded_link_poses(link)
            missed = recorded is not None and not PoseDeviation.between(
                replay.link_poses[link.name], recorded
            ).within(self.tolerance)
            if joint.parent in replay.detached or missed:
                replay.detached.append(link.name)
            if missed:
                replay.link_poses[link.name] = recorded
        return replay

    def child_poses(self, joint: ImportedJoint, replay: Replay) -> PoseTrack:
        """
        The world poses of a joint's child link from its parent link's.

        :param joint: The joint.
        :param replay: The replay so far, holding the parent link.
        """
        values = (
            self.motions[joint.name].values
            if joint.kind.movable
            else numpy.zeros(self.trace.frame_count)
        )
        return (replay.link_poses[joint.parent] @ joint.origin) @ joint.motions(values)

    def base_poses(
        self, root: ImportedLink, joint: ImportedJoint, replay: Replay
    ) -> PoseTrack:
        """
        The world poses of the robot's root link: from the first recorded link rigidly
        attached to it (the root itself included), else from its parent link.

        :param root: The robot's root link.
        :param joint: The joint holding the root link.
        :param replay: The replay so far, holding the root's parent link.
        """
        offsets = {root.name: Pose()}
        for name in self.tree.subtree(root.name):
            if name not in offsets:
                continue
            recorded = self.recorded_link_poses(self.tree.links[name])
            if recorded is not None:
                return recorded @ offsets[name].inverse()
            for child in self.tree.children[name]:
                child_joint = self.tree.parent_joints[child]
                if not child_joint.kind.movable:
                    offsets[child] = offsets[name] @ child_joint.origin
        return self.child_poses(joint, replay)

    def recorded_link_poses(self, link: ImportedLink) -> PoseTrack | None:
        """
        The recorded world poses of a link frame, or None when its prim was not
        recorded.

        :param link: The link.
        """
        if link.path not in self.trace.poses:
            return None
        return self.trace.poses[link.path] @ link.prim_to_link

    def objects(self, replay: Replay) -> list[AnimatedObject]:
        """
        The detached links with geometry, keyed by their prim names (numbered where
        they repeat).

        :param replay: The replay.
        """
        detached = [
            self.tree.links[name]
            for name in replay.detached
            if self.tree.links[name].meshes
        ]
        keys = UniqueNames().claim_each(
            {link.name: link.path.rsplit("/", 1)[-1] for link in detached}
        )
        return [
            AnimatedObject(keys[link.name], link, replay.link_poses[link.name])
            for link in detached
        ]

    def environment(self, replay: Replay) -> KinematicModel:
        """
        The links and joints neither the robot nor detached.

        :param replay: The replay.
        """
        excluded = set(self.robot_links) | set(replay.detached)
        links = [link for link in self.description.links if link.name not in excluded]
        kept = {link.name for link in links}
        joints = [joint for joint in self.description.joints if joint.child in kept]
        return KinematicModel(self.environment_name, links, joints)

    def placements(
        self,
        robot: RobotModel | None,
        objects: list[AnimatedObject],
        replay: Replay,
    ) -> dict[str, LinkPlacement]:
        """
        Where each converted link ended up.

        :param robot: The robot model, or None.
        :param objects: The objects.
        :param replay: The replay.
        """
        placements = {
            link.name: LinkPlacement(self.environment_name, link.name)
            for link in self.description.links
            if link.name not in replay.detached
        }
        if robot is not None:
            for original, renamed in zip(self.robot_links, robot.links):
                placements[original] = LinkPlacement(robot.name, renamed.name)
        for moving in objects:
            placements[moving.link.name] = LinkPlacement(moving.key, moving.key)
        return placements

    def joint_values(
        self, environment: KinematicModel, robot: RobotModel | None
    ) -> dict[str, numpy.ndarray]:
        """
        The value per frame of every movable joint of the models, by trajectory key.

        :param environment: The environment model.
        :param robot: The robot model, or None.
        """
        values = {
            environment.key_of(joint): self.motions[joint.name].values
            for joint in environment.movable_joints
        }
        if robot is None:
            return values
        for original, renamed in zip(self.robot_naming.joints, robot.joints):
            if renamed.kind.movable:
                values[robot.key_of(renamed)] = self.motions[original.name].values
        return values

    def body_residuals(
        self, replay: Replay, placements: dict[str, LinkPlacement]
    ) -> list[Residual]:
        """
        How far each recorded body that a link stands for is replayed from its
        recording.

        :param replay: The replay.
        :param placements: Where each link ended up.
        """
        residuals = []
        for path, recorded in self.trace.poses.items():
            link = self.tree.links_by_path.get(path)
            if link is None:
                continue
            replayed = replay.link_poses[link.name] @ link.prim_to_link.inverse()
            residuals.append(
                Residual(
                    ResidualKind.BODY,
                    path,
                    placements.get(link.name, LinkPlacement("", link.name)),
                    PoseDeviation.between(recorded, replayed),
                    self.tolerance,
                )
            )
        return residuals

    def joint_residuals(self, placements: dict[str, LinkPlacement]) -> list[Residual]:
        """
        How much recorded motion each movable joint cannot make.

        :param placements: Where each link ended up.
        """
        names = {} if self.robot_naming is None else self.robot_naming.joint_names
        residuals = []
        for joint in self.description.movable_joints:
            child = placements.get(joint.child, LinkPlacement("", joint.child))
            residuals.append(
                Residual(
                    ResidualKind.JOINT,
                    joint.path,
                    LinkPlacement(child.model, names.get(joint.name, joint.name)),
                    self.motions[joint.name].off_axis,
                    self.tolerance,
                )
            )
        return residuals
