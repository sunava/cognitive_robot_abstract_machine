"""
``cramera-import``: convert a USD stage (an Isaac Sim twin) into a CRAMERA scene bundle.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy
from typing_extensions import Self

from cramera import paths
from cramera.onboard.usd_scene.animation import (
    BodyTrace,
    RecordedRun,
    SceneAnimation,
    SceneAnimator,
)
from cramera.onboard.usd_scene.bundle import SceneBundle, SceneCamera
from cramera.onboard.usd_scene.bundle_layout import BundleFile
from cramera.onboard.usd_scene.scene_description import (
    ConversionReport,
    JointKind,
    ReportCount,
    SceneDescription,
    SemanticAttributeName,
    SemanticNamespace,
)
from cramera.onboard.usd_scene.stage_reader import (
    PrimExclusion,
    RobotMarker,
    StageReader,
    VariantSelection,
)

# %% options


class ImportOption(StrEnum):
    """
    The command line options of ``cramera-import``.
    """

    NAME = "--name"
    """
    The bundle name.
    """

    OUT = "--out"
    """
    The scenes directory receiving the bundle.
    """

    PRIM = "--prim"
    """
    A prim whose subtree is converted; repeatable.
    """

    EXCLUDE = "--exclude"
    """
    A prim whose subtree is left out; repeatable.
    """

    ENVIRONMENT_NAME = "--environment-name"
    """
    The environment's name in the viewer's picker.
    """

    TASK = "--task"
    """
    The scene's description in the viewer's picker.
    """

    EXPOSURE = "--exposure"
    """
    The viewer's tone mapping exposure.
    """

    CAMERA_POSITION = "--camera-position"
    """
    The initial camera position.
    """

    CAMERA_TARGET = "--camera-target"
    """
    The point the initial camera looks at.
    """

    CAMERA_FIELD_OF_VIEW = "--camera-fov"
    """
    The vertical field of view of the initial camera in degrees.
    """

    TRACE = "--trace"
    """
    A recorded body trace to play back instead of the stage's time samples.
    """

    FRAMES_PER_SECOND = "--fps"
    """
    The playback rate.
    """

    ROBOT = "--robot"
    """
    The prim of the robot's root link, converted as a model of its own.
    """

    KEEP_ROBOTS = "--keep-robots"
    """
    Keep the robots the stage marks in the environment instead of leaving them out.
    """

    VARIANT = "--variant"
    """
    A variant selected in the stage's session layer; repeatable.
    """

    RECORD = "--record"
    """
    A run record whose steps become the playback's segments.
    """


# %% command


@dataclass
class ImportCommand:
    """
    One run of ``cramera-import``.
    """

    stage: Path
    """
    The stage file to convert.
    """

    name: str
    """
    The bundle name.
    """

    scenes_directory: Path
    """
    The directory holding the bundles and ``index.json``.
    """

    prim_paths: list[str] = field(default_factory=list)
    """
    The prims whose subtrees are converted; empty for the whole stage.
    """

    excluded_paths: list[str] = field(default_factory=list)
    """
    The prims whose subtrees are left out.
    """

    keep_robots: bool = False
    """
    Whether the robots the stage marks stay in the environment.
    """

    environment_name: str = ""
    """
    The environment's name in the viewer's picker.
    """

    task: str = ""
    """
    The scene's description in the viewer's picker.
    """

    exposure: float | None = None
    """
    The viewer's tone mapping exposure, or None for the bundle's default.
    """

    camera: SceneCamera | None = None
    """
    The initial camera, or None to frame the geometry.
    """

    trace: Path | None = None
    """
    The recorded body trace to play back, or None to sample the stage's time samples.
    """

    frames_per_second: float = 30.0
    """
    The playback rate.
    """

    robot_path: str | None = None
    """
    The prim of the robot's root link, or None for a scene without a robot.
    """

    variant_selections: list[VariantSelection] = field(default_factory=list)
    """
    The variants selected in the stage's session layer.
    """

    record: Path | None = None
    """
    The run record whose steps label the playback, or None.
    """

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        """
        The argument parser of ``cramera-import``.
        """
        parser = argparse.ArgumentParser(
            prog="cramera-import",
            description="Convert a USD stage (.usd, .usda, .usdc) into a CRAMERA scene "
            "bundle of its environment: robots the stage marks are left out unless "
            f"{ImportOption.ROBOT} or {ImportOption.KEEP_ROBOTS} keeps them.",
        )
        parser.add_argument("stage", type=Path, help="the stage file to convert")
        parser.add_argument(
            ImportOption.NAME,
            required=True,
            help="bundle name (letters, digits, _ and -)",
        )
        parser.add_argument(
            ImportOption.OUT,
            type=Path,
            default=None,
            help="scenes directory (default: CRAMERA_SCENES or the local scenes directory)",
        )
        parser.add_argument(
            ImportOption.PRIM,
            action="append",
            default=[],
            help="convert only this prim's subtree; repeatable (default: the whole stage)",
        )
        parser.add_argument(
            ImportOption.EXCLUDE,
            action="append",
            default=[],
            metavar="PRIM",
            help="leave out this prim's subtree; repeatable",
        )
        parser.add_argument(
            ImportOption.ENVIRONMENT_NAME,
            default="",
            help="environment name in the picker",
        )
        parser.add_argument(
            ImportOption.TASK, default="", help="scene description in the picker"
        )
        parser.add_argument(
            ImportOption.EXPOSURE, type=float, help="tone mapping exposure"
        )
        parser.add_argument(
            ImportOption.CAMERA_POSITION, type=float, nargs=3, metavar=("X", "Y", "Z")
        )
        parser.add_argument(
            ImportOption.CAMERA_TARGET, type=float, nargs=3, metavar=("X", "Y", "Z")
        )
        parser.add_argument(
            ImportOption.CAMERA_FIELD_OF_VIEW,
            type=float,
            metavar="DEGREES",
            help="vertical field of view of the camera given by position and target",
        )
        parser.add_argument(
            ImportOption.TRACE,
            type=Path,
            help="recorded body trace (.npz: bodies, pos, quat w-first, sample_hz) to play back",
        )
        parser.add_argument(
            ImportOption.FRAMES_PER_SECOND,
            type=float,
            default=cls.frames_per_second,
            help="playback rate (default: %(default)s)",
        )
        parser.add_argument(
            ImportOption.ROBOT,
            metavar="PRIM",
            help="prim of the robot's root link, written as a robot model of its own "
            "and kept although the stage marks it as a robot",
        )
        parser.add_argument(
            ImportOption.KEEP_ROBOTS,
            action="store_true",
            help="keep the robots the stage marks (applied "
            f"{RobotMarker.ISAAC_ROBOT_SCHEMA}, or "
            + SemanticNamespace.SEMANTIC_DIGITAL_TWIN.attribute_name(
                SemanticAttributeName.ENTITY_TYPE
            )
            + f" {RobotMarker.ROBOT_ENTITY_TYPE}) in the environment; by default they "
            "are left out",
        )
        parser.add_argument(
            ImportOption.VARIANT,
            nargs=3,
            action="append",
            default=[],
            metavar=("PRIM", "SET", "VARIANT"),
            help="select a variant in the in-memory session layer; repeatable",
        )
        parser.add_argument(
            ImportOption.RECORD,
            type=Path,
            help="run record (JSON with steps) whose steps become the segments",
        )
        return parser

    @classmethod
    def from_arguments(cls, arguments: list[str] | None) -> Self:
        """
        The command a list of command line arguments asks for.

        :param arguments: The arguments without the program name, or None to read
            them from ``sys.argv``.
        """
        parser = cls.parser()
        options = parser.parse_args(arguments)
        camera = None
        if options.camera_position is not None and options.camera_target is not None:
            camera = SceneCamera(
                numpy.array(options.camera_position), numpy.array(options.camera_target)
            )
            if options.camera_fov is not None:
                camera.field_of_view = options.camera_fov
        elif (
            options.camera_position is not None
            or options.camera_target is not None
            or options.camera_fov is not None
        ):
            parser.error(
                f"{ImportOption.CAMERA_POSITION} and {ImportOption.CAMERA_TARGET} "
                f"give the camera together, and {ImportOption.CAMERA_FIELD_OF_VIEW} "
                "only goes with both"
            )
        return cls(
            stage=options.stage,
            name=options.name,
            scenes_directory=options.out or paths.scenes_directory(),
            prim_paths=options.prim,
            excluded_paths=options.exclude,
            keep_robots=options.keep_robots,
            environment_name=options.environment_name,
            task=options.task,
            exposure=options.exposure,
            camera=camera,
            trace=options.trace,
            frames_per_second=options.fps,
            robot_path=options.robot,
            variant_selections=[
                VariantSelection(*selection) for selection in options.variant
            ],
            record=options.record,
        )

    def run(self) -> SceneBundle:
        """
        Read the stage and write the bundle.
        """
        started = time.perf_counter()
        reader = StageReader(
            self.stage,
            self.prim_paths,
            self.variant_selections,
            PrimExclusion(self.excluded_paths, self.keep_robots, self.robot_path),
        )
        description = reader.read()
        bundle = SceneBundle(
            description,
            self.name,
            self.scenes_directory,
            environment_name=self.environment_name,
            task=self.task,
            camera=self.camera,
            animation=self.animation(reader, description),
            read_seconds=time.perf_counter() - started,
        )
        if self.exposure is not None:
            bundle.exposure = self.exposure
        bundle.write()
        return bundle

    def animation(
        self, reader: StageReader, description: SceneDescription
    ) -> SceneAnimation | None:
        """
        The playback of the trace, else of the stage's time samples; None for a stage
        without moving links, recorded trace or robot.

        :param reader: The reader of the stage.
        :param description: The converted stage.
        """
        if self.trace is not None:
            trace = BodyTrace.from_file(self.trace).resampled(self.frames_per_second)
        else:
            trace = reader.motion(
                [link.path for link in description.links], self.frames_per_second
            )
        if self.trace is None and not trace.poses and self.robot_path is None:
            return None
        return SceneAnimator(
            description,
            trace,
            robot_path=self.robot_path,
            run=None if self.record is None else RecordedRun.from_file(self.record),
        ).animate()

    def summary(self, bundle: SceneBundle) -> str:
        """
        What was written, for the terminal.

        :param bundle: The written bundle.
        """
        description = bundle.description
        joints = {kind: 0 for kind in JointKind}
        for joint in description.joints:
            joints[joint.kind] += 1
        return "\n".join(
            [
                f"wrote {bundle.directory / BundleFile.SCENE}",
                f"  {len(description.links)} links, "
                + ", ".join(f"{count} {kind.value}" for kind, count in joints.items())
                + " joints",
                f"  {description.report.counts[ReportCount.MESHES]} meshes, "
                f"{description.report.counts[ReportCount.TRIANGLES]} triangles, "
                f"{len(description.materials)} materials, {bundle.size / 1e6:.1f} MB",
                f"  skipped: {dict(description.report.skipped)}",
                "  left out: "
                + (
                    ", ".join(
                        f"{excluded.path} ({excluded.reason})"
                        for excluded in description.report.excluded
                    )
                    or "-"
                ),
                *self.report_summary(description.report),
                *self.animation_summary(bundle.animation),
                f"  view it with `cramera` at index.html?scene={bundle.name}",
            ]
        )

    @classmethod
    def report_summary(cls, report: ConversionReport) -> list[str]:
        """
        What of the conversion report asks for a look: the bodies detached from
        left-out prims, the articulations kept in the environment and the warnings.

        :param report: The conversion report.
        """
        return [
            *cls.listed(
                "detached from left-out prims",
                [
                    f"{detached.body} from {detached.parent_body} "
                    f"(joint {detached.joint})"
                    + (
                        f", movable joints below it: {len(detached.movable_joints)}, "
                        f"a robot part? {ImportOption.EXCLUDE} it"
                        if detached.movable_joints
                        else ""
                    )
                    for detached in report.detached
                ],
            ),
            *cls.listed(
                "articulations kept in the environment "
                f"({ImportOption.EXCLUDE} a robot the stage does not mark)",
                [
                    f"{articulation.path}, movable joints: "
                    f"{len(articulation.movable_joints)}"
                    for articulation in report.articulations
                ],
            ),
            *cls.listed("warnings", report.warnings),
        ]

    @staticmethod
    def listed(title: str, items: list[str]) -> list[str]:
        """
        A titled list for the terminal, one item per line, or a dash when it is empty.

        :param title: What the items are.
        :param items: The items.
        """
        if not items:
            return [f"  {title}: -"]
        return [f"  {title}:", *(f"    {item}" for item in items)]

    @staticmethod
    def animation_summary(animation: SceneAnimation | None) -> list[str]:
        """
        What the playback holds and how well it reproduces the recording.

        :param animation: The playback, or None.
        """
        if animation is None:
            return []
        worst = max(
            animation.body_residuals,
            key=lambda residual: residual.largest_distance,
            default=None,
        )
        return [
            f"  playback: {animation.frame_count} frames at "
            f"{animation.frames_per_second:g} fps from {animation.source}, "
            f"{len(animation.joint_values)} joints, "
            f"robot {'none' if animation.robot is None else animation.robot.name}, "
            f"objects {[moving.key for moving in animation.objects]}",
            f"  replay: {len(animation.body_residuals)} bodies, worst "
            + (
                "-"
                if worst is None
                else f"{worst.largest_distance * 1e3:.3f} mm ({worst.path})"
            )
            + f", {len(animation.beyond_tolerance())} residuals beyond tolerance",
        ]


def main(arguments: list[str] | None = None) -> None:
    """
    ``cramera-import`` — convert a USD stage into a CRAMERA scene bundle.

    :param arguments: The command line arguments, or None to read them from
        ``sys.argv``.
    """
    command = ImportCommand.from_arguments(arguments)
    print(command.summary(command.run()))


if __name__ == "__main__":
    main()
