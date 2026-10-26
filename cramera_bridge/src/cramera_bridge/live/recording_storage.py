"""
Saving or discarding the ``__recording__`` bundle a finished recording was written to.

Everything is checked before anything is written: a refused save leaves the bundle and
the recording as they were.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from typing_extensions import List, Optional

from cramera.locations import DataLocations
from cramera.scenes import SceneSource
from cramera_bridge.documents import DocumentFile
from cramera_bridge.exceptions import (
    NoRecording,
    SceneNameTaken,
    SharedScenesUnavailable,
)
from cramera_bridge.export.scene_index import SceneIndexFile
from cramera_bridge.live.bundles import STAGING_PREFIX, STAGING_SUFFIX
from cramera_bridge.live.recording import FrameRange
from cramera_bridge.live.segments import PlanNodeClipping, SegmentClipping
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.bundle.index import ReservedSceneName, SceneIndexEntry, SceneName
from cramera_protocol.bundle.scene import SceneDescription
from cramera_protocol.bundle.statecharts import Statecharts
from cramera_protocol.bundle.trajectory import Trajectory
from cramera_protocol.live_messages import (
    RecordingSaveRequest,
    RecordingState,
    SavedScene,
    SceneDestination,
)

# %% storage


@dataclass(frozen=True)
class RecordingStorage:
    """
    The ``__recording__`` bundle in the local scenes directory and where it is saved.
    """

    locations: DataLocations
    """
    Where cramera keeps its scenes.
    """

    @property
    def bundle(self) -> Path:
        """
        The ``__recording__`` bundle directory.
        """
        return self.locations.local_scenes / ReservedSceneName.RECORDING

    def discard(self) -> None:
        """
        Delete the ``__recording__`` bundle, if there is one.
        """
        if self.bundle.exists():
            shutil.rmtree(self.bundle)

    def destination(self, destination: SceneDestination) -> Path:
        """
        :param destination: Where a saved recording goes.
        :return: The scenes directory.
        :raises SharedScenesUnavailable: For the shared directory while none other than
            the local one is configured.
        """
        if destination is SceneDestination.LOCAL:
            return self.locations.local_scenes
        shared = self.locations.shared_scenes
        if shared is None or shared.resolve() == self.locations.local_scenes.resolve():
            raise SharedScenesUnavailable()
        return shared

    def directories(self) -> List[Path]:
        """
        :return: Every scenes directory a saved name must not clash with.
        """
        shared = self.locations.shared_scenes
        return [self.locations.local_scenes] + ([] if shared is None else [shared])

    def save(self, request: RecordingSaveRequest) -> SavedScene:
        """
        Save the bundle under a name, trimmed to the requested frames.

        :param request: The save request.
        :return: The saved bundle's name.
        :raises InvalidSceneName: When the name is no valid bundle name.
        :raises SharedScenesUnavailable: For an unconfigured shared directory.
        :raises NoRecording: When there is no bundle to save.
        :raises SceneNameTaken: When a scenes directory holds the name already.
        :raises InvalidFrameRange: When the frames lie outside the recording.
        """
        name = SceneName(request.name).text
        target_directory = self.destination(request.destination)
        trajectory = DocumentFile(self.bundle / BundleFile.TRAJECTORY).read_if_present(
            Trajectory
        )
        if trajectory is None:
            raise NoRecording(RecordingState.FINALIZED)
        if any((directory / name).exists() for directory in self.directories()):
            raise SceneNameTaken(name)
        frame_range = FrameRange(
            first=0 if request.first_frame is None else request.first_frame,
            last=(
                trajectory.frame_count() - 1
                if request.last_frame is None
                else request.last_frame
            ),
        )
        frame_range.require_within(trajectory.frame_count())
        staging = target_directory / f"{STAGING_PREFIX}{name}{STAGING_SUFFIX}"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(self.bundle, staging)
        scene = SavedBundle(staging, frame_range).rewrite(name, request)
        staging.rename(target_directory / name)
        SceneIndexFile(
            target_directory,
            (
                SceneSource.LOCAL
                if request.destination is SceneDestination.LOCAL
                else SceneSource.SHARED
            ),
        ).register(
            SceneIndexEntry(
                name=name,
                robot=scene.robot_name
                or ("" if scene.robot is None else scene.robot.name),
                environment=scene.environment_name,
                task=scene.task,
            )
        )
        self.discard()
        return SavedScene(scene=name)


@dataclass(frozen=True)
class SavedBundle:
    """
    A copy of the ``__recording__`` bundle being turned into a saved scene.
    """

    directory: Path
    """
    The copy.
    """

    frame_range: FrameRange
    """
    The frames kept.
    """

    def rewrite(self, name: str, request: RecordingSaveRequest) -> SceneDescription:
        """
        Trim the copy's trajectory, statecharts, segments and plan steps to the kept
        frames, and name it.

        :param name: The saved name.
        :param request: The save request with the labels a person gave.
        :return: The rewritten scene description.
        """
        kept = slice(self.frame_range.first, self.frame_range.last + 1)
        trajectory_file = DocumentFile(self.directory / BundleFile.TRAJECTORY)
        trajectory = trajectory_file.read(Trajectory)
        trajectory = replace(
            trajectory,
            frames=trajectory.frames[kept],
            frame_times=trajectory.frame_times[kept],
            base=trajectory.base[kept],
            model_bases=trajectory.model_bases[kept],
            objects=trajectory.objects[kept],
        )
        trajectory_file.write(trajectory)
        statecharts_file = DocumentFile(self.directory / BundleFile.STATECHARTS)
        statecharts = statecharts_file.read_if_present(Statecharts)
        if statecharts is not None:
            statecharts.frames = statecharts.frames[kept]
            statecharts_file.write(statecharts)
        scene_file = DocumentFile(self.directory / BundleFile.SCENE)
        scene = scene_file.read(SceneDescription)
        first_poses = trajectory.objects[0] if trajectory.objects else {}
        scene = replace(
            scene,
            name=name,
            segments=SegmentClipping(self.frame_range).clip(scene.segments),
            plan_trees=PlanNodeClipping(self.frame_range).clip(scene.plan_trees),
            objects=[
                replace(entry, spawn=first_poses.get(entry.key, entry.spawn))
                for entry in scene.objects
            ],
            robot_name=self.label(request.robot, scene.robot_name),
            environment_name=self.label(request.environment, scene.environment_name),
            task=self.label(request.task, scene.task),
            world_bound=None,
            bundle_signature=None,
        )
        scene_file.write(scene)
        return scene

    @staticmethod
    def label(given: Optional[str], recorded: Optional[str]) -> Optional[str]:
        """
        :param given: What a person called it when saving, if anything.
        :param recorded: What the recording called it.
        :return: The person's label when given, else the recorded one.
        """
        return given or recorded
