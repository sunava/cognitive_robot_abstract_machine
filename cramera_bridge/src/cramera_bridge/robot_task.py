"""
Robot task demos: a CRAM robot performs a task in a scene environment while the run is
published to the viewer, measured and saved as a scene bundle.

A demo states its environment, its robot, its plan and how its outcome is measured; the
run itself, the recording and saving it, and publishing it later are the same for every
demo.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, Optional, Self, Sequence

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.plan import Plan
from cramera.locations import DataLocations
from cramera.scenes import SceneSource
from cramera_bridge.documents import DocumentFile
from cramera_bridge.exceptions import (
    EnvironmentBundleHoldsRobot,
    RecordingNotSaved,
    SceneNameTaken,
)
from cramera_bridge.export.scene_index import SceneIndexFile
from cramera_bridge.live.server import LIVE_PORT
from cramera_bridge.scenes.bundle_world import BundleEnvironment
from cramera_bridge.scenes.environment import SceneEnvironment
from cramera_bridge.scenes.presentation import ScenePresentation
from cramera_bridge.visualization import LiveVisualization
from cramera_protocol.bundle.acceptance import AcceptanceMetrics
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.bundle.index import SceneIndexEntry, SceneName
from cramera_protocol.bundle.scene import Camera, SceneDescription
from cramera_protocol.live_messages import (
    RecordingSaveRequest,
    SavedScene,
    SceneDestination,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World

PUBLISHING_PREFIX = ".publish_"
"""
The prefix of the folder a bundle is staged in while it is published.
"""

REPLACED_SUFFIX = ".replaced"
"""
The suffix of a shared bundle's old version while the new one takes its place.
"""

NAME_PART_SEPARATOR = "_"
"""
What separates the environment, the robot and the task in a recording's name.
"""

# %% demo


@dataclass
class RobotTaskDemo(ABC):
    """
    One robot task in one scene environment, recorded into a bundle named after the
    scene, the robot and the task.

    Subclasses build the plan and measure the outcome from the world; :meth:`run` does
    the rest: it builds the world, publishes it while the plan runs, records the run,
    checks the outcome and saves the recording into the local scenes directory.
    """

    environment: SceneEnvironment
    """
    Builds the world the task is performed in.
    """

    robot: RobotSpecification
    """
    The CRAM robot and where it starts.
    """

    name: str
    """
    The name the recording is saved under.
    """

    task: str
    """
    What the task does, as the scene picker labels it.
    """

    environment_name: Optional[str] = None
    """
    The environment's display name; recordings in the same environment share it.
    """

    camera: Optional[Camera] = None
    """
    The first view of the recording, or None to keep the environment's view.
    """

    presentation: Optional[ScenePresentation] = None
    """
    The appearance of the environment bundle the recording takes over, if any.
    """

    locations: DataLocations = field(
        default_factory=lambda: DataLocations.from_environment(
            dict(os.environ), Path.home()
        )
    )
    """
    Where the recording is saved: its local scenes directory, as the process's
    environment configures it.
    """

    port: int = LIVE_PORT
    """
    The port the run is published on while it runs.
    """

    execution_type: ExecutionType = ExecutionType.KINEMATICALLY_SIMULATED
    """
    How the plan's motions are executed.
    """

    collision_avoidance: bool = False
    """
    Whether every motion avoids collisions with the environment.
    """

    @abstractmethod
    def build_plan(self, context: Context) -> Plan:
        """
        :param context: The world and the robot the task is performed with.
        :return: The task's plan.
        """

    @abstractmethod
    def measure(self, context: Context) -> AcceptanceMetrics:
        """
        Measure the outcome from the world after the plan ran.

        :param context: The world and the robot the task was performed with.
        :return: The measurements.
        :raises TaskFailed: When the outcome is not the task's.
        """

    @classmethod
    def in_environment_bundle(
        cls,
        directory: Path,
        robot: RobotSpecification,
        robot_name: str,
        task_identifier: str,
        task: str,
        **fields: Any,
    ) -> Self:
        """
        A task in the environment of a scene bundle, recorded as ``<bundle>_<robot
        name>_<task identifier>`` with the bundle's environment name and appearance, so
        the viewer offers the recording beside its environment.

        :param directory: An environment bundle.
        :param robot: The CRAM robot and where it starts.
        :param robot_name: The robot's part of the recording's name.
        :param task_identifier: The task's part of the recording's name.
        :param task: What the task does, as the scene picker labels it.
        :param fields: The other fields of the demo.
        :return: The demo.
        :raises BundleNotFound: When the directory holds no bundle.
        :raises EnvironmentBundleHoldsRobot: When the bundle holds a robot model.
        """
        environment = BundleEnvironment(directory)
        scene = environment.contents.scene
        if any(model.robot for model in scene.models):
            raise EnvironmentBundleHoldsRobot(environment.directory)
        return cls(
            environment=environment,
            robot=robot,
            name=NAME_PART_SEPARATOR.join(
                (environment.directory.name, robot_name, task_identifier)
            ),
            task=task,
            environment_name=scene.environment_name or scene.name,
            presentation=ScenePresentation.of_bundle(environment.directory),
            **fields,
        )

    # %% running

    def scene_name(self) -> str:
        """
        :return: The validated name the recording is saved under.
        :raises InvalidSceneName: When the name is no bundle name.
        """
        return SceneName(self.name).text

    def build_world(self) -> World:
        """
        :return: A new world holding the environment and the robot.
        """
        return self.environment.create_world(robots=[self.robot])

    def build_context(self, world: World) -> Context:
        """
        :param world: The world built for the task.
        :return: The context naming the robot of the task.
        """
        robot = world.get_semantic_annotations_by_type(
            self.robot.semantic_annotation_type
        )[0]
        return Context(world=world, robot=robot)

    def run(self) -> SavedScene:
        """
        Perform the task while it is published and recorded, measure the outcome and
        save the recording with its measurements.

        :return: The saved bundle.
        :raises InvalidSceneName: When the name is no bundle name.
        :raises SceneNameTaken: When the local scenes directory already holds the name.
        :raises TaskFailed: When the measured outcome is not the task's; nothing is
            saved then.
        """
        name = self.scene_name()
        if (self.locations.local_scenes / name).exists():
            raise SceneNameTaken(name)
        world = self.build_world()
        context = self.build_context(world)
        plan = self.build_plan(context)
        visualization = LiveVisualization(
            world=world, port=self.port, locations=self.locations
        ).start()
        try:
            plan.node_callbacks.append(visualization.plan_callback(plan))
            with ExecutionEnvironment(
                execution_type=self.execution_type,
                collision_avoidance=self.collision_avoidance,
            ):
                plan.perform()
            metrics = self.measure(context)
            visualization.publisher.finish_recording()
            saved = visualization.publisher.save_recording(
                RecordingSaveRequest(
                    name=name,
                    destination=SceneDestination.LOCAL,
                    environment=self.environment_name,
                    task=self.task,
                )
            )
        finally:
            visualization.stop()
        self.complete(self.locations.local_scenes / saved.scene, metrics)
        return saved

    def complete(self, bundle: Path, metrics: AcceptanceMetrics) -> None:
        """
        Give the saved bundle the demo's view and the environment's appearance, and
        store the measurements beside it.

        :param bundle: The saved bundle's directory.
        :param metrics: The measured outcome.
        """
        scene_file = DocumentFile(bundle / BundleFile.SCENE)
        scene = scene_file.read(SceneDescription)
        if self.presentation is not None:
            self.presentation.apply(scene, bundle)
        if self.camera is not None:
            scene.camera = self.camera
        scene_file.write(scene)
        DocumentFile(bundle / BundleFile.ACCEPTANCE_METRICS).write(metrics)

    # %% publishing

    def publish(self, shared: Path) -> SavedScene:
        """
        Copy the recording a previous run saved into a shared scenes directory, without
        running the task again; the copy is staged beside its destination and moved in
        whole, so the shared directory never holds half a bundle.

        :param shared: The shared scenes directory.
        :return: The published bundle.
        :raises RecordingNotSaved: When no run saved the recording yet.
        """
        name = self.scene_name()
        saved = self.locations.local_scenes / name
        scene_file = DocumentFile(saved / BundleFile.SCENE)
        if not scene_file.path.is_file():
            raise RecordingNotSaved(name)
        shared.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f"{PUBLISHING_PREFIX}{name}.", dir=shared)
        )
        shutil.copytree(saved, staging / name)
        target = shared / name
        if target.exists():
            target.rename(staging / f"{name}{REPLACED_SUFFIX}")
        (staging / name).rename(target)
        shutil.rmtree(staging)
        scene = scene_file.read(SceneDescription)
        SceneIndexFile(shared, SceneSource.SHARED).register(
            SceneIndexEntry(
                name=name,
                robot=scene.robot_name
                or ("" if scene.robot is None else scene.robot.name),
                environment=scene.environment_name,
                task=scene.task,
            )
        )
        return SavedScene(scene=name)


# %% command line


class TaskArgument(StrEnum):
    """
    The arguments of a task demo's command.
    """

    PUBLISH = "--publish"
    """
    Copy the saved recording into this shared scenes directory instead of running.
    """

    PORT = "--port"
    """
    The port the run is published on.
    """


@dataclass(frozen=True)
class TaskCommand:
    """
    The command line of one task demo: run it, or publish its saved recording.
    """

    demo: RobotTaskDemo
    """
    The demo.
    """

    def main(self, arguments: Sequence[str]) -> SavedScene:
        """
        :param arguments: The command's arguments.
        :return: The saved or published bundle.
        """
        parser = ArgumentParser(description=self.demo.task)
        parser.add_argument(
            TaskArgument.PUBLISH,
            type=Path,
            default=None,
            help="copy the saved recording into this shared scenes directory",
        )
        parser.add_argument(
            TaskArgument.PORT,
            type=int,
            default=self.demo.port,
            help="the port the run is published on",
        )
        options = parser.parse_args(list(arguments))
        if options.publish is not None:
            return self.demo.publish(options.publish)
        self.demo.port = options.port
        return self.demo.run()
