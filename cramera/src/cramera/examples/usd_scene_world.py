"""
Load a scene converted by ``cramera-import`` into a CRAM world, report what it holds and
optionally serve it to the live viewer, where its prims can be queried with EQL.
"""

from __future__ import annotations

import argparse
import signal
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Self, Sequence

from cramera.live.http import DEFAULT_PORT
from cramera.paths import LIVE_SCENE_NAME
from cramera.usd_scene_world import UsdPrimAnnotation, UsdSceneEnvironment
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


# %% example configuration
class ExampleArgument(StrEnum):
    """
    Command-line options accepted by the example.
    """

    SCENE = "--scene"
    """
    Load the bundle of this name from CRAMERA's scene roots.
    """

    BUNDLE_DIRECTORY = "--bundle-directory"
    """
    Load the bundle in this directory.
    """

    LIVE = "--live"
    """
    Serve the world to the live viewer until interrupted.
    """

    PORT = "--port"
    """
    The port of the live bridge.
    """


# %% loaded scene
@dataclass(frozen=True)
class LoadedUsdScene:
    """
    A world created from a converted scene and how long creating it took.
    """

    environment: UsdSceneEnvironment
    """
    The converted scene.
    """

    world: World
    """
    The world holding the scene.
    """

    load_seconds: float
    """
    The wall time creating the world took.
    """

    @classmethod
    def load(cls, environment: UsdSceneEnvironment) -> Self:
        """
        Create a world from a converted scene, timing it.

        :param environment: The converted scene.
        """
        started = time.perf_counter()
        world = environment.create_world()
        return cls(environment, world, time.perf_counter() - started)

    @property
    def movable_joint_count(self) -> int:
        """
        The number of connections moving along one degree of freedom.
        """
        return sum(
            isinstance(connection, ActiveConnection1DOF)
            for connection in self.world.connections
        )

    @property
    def label_count(self) -> int:
        """
        The number of prim labels.
        """
        return len(self.world.get_semantic_annotations_by_type(UsdPrimAnnotation))

    def summary(self) -> str:
        """
        What the world holds, one fact per line.
        """
        return "\n".join(
            [
                f"Scene: {self.environment.bundle_directory}",
                f"Bodies: {len(self.world.bodies)}",
                f"Movable joints: {self.movable_joint_count}",
                f"Prim labels: {self.label_count}",
                "Class annotations: "
                f"{len(self.world.semantic_annotations) - self.label_count}",
                f"Load time: {self.load_seconds:.2f} s",
            ]
        )


# %% entry point
def main(arguments: Sequence[str] | None = None) -> None:
    """
    Load a converted scene, print what its world holds and optionally serve it live.

    :param arguments: Command-line arguments, or None to read the process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(ExampleArgument.SCENE)
    source.add_argument(ExampleArgument.BUNDLE_DIRECTORY, type=Path)
    parser.add_argument(ExampleArgument.LIVE, action="store_true")
    parser.add_argument(ExampleArgument.PORT, type=int, default=DEFAULT_PORT)
    options = parser.parse_args(arguments)
    environment = (
        UsdSceneEnvironment.from_scene_name(options.scene)
        if options.scene is not None
        else UsdSceneEnvironment(bundle_directory=options.bundle_directory)
    )
    scene = LoadedUsdScene.load(environment)
    print(scene.summary())
    if not options.live:
        return
    scene.environment.create_visualization(scene.world, port=options.port).start()
    print(
        f"Serving the live world; open the viewer at "
        f"index.html?scene={LIVE_SCENE_NAME}&live=127.0.0.1:{options.port}"
    )
    signal.pause()


if __name__ == "__main__":
    main()
