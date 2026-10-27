"""
Onboarding a coraplex demo: run the demo file unchanged with the cramera backend, record
everything its visualization publishes and save the recording as a named scene bundle.

Nothing of the demo is patched: the demo selects coraplex's cramera backend, and the
backend's visualizations are created by this onboarding, which records through them.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import List, Optional

from cramera.locations import DataLocations
from cramera_bridge.coraplex_backend import CoraplexBackendRegistration
from cramera_bridge.demo_script import DemoScript
from cramera_bridge.detections import DetectionRecorder
from cramera_bridge.documents import DocumentFile
from cramera_bridge.exceptions import DemoStartedNoWorld, SceneNameTaken
from cramera_bridge.live.recording_storage import RecordingStorage
from cramera_bridge.visualization import LiveVisualization
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.bundle.index import SceneName
from cramera_protocol.bundle.scene import SceneDescription
from cramera_protocol.live_messages import (
    RecordingSaveRequest,
    SavedScene,
    SceneDestination,
)
from semantic_digital_twin.world import World

ONBOARDING_PORT = 0
"""
The port the onboarded run is published on: any free one, so onboarding never collides
with a live publisher on the default port.
"""

# %% onboarding


@dataclass
class DemoOnboarding:
    """
    Runs one demo file and saves what its visualization recorded as a scene bundle.
    """

    demo: DemoScript
    """
    The demo file and its arguments.
    """

    name: str
    """
    The bundle name the recording is saved under.
    """

    locations: DataLocations
    """
    Where cramera keeps its scenes.
    """

    destination: SceneDestination = SceneDestination.LOCAL
    """
    The scenes directory the bundle is saved into.
    """

    task: Optional[str] = None
    """
    What the demo does, as the scene picker labels it.
    """

    detect_events: bool = False
    """
    Whether segmind's detectors watch the run and the bundle lists what they saw.
    """

    visualizations: List[LiveVisualization] = field(default_factory=list, init=False)
    """
    The visualizations the demo started.
    """

    recorders: List[DetectionRecorder] = field(default_factory=list, init=False)
    """
    The detectors watching the demo's worlds.
    """

    def visualization(self, world: World) -> LiveVisualization:
        """
        Create the visualization coraplex asks for, recording into this onboarding's
        locations, and the detectors of its world when events are detected.

        :param world: The world the demo visualizes.
        :return: The visualization, not yet started.
        """
        visualization = LiveVisualization(
            world=world,
            port=ONBOARDING_PORT,
            locations=self.locations,
            read_ros_markers=False,
            read_robokudo_images=False,
        )
        self.visualizations.append(visualization)
        if self.detect_events:
            self.recorders.append(DetectionRecorder(world).start())
        return visualization

    def run(self) -> SavedScene:
        """
        Run the demo and save the recording its last stopped visualization wrote.

        :return: The saved bundle.
        :raises InvalidSceneName: When the name is no bundle name.
        :raises SharedScenesUnavailable: For an unconfigured shared directory.
        :raises SceneNameTaken: When a scenes directory already holds the name; checked
            before the demo runs.
        :raises DemoStartedNoWorld: When the demo started no visualization.
        """
        storage = RecordingStorage(self.locations)
        name = SceneName(self.name).text
        destination = storage.destination(self.destination)
        if any((directory / name).exists() for directory in storage.directories()):
            raise SceneNameTaken(name)
        with ExitStack() as running:
            running.callback(self.stop_watching)
            with CoraplexBackendRegistration(factory=self.visualization):
                self.demo.run()
        if not self.visualizations:
            raise DemoStartedNoWorld(self.demo.path)
        saved = storage.save(
            RecordingSaveRequest(
                name=name, destination=self.destination, task=self.task
            )
        )
        self.add_detected_events(destination / saved.scene)
        return saved

    def stop_watching(self) -> None:
        """
        Stop the demo's visualizations, which writes their recordings, and its
        detectors; stopping what already stopped changes nothing.
        """
        for visualization in self.visualizations:
            visualization.stop()
        for recorder in self.recorders:
            recorder.stop()

    def add_detected_events(self, bundle: Path) -> None:
        """
        List what the detectors saw in the saved bundle's scene.

        :param bundle: The saved bundle's directory.
        """
        events = [event for recorder in self.recorders for event in recorder.events()]
        if not events:
            return
        scene_file = DocumentFile(bundle / BundleFile.SCENE)
        scene = scene_file.read(SceneDescription)
        scene.detected_events = events
        scene_file.write(scene)
