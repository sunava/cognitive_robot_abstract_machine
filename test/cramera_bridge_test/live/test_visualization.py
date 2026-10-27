"""
The coraplex backend: publishing a world and its plans while a demo runs, and the
command running a demo.
"""

from __future__ import annotations

import json
import signal
import socket
import sys
import threading
import urllib.request

import pytest

from cramera_bridge.command_line import DEMO_ARGUMENTS_SEPARATOR, LiveCommand, live
from cramera_bridge.coraplex_backend import CoraplexBackend
from cramera_bridge.documents import DocumentFile
from cramera_bridge.live.plan import PlanNodeStatus
from cramera_bridge.visualization import (
    LiveVisualization,
    PublishedModelChange,
    PublishedStateChange,
)
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.bundle.index import ReservedSceneName
from cramera_protocol.bundle.scene import SceneDescription
from cramera_protocol.live import LiveInfo, LiveRoute
from cramera_protocol.live_messages import LivePlan
from cramera_bridge.questions.workspace_classes import WorkspaceClassIndex
from giskardpy.motion_statechart.data_types import LifeCycleValues
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from ..dataset.live_worlds import HeightChart
from .conftest import LIVE_DATASET

DEMO_OUTPUT_VARIABLE = "CRAMERA_TEST_DEMO_OUTPUT"
"""
Where the test demo writes what it was given.
"""

# %% backend


def visualization(kitchen, locations, port: int = 0) -> LiveVisualization:
    """
    :return: A visualization of the kitchen without ROS markers.
    """
    return LiveVisualization(
        world=kitchen.world,
        port=port,
        locations=locations,
        read_ros_markers=False,
        class_index=WorkspaceClassIndex(),
    )


def state_callbacks(kitchen):
    """
    :return: The publisher callbacks registered on the kitchen world.
    """
    return [
        callback
        for callback in kitchen.world.state.state_change_callbacks
        if isinstance(callback, PublishedStateChange)
    ]


def test_a_started_visualization_serves_and_records(kitchen, locations, small_plan):
    started = visualization(kitchen, locations).start()
    with urllib.request.urlopen(started.url() + LiveRoute.INFO) as answer:
        info = LiveInfo.from_json(json.loads(answer.read()))
    progress = started.plan_callback(small_plan.plan)
    small_plan.plan.node_callbacks.append(progress)
    small_plan.root.status = LifeCycleValues.RUNNING
    progress.on_start(small_plan.motion)
    chart = HeightChart.build(kitchen.world, kitchen.milk)
    for step in range(3):
        kitchen.milk_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            1.0 + step, 0.0, 0.9, reference_frame=kitchen.root
        )
        progress.on_motion_tick(chart.chart)
    progress.on_end(small_plan.motion)
    with urllib.request.urlopen(started.url() + LiveRoute.PLAN) as answer:
        plan = LivePlan.from_json(json.loads(answer.read()))
    started.stop()
    scene = DocumentFile(
        locations.local_scenes / ReservedSceneName.RECORDING / BundleFile.SCENE
    ).read(SceneDescription)
    assert info.running is True
    assert plan.nodes[0].status == PlanNodeStatus.RUNNING
    assert scene.name == ReservedSceneName.RECORDING
    assert state_callbacks(kitchen) == []
    assert started.publisher is None


def test_a_model_change_reaches_the_publisher(kitchen, locations):
    started = visualization(kitchen, locations).start()
    with kitchen.world.modify_world():
        kitchen.world.remove_connection(kitchen.frame.parent_connection)
        kitchen.world.remove_kinematic_structure_entity(kitchen.frame)
    keys = [entry.key for entry in started.publisher.object_catalog().objects]
    started.stop()
    assert str(kitchen.frame.id) not in keys


def test_a_failed_start_releases_what_it_acquired(kitchen, locations):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(OSError):
            visualization(kitchen, locations, occupied.getsockname()[1]).start()
    assert state_callbacks(kitchen) == []
    assert not [
        callback
        for callback in kitchen.world.get_world_model_manager().model_change_callbacks
        if isinstance(callback, PublishedModelChange)
    ]


# %% command


def test_the_live_command_runs_a_demo_with_the_cramera_backend(tmp_path, monkeypatch):
    output = tmp_path / "output.txt"
    monkeypatch.setenv(DEMO_OUTPUT_VARIABLE, str(output))
    monkeypatch.delenv(CoraplexBackend.VARIABLE, raising=False)
    path_before = list(sys.path)
    command = LiveCommand.parse([str(LIVE_DATASET / "demo_script.py"), "a", "b"])
    command.run_demo()
    assert output.read_text(encoding="utf-8") == f"a b {CoraplexBackend.VALUE}"
    assert sys.path == path_before


def test_demo_options_follow_the_separator(tmp_path, monkeypatch):
    output = tmp_path / "output.txt"
    monkeypatch.setenv(DEMO_OUTPUT_VARIABLE, str(output))
    monkeypatch.delenv(CoraplexBackend.VARIABLE, raising=False)
    command = LiveCommand.parse(
        [str(LIVE_DATASET / "demo_script.py"), DEMO_ARGUMENTS_SEPARATOR, "--seed", "3"]
    )
    command.run_demo()
    assert output.read_text(encoding="utf-8") == f"--seed 3 {CoraplexBackend.VALUE}"


def test_the_live_command_runs_until_interrupted(tmp_path, monkeypatch):
    output = tmp_path / "output.txt"
    monkeypatch.setenv(DEMO_OUTPUT_VARIABLE, str(output))
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
    live([str(LIVE_DATASET / "demo_script.py")])
    pending = signal.sigpending()
    signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)
    assert output.read_text(encoding="utf-8") == CoraplexBackend.VALUE
    assert signal.SIGINT not in pending
