"""
Sample each native control tick after independent workshop contents update.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.plan_node import ActionNode
from cramera.body_geometry import POSE_PRECISION
from cramera.live.bridge import Bridge
from cramera.live.recording import Recording
from cramera.live.recording_bundle import write_recording_bundle
from cramera.live.visualization import WorldModelSync, WorldStateSync
from cramera.workshop_assembly_demo import AssemblyObservation
from cramera.workshop_contents import WorkshopContents
from giskardpy.motion_statechart.motion_statechart import MotionStatechart

from .test_workshop_actions import aligned_assembly
from .test_workshop_bundle import workshop_directory
from .test_workshop_contents import (
    CarrierAction,
    contents,
    move_box,
)
from .test_workshop_world import workshop_bundle


# %% controller sampling after contact updates
def test_retrieval_samples_the_final_contact_state_once_per_control_tick(
    contents: WorkshopContents, tmp_path: Path
) -> None:
    """
    Extra carrier and settling snapshots must not stretch playback or shift stages.
    """
    bridge = Bridge()
    bridge.attach(contents.world)
    bridge.recording = Recording()
    bridge.recording.start()
    WorldStateSync(_world=contents.world, bridge=bridge)
    WorldModelSync(_world=contents.world, bridge=bridge)
    contents.world.collision_manager.update_collision_matrix()
    retrieval, following = CarrierAction(), CarrierAction()
    contents.action = retrieval
    contents.start()
    observation = AssemblyObservation(
        context=Context(world=contents.world, robot=None),
        actions=[retrieval, following],
        labels=["Retrieve carrier", "Continue after placement"],
        bridge=bridge,
    )
    tick = MotionStatechart()
    retrieval_node = ActionNode(designator=retrieval)
    contents.on_start(retrieval_node)
    observation.on_start(retrieval_node)
    original = contents.box.global_pose.to_np()[:3, 3].copy()
    expected_screws = []
    for index in range(3):
        position = original + np.asarray([0, -0.01, 0.01]) * (index + 1)
        move_box(contents, position, Rotation.identity())
        contents.on_motion_tick(tick)
        expected_screws.append(
            contents.contents[0].global_pose.to_position_quaternion_list()
        )
        observation.on_motion_tick(tick)
    contents.on_end(retrieval_node)
    observation.on_start(ActionNode(designator=following))
    for index in range(2):
        position = original + np.asarray([0, -0.04, 0.04]) * (index + 1)
        move_box(contents, position, Rotation.identity())
        expected_screws.append(
            contents.contents[0].global_pose.to_position_quaternion_list()
        )
        observation.on_motion_tick(tick)

    recorded = bridge.recording.stop()
    frames = [recorded[index] for index in observation.frame_indices]
    assert len(recorded) > len(frames)
    assert len(frames) == observation.ticks == len(expected_screws)
    for frame, expected in zip(frames, expected_screws):
        assert frame.objects[str(contents.contents[0].name)] == [
            round(value, POSE_PRECISION) for value in expected
        ]
    assert observation.ranges == [
        {"name": observation.labels[0], "startFrame": 0, "endFrame": 2},
        {"name": observation.labels[1], "startFrame": 3, "endFrame": 4},
    ]

    output = tmp_path / "sampled-replay"
    frequency = 1.0 / contents.control_dt
    write_recording_bundle(bridge, frames, frequency, output, "workshop-sampling-test")
    trajectory = json.loads((output / "trajectory.json").read_text())
    assert trajectory["objects"] == [frame.objects for frame in frames]
    assert len(trajectory["frames"]) / trajectory["framesPerSecond"] == pytest.approx(
        observation.ticks * contents.control_dt
    )


def test_thread_installation_records_a_consistent_model_state(aligned_assembly) -> None:
    """
    Changing the screw connection leaves live model and pose callbacks usable.
    """
    world, assembly = aligned_assembly
    bridge = Bridge()
    bridge.attach(world)
    bridge.recording = Recording()
    bridge.recording.start()
    WorldStateSync(_world=world, bridge=bridge)
    WorldModelSync(_world=world, bridge=bridge)
    connection = assembly.install(world)
    connection.position = -np.pi / 2
    frames = bridge.recording.stop()
    assert frames[-1].objects[str(assembly.screw.name)] == [
        round(value, POSE_PRECISION)
        for value in assembly.screw.global_pose.to_position_quaternion_list()
    ]
