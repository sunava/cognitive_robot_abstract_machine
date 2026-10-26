"""
Recording a run's frames, without a world.
"""

from __future__ import annotations

import pytest

from cramera_bridge.exceptions import InvalidFrameRange, NoRecording
from cramera_bridge.live.recording import (
    DURATION_PRECISION,
    MINIMUM_FRAMES_PER_SECOND,
    FrameRange,
    Recording,
)
from cramera_protocol.geometry import Pose
from cramera_protocol.live_messages import LiveState, RecordingState

from .clock import ManualClock

FRAME_INTERVAL = 0.25
"""
The seconds between two recorded frames.
"""

JOINT = "pr2/torso_lift_joint"
"""
A joint of the recorded state.
"""

OBJECT_KEY = "milk"
"""
A free object of the recorded state.
"""


def state(value: float) -> LiveState:
    """
    :return: A state with one joint and one object at the given value.
    """
    return LiveState(
        frames={JOINT: value},
        objects={OBJECT_KEY: Pose(value, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
    )


def recorded(count: int, clock: ManualClock) -> Recording:
    """
    :return: A recording of the given number of frames, one interval apart.
    """
    recording = Recording(clock=clock)
    recording.start()
    for index in range(count):
        clock.now = index * FRAME_INTERVAL
        recording.append(state(float(index)), None, None)
    return recording


# %% recording


def test_frames_are_recorded_only_while_recording():
    clock = ManualClock()
    recording = Recording(clock=clock)
    recording.append(state(0.0), None, None)
    assert recording.frame_count() == 0
    recording.start()
    recording.append(state(1.0), None, None)
    assert recording.frame_count() == 1


def test_a_recorded_frame_keeps_its_values_when_the_state_changes():
    recording = Recording(clock=ManualClock())
    recording.start()
    live = state(1.0)
    recording.append(live, None, None)
    live.frames[JOINT] = 2.0
    live.objects[OBJECT_KEY] = Pose(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    [frame] = recording.finish()
    assert frame.frames == state(1.0).frames
    assert frame.objects == state(1.0).objects


def test_finishing_keeps_the_frames_and_finishing_again_returns_them():
    recording = recorded(3, ManualClock())
    frames = recording.finish()
    assert recording.finish() == frames
    assert recording.status().state is RecordingState.FINALIZED


def test_finishing_nothing_is_refused():
    with pytest.raises(NoRecording):
        Recording().finish()


def test_discarding_drops_the_frames():
    recording = recorded(3, ManualClock())
    recording.written("saved")
    recording.discard()
    status = recording.status()
    assert (status.state, status.frame_count, status.scene_name) == (
        RecordingState.IDLE,
        0,
        None,
    )


def test_the_status_measures_the_length_and_the_frame_rate():
    count = 5
    status = recorded(count, ManualClock()).status()
    duration = (count - 1) * FRAME_INTERVAL
    assert status.frame_count == count
    assert status.duration_seconds == round(duration, DURATION_PRECISION)
    assert status.frames_per_second == round(count / duration, DURATION_PRECISION)


def test_a_single_frame_has_no_frame_rate():
    status = recorded(1, ManualClock()).status()
    assert status.duration_seconds == 0.0
    assert status.frames_per_second is None


def test_a_slow_recording_reports_the_minimum_frame_rate():
    clock = ManualClock()
    recording = Recording(clock=clock)
    recording.start()
    recording.append(state(0.0), None, None)
    clock.now = 10.0
    recording.append(state(1.0), None, None)
    assert recording.status().frames_per_second == MINIMUM_FRAMES_PER_SECOND


# %% frame ranges


def test_a_range_inside_the_frames_is_accepted():
    frame_range = FrameRange(first=1, last=3)
    frame_range.require_within(4)
    assert frame_range.length() == 3


@pytest.mark.parametrize("first, last", [(-1, 2), (3, 2), (0, 4)])
def test_a_range_outside_the_frames_is_refused(first, last):
    with pytest.raises(InvalidFrameRange):
        FrameRange(first=first, last=last).require_within(4)
