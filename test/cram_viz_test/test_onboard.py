"""
Tests for the pure, CRAM-stack-free post-processing helpers in the onboarder.
"""

from __future__ import annotations

import time

from cram_viz.onboard import demo
from cram_viz.onboard.demo import (
    ObjectPalette,
    Recorder,
    Timer,
    derive_segments,
    first_base_motion,
    link_set,
    moved,
    object_windows,
)

# %% moved


class TestMoved:
    def test_within_epsilon_is_not_moved(self):
        assert not moved([0.0, 0.0, 0.0], [0.0, 0.0, 0.01])

    def test_beyond_epsilon_is_moved(self):
        assert moved([0.0, 0.0, 0.0], [0.1, 0.0, 0.0])


# %% object_windows


def _recorder_with_object_frames(*frames: dict) -> Recorder:
    recorder = Recorder(timer=Timer(start=0.0))
    recorder.obj_frames = list(frames)
    return recorder


class TestObjectWindows:
    def test_stationary_object_has_no_window(self):
        pose = [1.0, 1.0, 0.0, 0, 0, 0, 1]
        recorder = _recorder_with_object_frames({"milk": pose}, {"milk": pose}, {"milk": pose})
        assert object_windows(recorder) == []

    def test_object_that_travelled_has_one_window(self):
        start, mid, end = [1.0, 1.0, 0.0, 0, 0, 0, 1], [1.5, 1.0, 0.0, 0, 0, 0, 1], [2.0, 1.0, 0.0, 0, 0, 0, 1]
        recorder = _recorder_with_object_frames(
            {"milk": start}, {"milk": start}, {"milk": mid}, {"milk": end}, {"milk": end}
        )
        windows = object_windows(recorder)
        assert len(windows) == 1
        assert windows[0]["object"] == "milk"
        assert windows[0]["attach"] < windows[0]["detach"]
        assert windows[0]["place"] == [2.0, 1.0, 0.0]


# %% first_base_motion


class TestFirstBaseMotion:
    def test_returns_before_when_base_never_moves(self):
        recorder = Recorder(timer=Timer(start=0.0))
        recorder.base_frames = [[0, 0, 0, 0, 0, 0, 1]] * 5
        assert first_base_motion(recorder, before=5) == 5

    def test_returns_first_frame_base_left_spawn(self):
        recorder = Recorder(timer=Timer(start=0.0))
        recorder.base_frames = [
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0.5, 0, 0, 0, 0, 0, 1],
            [1.0, 0, 0, 0, 0, 0, 1],
        ]
        assert first_base_motion(recorder, before=4) == 2


# %% derive_segments


class TestDeriveSegments:
    def test_single_action_with_no_target_becomes_one_segment(self):
        recorder = Recorder(timer=Timer(start=0.0))
        recorder.frames = [{}] * 4
        recorder.obj_frames = [{}] * 4
        recorder.base_frames = [[0, 0, 0, 0, 0, 0, 1]] * 4
        recorder.actions = [{"action": "ParkArmsAction", "arm": None, "target": None}]

        segments = derive_segments(recorder)

        assert len(segments) == 1
        assert segments[0]["step"] == "parkarms"
        assert segments[0]["start"] == 0
        assert segments[0]["end"] == 3

    def test_manipulation_action_produces_a_pick_place_segment(self):
        start = [1.0, 1.0, 0.0, 0, 0, 0, 1]
        mid = [1.5, 1.0, 0.0, 0, 0, 0, 1]
        pose = [2.0, 1.0, 0.0, 0, 0, 0, 1]
        recorder = Recorder(timer=Timer(start=0.0))
        recorder.frames = [{}] * 20
        recorder.base_frames = [[0, 0, 0, 0, 0, 0, 1]] * 20
        recorder.obj_frames = (
            [{"milk.stl": start}] * 10 + [{"milk.stl": mid}] + [{"milk.stl": pose}] * 9
        )
        recorder.actions = [{"action": "TransportAction", "arm": "left", "target": "milk.stl"}]

        segments = derive_segments(recorder)

        picked = next(segment for segment in segments if segment.get("picks") == "milk")
        assert picked["step"] == "transport_milk"
        assert picked["arm"] == "left"
        assert picked["place"] == pose[:3]


# %% link_set


class _FakeBody:
    def __init__(self, name: str):
        self.name = name


class _FakePart:
    def __init__(self, *names: str):
        self.bodies = [_FakeBody(name) for name in names]


class TestLinkSet:
    def test_strips_model_name_prefix(self):
        part = _FakePart("pr2/l_shoulder_link", "pr2/l_wrist_link")
        assert link_set(part) == ["l_shoulder_link", "l_wrist_link"]

    def test_keeps_name_without_prefix(self):
        part = _FakePart("base_link")
        assert link_set(part) == ["base_link"]

    def test_empty_part_has_no_links(self):
        assert link_set(_FakePart()) == []


# %% Timer / ObjectPalette


class TestTimer:
    def test_log_formats_message_with_elapsed_time(self, monkeypatch):
        calls = []
        monkeypatch.setattr(demo.logger, "info", lambda *args: calls.append(args))
        timer = Timer(start=time.time() - 1.5)

        timer.log("hello", 42)

        assert calls == [("[%6.1fs] %s", calls[0][1], "hello 42")]
        assert calls[0][1] >= 1.5


class TestObjectPalette:
    def test_cycles_through_colors(self):
        palette = ObjectPalette()
        assert palette.color_for(0) == palette.colors[0]
        assert palette.color_for(len(palette.colors)) == palette.colors[0]
