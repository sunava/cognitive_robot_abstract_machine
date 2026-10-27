"""
Dividing a recording into the steps of its replay timeline.
"""

from __future__ import annotations

from typing_extensions import List, Optional

from cramera_bridge.live.recording import FrameRange, RecordedFrame
from cramera_bridge.live.segments import RecordedRun, SegmentClipping, UnlabelledStep
from cramera_protocol.bundle.scene import Segment
from cramera_protocol.geometry import Pose

OBJECT_KEY = "cup"
"""
The key of the carried object.
"""

# %% helpers


def frame(x: Optional[float], step: Optional[str] = None) -> RecordedFrame:
    """
    :param x: Where the object stands along x, or None when it is not in the frame.
    :param step: The action reported.
    :return: A frame.
    """
    objects = {} if x is None else {OBJECT_KEY: Pose(x, 0, 0, 0, 0, 0, 1)}
    return RecordedFrame(
        time=0.0, frames={}, base=None, objects=objects, model_bases={}, step=step
    )


def steps(segments: List[Segment]) -> List[str]:
    """
    :return: The names of the segments.
    """
    return [segment.step for segment in segments]


# %% segments


def test_a_run_without_carried_objects_follows_the_reported_actions():
    run = RecordedRun([frame(0, "Navigate"), frame(0, "Navigate"), frame(0, "Look")])
    segments = run.segments()
    assert steps(segments) == ["Navigate", "Look"]
    assert (segments[0].start, segments[1].end) == (0, 2)


def test_a_run_reporting_nothing_is_one_unlabelled_step():
    assert steps(RecordedRun([frame(0), frame(0)]).segments()) == [UnlabelledStep.RUN]


def test_a_carried_object_gets_a_step_with_its_pick_and_place():
    run = RecordedRun([frame(0), frame(0), frame(0.5), frame(1.0), frame(1.0)])
    segments = run.segments()
    carried = segments[1]
    assert steps(segments) == [
        UnlabelledStep.RUN,
        f"{UnlabelledStep.MOVE}_{OBJECT_KEY}",
    ]
    assert (carried.picks, carried.attach, carried.detach) == (OBJECT_KEY, 2, 3)


def test_an_object_appearing_late_is_no_carried_object():
    run = RecordedRun([frame(None), frame(1.0)])
    assert steps(run.segments()) == [UnlabelledStep.RUN]


def test_an_empty_run_has_no_steps():
    assert RecordedRun([]).segments() == []


def test_clipping_keeps_and_rebases_the_overlapping_steps():
    segments = [
        Segment(step="a", start=0, end=3),
        Segment(step="b", start=3, end=8, attach=4, detach=7),
        Segment(step="c", start=8, end=9),
    ]
    clipped = SegmentClipping(FrameRange(4, 7)).clip(segments)
    assert steps(clipped) == ["b"]
    assert (clipped[0].start, clipped[0].end, clipped[0].attach) == (0, 3, 0)
