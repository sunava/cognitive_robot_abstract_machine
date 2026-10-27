"""
The steps a recording is divided into on the replay timeline: what the plan performed,
and when an object was picked up and set down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

from typing_extensions import List, Optional

from cramera_bridge.live.recording import FrameRange, RecordedFrame
from cramera_protocol.bundle.scene import PlanNode, Segment
from cramera_protocol.geometry import Pose

TRANSPORT_TOLERANCE_METRES = 0.02
"""
How far an object must end up from where it started to count as carried.
"""

MOTION_TOLERANCE_METRES = 0.005
"""
How far an object must move from a pose to count as having left it.
"""


class UnlabelledStep(StrEnum):
    """
    The names of stretches the plan reported no action for.
    """

    RUN = "run"
    """
    A stretch in which nothing was carried.
    """

    MOVE = "move"
    """
    The start of the name of a stretch in which an object was carried.
    """


# %% carried objects


@dataclass(frozen=True)
class CarryWindow:
    """
    The frames over which one object was carried.
    """

    object_key: str
    """
    The object's key.
    """

    attach: int
    """
    The first frame the object had left where it started.
    """

    detach: int
    """
    The first frame the object rested where it ended up.
    """


@dataclass(frozen=True)
class RecordedRun:
    """
    The frames of a recording, read for the steps they show.
    """

    frames: List[RecordedFrame]
    """
    The frames in recording order.
    """

    @staticmethod
    def distance(first: Pose, second: Pose) -> float:
        """
        :param first: A pose.
        :param second: Another pose.
        :return: The distance between their positions in metres.
        """
        return math.dist((first.x, first.y, first.z), (second.x, second.y, second.z))

    def windows(self) -> List[CarryWindow]:
        """
        :return: The carry window of every object that ended elsewhere than it began,
            in the order they were picked up.
        """
        if not self.frames:
            return []
        first, last = self.frames[0].objects, self.frames[-1].objects
        windows = []
        for key, start in first.items():
            end = last.get(key)
            if end is None or self.distance(start, end) <= TRANSPORT_TOLERANCE_METRES:
                continue
            attach = self.first_frame_away(key, start)
            detach = self.last_frame_away(key, end) + 1
            if attach <= detach:
                windows.append(CarryWindow(key, attach, detach))
        return sorted(windows, key=lambda window: window.attach)

    def first_frame_away(self, key: str, pose: Pose) -> int:
        """
        :param key: An object key.
        :param pose: A pose the object leaves.
        :return: The first frame the object was away from the pose.
        """
        for index, frame in enumerate(self.frames):
            current = frame.objects.get(key)
            if (
                current is not None
                and self.distance(current, pose) > MOTION_TOLERANCE_METRES
            ):
                return index
        return len(self.frames) - 1

    def last_frame_away(self, key: str, pose: Pose) -> int:
        """
        :param key: An object key.
        :param pose: A pose the object comes to rest at.
        :return: The last frame the object was not yet resting there.
        """
        for index in range(len(self.frames) - 1, -1, -1):
            current = self.frames[index].objects.get(key)
            if (
                current is not None
                and self.distance(current, pose) > MOTION_TOLERANCE_METRES
            ):
                return index
        return 0

    def step_at(self, index: int) -> Optional[str]:
        """
        :param index: A frame.
        :return: The action reported at the frame or last before it.
        """
        for frame in reversed(self.frames[: index + 1]):
            if frame.step:
                return frame.step
        return None

    # %% segments

    def segments(self) -> List[Segment]:
        """
        :return: One segment per carried object, named after the action reported when
            it was picked up, plus the stretches before and after; a run without carried
            objects is divided by the reported actions alone.
        """
        if not self.frames:
            return []
        windows = self.windows()
        if not windows:
            return self.step_segments()
        last_frame = len(self.frames) - 1
        segments: List[Segment] = []
        previous_end = 0
        if windows[0].attach > 0:
            segments.append(
                Segment(
                    step=self.step_at(0) or UnlabelledStep.RUN,
                    start=0,
                    end=windows[0].attach,
                )
            )
            previous_end = windows[0].attach
        for index, window in enumerate(windows):
            following = (
                windows[index + 1].attach if index + 1 < len(windows) else last_frame
            )
            segments.append(
                Segment(
                    step=self.step_at(window.attach)
                    or f"{UnlabelledStep.MOVE}_{window.object_key}",
                    start=previous_end,
                    end=following,
                    picks=window.object_key,
                    attach=window.attach,
                    detach=window.detach,
                )
            )
            previous_end = following
        if previous_end < last_frame:
            segments.append(
                Segment(
                    step=self.step_at(previous_end) or UnlabelledStep.RUN,
                    start=previous_end,
                    end=last_frame,
                )
            )
        return segments

    def step_segments(self) -> List[Segment]:
        """
        :return: One segment per run of frames reporting the same action, or one
            unlabelled segment for a run that reported none.
        """
        segments: List[Segment] = []
        start = 0
        for index in range(1, len(self.frames) + 1):
            if (
                index < len(self.frames)
                and self.frames[index].step == self.frames[start].step
            ):
                continue
            step = self.frames[start].step
            if step:
                segments.append(
                    Segment(
                        step=step, start=start, end=min(index, len(self.frames) - 1)
                    )
                )
            start = index
        return segments or [
            Segment(step=UnlabelledStep.RUN, start=0, end=len(self.frames) - 1)
        ]


# %% trimming


@dataclass(frozen=True)
class SegmentClipping:
    """
    Narrows segments to a kept stretch of the run and counts their frames from its
    start.
    """

    frame_range: FrameRange
    """
    The stretch kept.
    """

    def frame(self, frame: Optional[int]) -> Optional[int]:
        """
        :param frame: A frame of the whole run, or None.
        :return: The frame held within the stretch, counted from its start.
        """
        if frame is None:
            return None
        inside = min(max(frame, self.frame_range.first), self.frame_range.last)
        return inside - self.frame_range.first

    def clip(self, segments: List[Segment]) -> List[Segment]:
        """
        :param segments: The segments of the whole run.
        :return: The segments overlapping the stretch, clamped and rebased.
        """
        return [
            replace(
                segment,
                start=self.frame(segment.start),
                end=self.frame(segment.end),
                attach=self.frame(segment.attach),
                detach=self.frame(segment.detach),
            )
            for segment in segments
            if segment.end >= self.frame_range.first
            and segment.start <= self.frame_range.last
        ]


@dataclass(frozen=True)
class PlanNodeClipping:
    """
    Narrows the frames the nodes of a recorded plan ran in to a kept stretch of the run
    and counts them from its start.
    """

    frame_range: FrameRange
    """
    The stretch kept.
    """

    def clip(self, nodes: List[PlanNode]) -> List[PlanNode]:
        """
        :param nodes: Plan nodes with their frames in the whole run.
        :return: The same nodes and their descendants; a node that ran within the stretch
            keeps its frames clamped and rebased, any other loses them.
        """
        return [self.clip_node(node) for node in nodes]

    def clip_node(self, node: PlanNode) -> PlanNode:
        """
        :param node: A plan node with its frames in the whole run.
        :return: The node and its descendants with their frames within the stretch.
        """
        children = self.clip(node.children)
        if (
            node.start is None
            or node.end is None
            or node.end < self.frame_range.first
            or node.start > self.frame_range.last
        ):
            return replace(node, start=None, end=None, children=children)
        frames = SegmentClipping(self.frame_range)
        return replace(
            node,
            start=frames.frame(node.start),
            end=frames.frame(node.end),
            children=children,
        )
