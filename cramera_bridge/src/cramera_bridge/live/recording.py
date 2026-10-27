"""
Recording a live run frame by frame, so it can be bundled and replayed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from typing_extensions import Callable, Dict, List, Optional

from cramera_bridge.exceptions import InvalidFrameRange, NoRecording
from cramera_protocol.geometry import Pose
from cramera_protocol.live_messages import (
    LiveState,
    LiveStatechart,
    RecordingState,
    RecordingStatus,
)

DURATION_PRECISION = 2
"""
Decimal places of a published recording duration and frame rate.
"""

MINIMUM_FRAMES_PER_SECOND = 1.0
"""
The lowest frame rate a recording reports, so a long pause never makes it zero.
"""

# %% frames


@dataclass(frozen=True)
class FrameRange:
    """
    A stretch of a recording; both ends are kept.
    """

    first: int
    """
    The first frame kept.
    """

    last: int
    """
    The last frame kept.
    """

    def require_within(self, frame_count: int) -> None:
        """
        :param frame_count: How many frames the recording holds.
        :raises InvalidFrameRange: When the range runs backwards or past the frames.
        """
        if self.first < 0 or self.last < self.first or self.last >= frame_count:
            raise InvalidFrameRange(self.first, self.last, frame_count)

    def length(self) -> int:
        """
        :return: How many frames the range keeps.
        """
        return self.last - self.first + 1


@dataclass(frozen=True)
class RunningAction:
    """
    The action a plan performs at one moment: its deepest running action node.
    """

    identifier: str
    """
    The identity of the action's node in the published plan.
    """

    label: str
    """
    The class name of the action's designator.
    """


@dataclass(frozen=True)
class RecordedFrame:
    """
    One recorded moment of the run; its poses are copies the world cannot change.
    """

    time: float
    """
    The wall-clock time of the frame, in seconds since the epoch.
    """

    frames: Dict[str, float]
    """
    The value of every movable joint by its prefixed name.
    """

    base: Optional[Pose]
    """
    The selected robot's root pose.
    """

    objects: Dict[str, Pose]
    """
    The pose of every free object by its key.
    """

    model_bases: Dict[str, Pose]
    """
    The root pose of every robot by its identity.
    """

    step: Optional[str] = None
    """
    The action the plan performed, or None between actions.
    """

    statechart: Optional[LiveStatechart] = None
    """
    The statechart being ticked, or None while no motion ran.
    """

    action_node: Optional[str] = None
    """
    The identity of the plan node of the action the plan performed, or None between
    actions.
    """

    @classmethod
    def of_state(
        cls,
        state: LiveState,
        time: float,
        action: Optional[RunningAction],
        statechart: Optional[LiveStatechart],
    ) -> RecordedFrame:
        """
        :param state: A state snapshot.
        :param time: When it was taken.
        :param action: The action being performed.
        :param statechart: The statechart being ticked.
        :return: The frame, with copies of the snapshot's mappings.
        """
        return cls(
            time=time,
            frames=dict(state.frames),
            base=state.base,
            objects=dict(state.objects),
            model_bases=dict(state.model_bases),
            step=None if action is None else action.label,
            statechart=statechart,
            action_node=None if action is None else action.identifier,
        )


# %% recording


@dataclass
class Recording:
    """
    One run's frames; the simulation thread appends, HTTP threads read and end it.
    """

    clock: Callable[[], float] = time.time
    """
    The wall clock frames are stamped with.
    """

    state: RecordingState = RecordingState.IDLE
    """
    Where the recording is in its life.
    """

    frames: List[RecordedFrame] = field(default_factory=list)
    """
    The frames in recording order.
    """

    scene_name: Optional[str] = None
    """
    The bundle the frames were written to, once they were.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards every field.
    """

    def start(self) -> None:
        """
        Start recording, dropping whatever an earlier recording held.
        """
        with self.lock:
            self.state = RecordingState.RECORDING
            self.frames = []
            self.scene_name = None

    def append(
        self,
        state: LiveState,
        action: Optional[RunningAction],
        statechart: Optional[LiveStatechart],
    ) -> None:
        """
        Record one moment, or nothing while not recording.

        :param state: The state snapshot.
        :param action: The action being performed.
        :param statechart: The statechart being ticked.
        """
        frame = RecordedFrame.of_state(state, self.clock(), action, statechart)
        with self.lock:
            if self.state is RecordingState.RECORDING:
                self.frames.append(frame)

    def finish(self) -> List[RecordedFrame]:
        """
        End the recording; ending it again returns the same frames.

        :return: The frames.
        :raises NoRecording: When nothing was recorded.
        """
        with self.lock:
            if self.state is RecordingState.IDLE:
                raise NoRecording(RecordingState.RECORDING)
            self.state = RecordingState.FINALIZED
            return list(self.frames)

    def discard(self) -> None:
        """
        Drop the frames and become idle.
        """
        with self.lock:
            self.state = RecordingState.IDLE
            self.frames = []
            self.scene_name = None

    def written(self, scene_name: str) -> None:
        """
        :param scene_name: The bundle the frames were written to.
        """
        with self.lock:
            self.scene_name = scene_name

    def frame_count(self) -> int:
        """
        :return: How many frames were recorded.
        """
        with self.lock:
            return len(self.frames)

    def status(self) -> RecordingStatus:
        """
        :return: The recording's state, length and measured frame rate.
        """
        with self.lock:
            return RecordingStatus(
                state=self.state,
                frame_count=len(self.frames),
                duration_seconds=round(
                    FrameTiming(self.frames).duration(), DURATION_PRECISION
                ),
                frames_per_second=FrameTiming(self.frames).frames_per_second(),
                scene_name=self.scene_name,
            )


@dataclass(frozen=True)
class FrameTiming:
    """
    How long a list of frames lasts and how fast they came.
    """

    frames: List[RecordedFrame]
    """
    The frames in recording order.
    """

    def duration(self) -> float:
        """
        :return: The seconds between the first and the last frame.
        """
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].time - self.frames[0].time

    def frames_per_second(self) -> Optional[float]:
        """
        :return: The measured frame rate, or None with fewer than two frames or no
            time between them.
        """
        duration = self.duration()
        if duration <= 0:
            return None
        return max(
            MINIMUM_FRAMES_PER_SECOND,
            round(len(self.frames) / duration, DURATION_PRECISION),
        )
