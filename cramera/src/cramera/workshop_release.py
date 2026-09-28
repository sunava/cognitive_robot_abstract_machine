"""
Validate the complete native workshop recording before publication.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing_extensions import ClassVar


# %% portable recording contract
class InvalidWorkshopRelease(ValueError):
    """
    A recording lacks complete, consistent native execution evidence.
    """


class RecordingFile(StrEnum):
    """
    Documents that must belong to the same completed recording.
    """

    SCENE = "scene.json"
    ACCEPTANCE = "acceptance_metrics.json"
    TRAJECTORY = "trajectory.json"


class RecordingField(StrEnum):
    """
    Fields supplied by native workshop recording and acceptance documents.
    """

    NAME = "name"
    DEMO = "workshopDemo"
    KIND = "kind"
    STEPS = "steps"
    RANGES = "stepRanges"
    START = "startFrame"
    END = "endFrame"
    RESULT = "result"
    SUCCESS = "success"
    STATUSES = "action_statuses"
    BOX_ERROR = "box_position_error"
    DEPTH = "screw_depth"
    REQUIRED_DEPTH = "required_depth"
    DRIVER_RELEASED = "driver_released"
    SUPPORT_RELEASED = "support_released"
    DISTANCE = "minimum_collision_distance"
    FRAMES = "frames"
    RATE = "framesPerSecond"
    TRAJECTORY = "trajectory"
    CONTENTS = "contentsPhysics"
    MODEL = "model"
    TICKS = "ticks"
    SECONDS = "simulated_seconds"
    POSITION_LAG = "maximum_box_position_lag"
    ANGULAR_LAG = "maximum_box_angular_lag"
    DISPLACEMENT = "maximum_relative_displacement"
    CORRECTION = "maximum_projection_correction"
    RETAINED = "retained_contents"
    ESCAPED = "escaped_contents"
    CONTENTS_COUNT = "contents_count"
    OBJECTS = "objects"
    BASE = "base"
    MISSING_ASSETS = "missingAssets"


class RecordingValue(StrEnum):
    """
    Required native execution and one-way contact model identifiers.
    """

    SCENE = "aicor_workshop_pr2"
    NATIVE = "native-recording"
    SUCCEEDED = "SUCCEEDED"
    CONTACT = "one-way-relative-contact-contents"
    BOX = "screw_box"
    DRIVER = "driver"


# %% publication validation
@dataclass(frozen=True)
class WorkshopRelease:
    """
    Check a frozen recording without importing robotics or contacting a server.
    """

    directory: Path
    """
    Snapshot directory that will be copied unchanged to the release.
    """

    required_stages: ClassVar[int] = 10
    """
    Number of actions in the complete retrieval and two-handed assembly plan.
    """

    required_contents: ClassVar[int] = 5
    """
    Number of independent screws transported in the carrier.
    """

    controller_frequency: ClassVar[float] = 50.0
    """
    Native control ticks recorded per simulated second.
    """

    @staticmethod
    def require(condition: bool, message: str) -> None:
        """
        Stop publication at the first unmet recording requirement.
        """
        if not condition:
            raise InvalidWorkshopRelease(message)

    @staticmethod
    def number(value: object) -> bool:
        """
        Reject booleans and non-finite values in numerical measurements.
        """
        return type(value) is int or type(value) is float and math.isfinite(value)

    @classmethod
    def finite(cls, value: object) -> bool:
        """
        Check every nested JSON number, including trajectory coordinates.
        """
        if isinstance(value, dict):
            return all(cls.finite(item) for item in value.values())
        if isinstance(value, list):
            return all(cls.finite(item) for item in value)
        return not isinstance(value, float) or math.isfinite(value)

    def read(self, filename: RecordingFile) -> dict:
        """
        Load one finite object document from the frozen export.
        """
        path = self.directory / filename
        self.require(path.is_file(), f"Missing recording document: {filename}")
        document = json.loads(path.read_text())
        self.require(
            isinstance(document, dict) and self.finite(document),
            f"Invalid or non-finite recording document: {filename}",
        )
        return document

    def validate(self) -> None:
        """
        Require matching acceptance, ten stage intervals and measured frame data.
        """
        scene = self.read(RecordingFile.SCENE)
        metrics = self.read(RecordingFile.ACCEPTANCE)
        trajectory = self.read(RecordingFile.TRAJECTORY)
        self.require(
            scene.get(RecordingField.NAME) == RecordingValue.SCENE
            and scene.get(RecordingField.TRAJECTORY) == RecordingFile.TRAJECTORY
            and not scene.get(RecordingField.MISSING_ASSETS, []),
            "The recording must contain the complete native workshop scene.",
        )
        demo = scene.get(RecordingField.DEMO, {})
        self.require(
            isinstance(demo, dict)
            and demo.get(RecordingField.KIND) == RecordingValue.NATIVE
            and demo.get(RecordingField.RESULT) == metrics,
            "Scene metadata and native acceptance must describe the same recording.",
        )
        self.validate_acceptance(metrics)
        self.validate_contents(metrics.get(RecordingField.CONTENTS, {}))
        self.validate_stages(demo, metrics[RecordingField.FRAMES])
        self.validate_frames(scene, trajectory, metrics[RecordingField.FRAMES])

    def validate_acceptance(self, metrics: dict) -> None:
        """
        Recheck measured outcomes rather than trusting the success flag alone.
        """
        fields = (
            RecordingField.BOX_ERROR,
            RecordingField.DEPTH,
            RecordingField.REQUIRED_DEPTH,
            RecordingField.DISTANCE,
        )
        self.require(
            all(self.number(metrics.get(key)) for key in fields),
            "Acceptance requires finite measured errors, depth and collision distance.",
        )
        self.require(
            metrics.get(RecordingField.SUCCESS) is True
            and metrics.get(RecordingField.STATUSES)
            == [RecordingValue.SUCCEEDED] * self.required_stages
            and 0 <= metrics[RecordingField.BOX_ERROR] < 0.003
            and metrics[RecordingField.DEPTH] >= 0
            and metrics[RecordingField.REQUIRED_DEPTH] > 0
            and abs(
                metrics[RecordingField.DEPTH] - metrics[RecordingField.REQUIRED_DEPTH]
            )
            < 0.002
            and metrics[RecordingField.DRIVER_RELEASED] is True
            and metrics[RecordingField.SUPPORT_RELEASED] is True
            and metrics[RecordingField.DISTANCE] >= -0.0005
            and type(metrics.get(RecordingField.FRAMES)) is int
            and metrics[RecordingField.FRAMES] > 0,
            "All ten actions must succeed with a seated screw, released hands and valid clearances.",
        )

    def validate_contents(self, contents: dict) -> None:
        """
        Require measured retention of every independently simulated screw.
        """
        fields = (
            RecordingField.SECONDS,
            RecordingField.POSITION_LAG,
            RecordingField.ANGULAR_LAG,
            RecordingField.DISPLACEMENT,
            RecordingField.CORRECTION,
        )
        self.require(
            isinstance(contents, dict)
            and all(
                self.number(contents.get(key)) and contents[key] >= 0 for key in fields
            )
            and contents.get(RecordingField.MODEL) == RecordingValue.CONTACT
            and contents.get(RecordingField.RETAINED) is True
            and contents.get(RecordingField.ESCAPED) == []
            and contents.get(RecordingField.CONTENTS_COUNT) == self.required_contents
            and type(contents.get(RecordingField.TICKS)) is int
            and contents[RecordingField.TICKS] > 0
            and contents[RecordingField.SECONDS] > 0,
            "All five screws require finite one-way contact measurements and confirmed retention.",
        )

    def validate_stages(self, demo: dict, count: int) -> None:
        """
        Require nonempty consecutive stage intervals covering every recorded tick.
        """
        steps = demo.get(RecordingField.STEPS)
        ranges = demo.get(RecordingField.RANGES)
        self.require(
            isinstance(steps, list)
            and len(steps) == self.required_stages
            and all(isinstance(step, str) and step.strip() for step in steps)
            and len(set(steps)) == len(steps)
            and isinstance(ranges, list)
            and len(ranges) == len(steps),
            "The replay must present all ten distinct native actions.",
        )
        expected_start = 0
        for step, interval in zip(steps, ranges):
            self.require(
                isinstance(interval, dict)
                and interval.get(RecordingField.NAME) == step
                and type(interval.get(RecordingField.START)) is int
                and type(interval.get(RecordingField.END)) is int
                and interval[RecordingField.START] == expected_start
                and expected_start <= interval[RecordingField.END] < count,
                "Recorded stage boundaries must follow the actual controller frames.",
            )
            expected_start = interval[RecordingField.END] + 1
        self.require(
            expected_start == count, "Stage intervals must cover the complete replay."
        )

    def valid_pose(self, pose: object) -> bool:
        """
        Check finite translation and a unit quaternion in recorded xyzw order.
        """
        return bool(
            isinstance(pose, list)
            and len(pose) == 7
            and all(self.number(value) for value in pose)
            and abs(sum(value * value for value in pose[3:]) - 1.0) < 0.01
        )

    def validate_frames(self, scene: dict, trajectory: dict, count: int) -> None:
        """
        Require one valid robot and object snapshot per accepted controller tick.
        """
        self.require(
            scene.get(RecordingField.RATE) == self.controller_frequency
            and trajectory.get(RecordingField.RATE) == self.controller_frequency,
            "The recording must preserve the native 50 Hz controller timing.",
        )
        for field in (
            RecordingField.FRAMES,
            RecordingField.BASE,
            RecordingField.OBJECTS,
        ):
            self.require(
                isinstance(trajectory.get(field), list)
                and len(trajectory[field]) == count,
                "Trajectory lengths must match the accepted controller frame count.",
            )
        required_objects = {
            RecordingValue.BOX,
            RecordingValue.DRIVER,
            *(f"screw_{index + 1}" for index in range(self.required_contents)),
        }
        for joints, base, objects in zip(
            trajectory[RecordingField.FRAMES],
            trajectory[RecordingField.BASE],
            trajectory[RecordingField.OBJECTS],
        ):
            self.require(
                isinstance(joints, dict)
                and bool(joints)
                and all(self.number(value) for value in joints.values())
                and self.valid_pose(base)
                and isinstance(objects, dict)
                and required_objects <= objects.keys()
                and all(self.valid_pose(pose) for pose in objects.values()),
                "Every frame must contain valid robot, carrier, tool and five-screw poses.",
            )
