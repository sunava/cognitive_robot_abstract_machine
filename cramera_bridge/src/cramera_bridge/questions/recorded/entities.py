"""
The entities of a recorded scene a query ranges over.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from typing_extensions import Optional

from cramera_protocol.bundle.scene import PartSide
from cramera_bridge.questions.entity import NamedEntity

# %% values


@dataclass(frozen=True)
class Position:
    """
    A point in the scene, in metres.
    """

    x: float
    """
    The x coordinate.
    """

    y: float
    """
    The y coordinate.
    """

    z: float
    """
    The z coordinate.
    """


# %% robot


@dataclass(unsafe_hash=True)
class RecordedRobot(NamedEntity):
    """
    The robot that performed the recorded run.
    """

    arm_count: int
    """
    How many arms the robot's annotations name.
    """


@dataclass(unsafe_hash=True)
class RecordedGripper(NamedEntity):
    """
    An end effector of the recorded robot.
    """

    side: Optional[PartSide]
    """
    The side of the arm carrying it, or None for a robot naming no sides.
    """


@dataclass(unsafe_hash=True)
class RecordedArm(NamedEntity):
    """
    An arm of the recorded robot.
    """

    side: Optional[PartSide]
    """
    The side the robot names the arm on, or None for a robot naming no sides.
    """

    robot: str
    """
    The name of the robot the arm belongs to.
    """

    gripper: Optional[RecordedGripper]
    """
    The end effector the arm carries, when the annotations name one.
    """


# %% scene


@dataclass(unsafe_hash=True)
class RecordedObject(NamedEntity):
    """
    A free object of the recorded scene.
    """

    key: str
    """
    The object's key in the trajectory's object poses.
    """

    height_metres: Optional[float]
    """
    The object's recorded height, or None when the recording measured none.
    """

    position: Position
    """
    Where the object stood before the run.
    """


@dataclass(unsafe_hash=True)
class RecordedStep(NamedEntity):
    """
    One step of the recorded run on its timeline.
    """

    index: int
    """
    The step's position in the run.
    """

    start_frame: int
    """
    The step's first frame.
    """

    end_frame: int
    """
    The step's last frame.
    """

    duration_seconds: Optional[float]
    """
    How long the step took, or None for a recording without a frame rate.
    """

    action: Optional[str]
    """
    The actions the step performed, as the recorder labelled them.
    """

    performed_by: Optional[RecordedArm]
    """
    The arm the recorder labelled the step with, when it names one of the robot's arms.
    """

    picks: Optional[RecordedObject]
    """
    The object the step picked up.
    """


@dataclass(unsafe_hash=True)
class JointMotion(NamedEntity):
    """
    How far one joint moved over the recorded run.
    """

    part: Optional[str]
    """
    The robot part whose links the joint moves, or None for a joint outside every part.
    """

    minimum: float
    """
    The smallest recorded value, in radians or metres.
    """

    maximum: float
    """
    The largest recorded value, in radians or metres.
    """

    travelled: float
    """
    The range between them.
    """


@dataclass(unsafe_hash=True)
class RecordedPlanNode(NamedEntity):
    """
    One node of the plan the run executed.
    """

    kind: str
    """
    The class name of the plan node.
    """

    label: str
    """
    The node's description.
    """

    status: str
    """
    The name of its native status when the run ended.
    """

    parent: Optional[str]
    """
    The name of its parent node, or None for a root.
    """

    target: Optional[str]
    """
    The object or place it acted on.
    """

    arm: Optional[str]
    """
    The arm it used.
    """


@dataclass(unsafe_hash=True)
class DetectedEventRecord(NamedEntity):
    """
    One event a detector saw during the run.
    """

    event_type: str
    """
    The event's class name.
    """

    timestamp: datetime
    """
    When the event happened.
    """

    tracked_object: Optional[str]
    """
    The object the event is about.
    """

    with_object: Optional[str]
    """
    The second object of a relation event.
    """
