"""
Snapshots of a world's joint values and poses in the trajectory frame format.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from typing_extensions import Optional

from cramera_bridge.geometry import POSE_PRECISION, NumericPose
from cramera_bridge.live.world_model import ObjectKey, WorldModelView
from cramera_bridge.robots import RobotRoster
from cramera_protocol.live_messages import LiveState
from semantic_digital_twin.robots.robot_parts import AbstractRobot

# %% snapshots


@dataclass
class WorldStatePublisher:
    """
    Takes state snapshots on the simulation thread and hands them to HTTP threads.
    """

    latest: LiveState = field(default_factory=LiveState)
    """
    The newest snapshot.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards :attr:`latest`.
    """

    def refresh(
        self,
        model: WorldModelView,
        selected_robot: Optional[AbstractRobot],
        markers_version: int,
    ) -> LiveState:
        """
        Read the world's current state into a new snapshot.

        :param model: The world's current model.
        :param selected_robot: The robot whose root is the snapshot's base.
        :param markers_version: The version of the marker overlay.
        :return: The new snapshot.
        """
        world = model.world
        frames = {
            str(joint.name): round(float(joint.position), POSE_PRECISION)
            for joint in model.joints
        }
        objects = {
            ObjectKey.of(body): NumericPose.of_entity(world, body).published()
            for body in model.objects
        }
        model_bases = {
            identifier: pose.published()
            for identifier, pose in RobotRoster(world).root_poses().items()
        }
        base = (
            None
            if selected_robot is None
            else NumericPose.of_entity(world, selected_robot.root).published()
        )
        with self.lock:
            self.latest = LiveState(
                sequence_number=self.latest.sequence_number + 1,
                frames=frames,
                base=base,
                objects=objects,
                model_bases=model_bases,
                markers_version=markers_version,
            )
            return self.latest

    def state(self) -> LiveState:
        """
        :return: The newest snapshot.
        """
        with self.lock:
            return self.latest
