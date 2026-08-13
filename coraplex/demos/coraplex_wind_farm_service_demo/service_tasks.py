"""
What the robot moves in the wind farm, and where it stands to do it.

Every pose is derived from the measurements in :mod:`service_layout` rather than written
out, so a part cannot end up hovering over a bench that was moved. The module holds no
plan and builds no world, which keeps the poses testable without running the demo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import Color, Scale

from service_layout import (
    DELIVERY_TRAILER,
    PAD_SURFACE_HEIGHT,
    SERVICE_BENCH,
    SERVICE_ROTOR,
    SERVICE_YAW,
)

# %% the robot

PELVIS_HEIGHT_ABOVE_FLOOR = 0.7923
"""
How far the G1's pelvis stands above the surface it stands on, with its leg joints at
zero.

The pelvis is the robot's root, so its ``odom`` has to be lifted by this much plus the
pad's own height for the robot's feet to rest on the pad rather than sink through it.
"""

STANDING_HEIGHT = PAD_SURFACE_HEIGHT + PELVIS_HEIGHT_ABOVE_FLOOR
"""
Height of the robot's root while it stands on the turbine's pad.
"""

STANDING_DISTANCE = 0.6
"""
How far the robot stands from a pose, in meters, opposite its FRONT-facing side.

One distance serves both surfaces here: the higher of the two puts a part's center
0.98 m above the pad, still inside the roughly 1.22 m the G1 reaches while extended
this far forward.
"""

ROBOT_START_POSE = Pose.from_xyz_rpy(12.7, 6.0, STANDING_HEIGHT, yaw=-np.pi / 2)
"""
Where the robot starts, on the pad north of the service area and facing it.
"""

# %% shutting the turbine down while the robot walks up

APPROACH_WAYPOINTS = (
    Pose.from_xyz_rpy(12.7, 4.0, STANDING_HEIGHT, yaw=-np.pi / 2),
    Pose.from_xyz_rpy(12.7, 2.0, STANDING_HEIGHT, yaw=-np.pi / 2),
    Pose.from_xyz_rpy(12.7, 0.0, STANDING_HEIGHT, yaw=-np.pi / 2),
)
"""
The legs the robot walks down the lane between the two surfaces before it starts work.

The turbine is commanded a step further towards its parked position after each leg. The
robot never touches the machine, so its shutdown is not something a plan action can
express; splitting it over the approach is what makes it happen while something else is
being recorded rather than in a single jump.
"""


def turbine_state(progress: float) -> dict[str, float]:
    """
    :param progress: How far the shutdown has run, from ``0.0`` to ``1.0``.
    :return: The yaw and rotor angles the turbine holds at that point.
    """
    return {"yaw": progress * SERVICE_YAW, "rotor": progress * SERVICE_ROTOR}


# %% the parts

PART_SCALE = Scale(0.14, 0.20, 0.16)
"""
Extents of one service part's case.
"""

PART_COLORS = (
    Color(0.90, 0.55, 0.10),
    Color(0.20, 0.45, 0.65),
    Color(0.35, 0.55, 0.25),
)
"""
One colour per part, so each can be followed from the trailer to its place on the bench.
"""

PART_INSET_ON_TRAILER = 0.10
"""
How far a part's center sits inside the trailer's working edge.
"""

PART_INSET_ON_BENCH = 0.20
"""
How far a laid-out part's center sits inside the bench's working edge.
"""


def standing_pose_in_front_of(pose: Pose, distance: float) -> Pose:
    """
    :param pose: The pose the robot should approach from its FRONT-facing side.
    :param distance: How far in front of the pose the robot stands, in meters.
    :return: The pose the robot stands in to reach that pose with a FRONT grasp.
    """
    yaw = float(pose.yaw)
    return Pose.from_xyz_rpy(
        pose.x - distance * np.cos(yaw),
        pose.y - distance * np.sin(yaw),
        STANDING_HEIGHT,
        yaw=yaw,
    )


def evenly_spaced(extent: tuple[float, float], count: int) -> list[float]:
    """
    :param extent: Lower and upper bound of the stretch to spread positions over.
    :param count: How many positions to place.
    :return: The positions, evenly spread with a half gap left at either end.
    """
    lower, upper = extent
    step = (upper - lower) / count
    return [lower + step * (index + 0.5) for index in range(count)]


@dataclass
class ServiceTransfer:
    """
    One part the robot takes off the delivery trailer and lays out on the bench.
    """

    name: str
    """
    Name of the part's body in the world.
    """

    color: Color
    """
    Colour the part is drawn in.
    """

    trailer_pose: Pose
    """
    Where the part waits on the trailer bed.
    """

    bench_pose: Pose
    """
    Where the part ends up on the bench.
    """

    @property
    def trailer_standing_pose(self) -> Pose:
        """
        :return: Where the robot stands to take the part off the trailer.
        """
        return standing_pose_in_front_of(self.trailer_pose, STANDING_DISTANCE)

    @property
    def bench_standing_pose(self) -> Pose:
        """
        :return: Where the robot stands to lay the part out on the bench.
        """
        return standing_pose_in_front_of(self.bench_pose, STANDING_DISTANCE)

    @property
    def turn_towards_bench(self) -> float:
        """
        :return: The relative yaw, wrapped to ``[-pi, pi]``, turning the robot from the
            trailer approach towards the bench approach before it drives off.
        """
        difference = float(self.bench_pose.yaw) - float(self.trailer_pose.yaw)
        return float(np.arctan2(np.sin(difference), np.cos(difference)))


def resting_pose_on(surface, y: float, inset: float) -> Pose:
    """
    :param surface: The service surface the part rests on.
    :param y: Where along the surface the part sits.
    :param inset: How far inside the surface's working edge the part sits.
    :return: The pose of a part resting on that surface, facing the robot.
    """
    return Pose.from_xyz_rpy(
        surface.inset_x(inset),
        y,
        surface.top_height + PART_SCALE.z / 2,
        yaw=surface.approach_yaw,
    )


def build_service_transfers() -> list[ServiceTransfer]:
    """
    :return: One transfer per delivered part, spread along the trailer bed and along the
        bench in the same order.
    """
    positions = zip(
        PART_COLORS,
        evenly_spaced(DELIVERY_TRAILER.extent_y, len(PART_COLORS)),
        evenly_spaced(SERVICE_BENCH.extent_y, len(PART_COLORS)),
    )
    return [
        ServiceTransfer(
            name="service_part_%d" % (index + 1),
            color=color,
            trailer_pose=resting_pose_on(
                DELIVERY_TRAILER, trailer_y, PART_INSET_ON_TRAILER
            ),
            bench_pose=resting_pose_on(SERVICE_BENCH, bench_y, PART_INSET_ON_BENCH),
        )
        for index, (color, trailer_y, bench_y) in enumerate(positions)
    ]


SERVICE_TRANSFERS = build_service_transfers()
"""
The three parts to lay out, in the order the robot works through them.
"""
