"""
What the robot moves in the storage warehouse, and where it stands to do it.

Every pose here is derived from the measurements in :mod:`warehouse_layout` rather than
written out, so a crate cannot end up hovering over a shelf that was moved. The module
holds no plan and builds no world, which keeps the poses testable without running the
demo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import Color, Scale

from warehouse_layout import (
    PALLET_CENTER,
    PALLET_FOOTPRINT,
    PALLET_LOAD_HEIGHT,
    RACK_SHELF_HEIGHT,
    TARGET_SHELF_CLEAR_Y,
    TARGET_SHELF_X,
)

# %% the robot

PELVIS_HEIGHT_ABOVE_FLOOR = 0.7923
"""
How far the G1's pelvis stands above the floor with all of its leg joints at zero.

The pelvis is the robot's root, so its ``odom`` has to be lifted by this much for the
robot's feet to rest on the floor rather than sink through it.
"""

ROBOT_START_POSE = Pose.from_xyz_rpy(5.30, 8.00, PELVIS_HEIGHT_ABOVE_FLOOR)
"""
Where the robot starts, in the working aisle north of the incoming pallet.
"""

PALLET_STANDING_DISTANCE = 0.6
"""
How far the robot stands from a crate on the pallet, opposite its FRONT-facing side.

Within the G1's reach, and far enough from the pallet to leave its footprint free.
"""

SHELF_STANDING_DISTANCE = 0.5
"""
How far the robot stands from a crate's place on the shelf.

Closer than at the pallet, because the two limits of the G1's reach trade off against
each other: reaching 0.6 m forward it grasps no higher than about 1.22 m, which is below
the 1.28 m a crate's center sits at once it rests on the 1.20 m shelf. Stepping in to
0.5 m buys the height back.
"""

# %% the crates

CRATE_SCALE = Scale(0.14, 0.20, 0.16)
"""
Extents of one incoming crate.
"""

CRATE_COLORS = (
    Color(0.85, 0.45, 0.10),
    Color(0.20, 0.45, 0.65),
    Color(0.35, 0.55, 0.25),
)
"""
One colour per crate, so each can be followed from the pallet to its place on the shelf.
"""

CRATE_INSET_ON_PALLET = 0.10
"""
How far a crate's center sits inside the pallet's aisle-facing edge.
"""

CRATE_INSET_ON_SHELF = 0.20
"""
How far a stowed crate's center sits inside the shelf's aisle-facing edge.
"""

PALLET_APPROACH_YAW = np.pi
"""
The pallet is picked from the aisle side, so the robot stands east of it facing west.
"""

SHELF_APPROACH_YAW = 0.0
"""
The shelf is loaded from the aisle side, so the robot stands west of it facing east.
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
        PELVIS_HEIGHT_ABOVE_FLOOR,
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
class StowTask:
    """
    One crate the robot takes off the incoming pallet and puts away on the shelf.
    """

    name: str
    """
    Name of the crate's body in the world.
    """

    color: Color
    """
    Colour the crate is drawn in.
    """

    pick_pose: Pose
    """
    Where the crate waits on the pallet load.
    """

    shelf_pose: Pose
    """
    Where the crate ends up on the rack shelf.
    """

    @property
    def pallet_standing_pose(self) -> Pose:
        """
        :return: Where the robot stands to take the crate off the pallet.
        """
        return standing_pose_in_front_of(self.pick_pose, PALLET_STANDING_DISTANCE)

    @property
    def shelf_standing_pose(self) -> Pose:
        """
        :return: Where the robot stands to put the crate on the shelf.
        """
        return standing_pose_in_front_of(self.shelf_pose, SHELF_STANDING_DISTANCE)

    @property
    def turn_towards_shelf(self) -> float:
        """
        :return: The relative yaw, wrapped to ``[-pi, pi]``, turning the robot from the
            pallet approach towards the shelf approach before it drives off.
        """
        difference = float(self.shelf_pose.yaw) - float(self.pick_pose.yaw)
        return float(np.arctan2(np.sin(difference), np.cos(difference)))


def build_stow_tasks() -> list[StowTask]:
    """
    :return: One task per incoming crate, spread along the pallet and along the free
        stretch of shelf in the same order.
    """
    pallet_edge_x = PALLET_CENTER[0] + PALLET_FOOTPRINT[0] / 2
    pallet_extent_y = (
        PALLET_CENTER[1] - PALLET_FOOTPRINT[1] / 2,
        PALLET_CENTER[1] + PALLET_FOOTPRINT[1] / 2,
    )
    positions = zip(
        CRATE_COLORS,
        evenly_spaced(pallet_extent_y, len(CRATE_COLORS)),
        evenly_spaced(TARGET_SHELF_CLEAR_Y, len(CRATE_COLORS)),
    )
    return [
        StowTask(
            name="incoming_crate_%d" % (index + 1),
            color=color,
            pick_pose=Pose.from_xyz_rpy(
                pallet_edge_x - CRATE_INSET_ON_PALLET,
                pallet_y,
                PALLET_LOAD_HEIGHT + CRATE_SCALE.z / 2,
                yaw=PALLET_APPROACH_YAW,
            ),
            shelf_pose=Pose.from_xyz_rpy(
                TARGET_SHELF_X[0] + CRATE_INSET_ON_SHELF,
                shelf_y,
                RACK_SHELF_HEIGHT + CRATE_SCALE.z / 2,
                yaw=SHELF_APPROACH_YAW,
            ),
        )
        for index, (color, pallet_y, shelf_y) in enumerate(positions)
    ]


STOW_TASKS = build_stow_tasks()
"""
The three crates to put away, in the order the robot works through them.
"""
