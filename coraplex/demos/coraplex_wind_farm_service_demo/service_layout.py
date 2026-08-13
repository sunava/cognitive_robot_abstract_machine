"""
Where things stand in the wind farm's service area.

The wind farm model is a MuJoCo scene of 25 turbines: towers, nacelles, hubs and blades,
over a kilometre of ground and up to 300 m tall. Nothing in it is at human scale and
nothing in it carries collision geometry, so a demo that manipulates anything has to
bring its own equipment. These measurements say where that equipment stands, and the
ones describing the turbine itself were read out of the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from semantic_digital_twin.world_description.geometry import Color, Scale

# %% the turbine the demo services

SERVICE_TURBINE = "Farm_Big_tall"
"""
Name prefix of the turbine the demo works on, the 300 m machine at the farm's centre.
"""

PAD_SURFACE_HEIGHT = 0.20
"""
Top of the turbine's concrete pad, which is the surface the robot stands and works on.

The model's pad geom is a box of half-height 0.1 whose body sits at ``z = 0.1``, so the
pad spans 0 m to 0.2 m and every working height in this module is measured against the
world floor, not against the pad.
"""

PAD_HALF_EXTENT = 50.0
"""
Half the width of the pad, which is square.
"""

TOWER_RADIUS = 8.55542
"""
Radius of the tower, centred on the pad, which the service area has to stay clear of.
"""

YAW_CONNECTION = "%s_yaw" % SERVICE_TURBINE
"""
The joint that turns the nacelle.
"""

ROTOR_CONNECTION = "%s_rotor" % SERVICE_TURBINE
"""
The joint the rotor turns on.
"""

SERVICE_YAW = np.pi / 2
"""
Nacelle angle for service, a quarter turn out of the running position.
"""

SERVICE_ROTOR = np.pi
"""
Rotor angle for service.

Half a turn from the running position puts one blade straight down, which is the
position a turbine is parked and locked in before anyone works on it.
"""

# %% the equipment the demo brings to the pad


class WorkingEdge(Enum):
    """
    The side of a service surface the robot works from.
    """

    EAST = auto()
    """
    The robot stands east of the surface and faces west.
    """

    WEST = auto()
    """
    The robot stands west of the surface and faces east.
    """


@dataclass(frozen=True)
class ServiceSurface:
    """
    A box standing on the pad that gives the robot something to work on.

    Modeled as a solid block from the pad up to its working height rather than as a real
    trailer or bench, because only its top surface and its footprint matter: the robot
    picks off it, places onto it, and must not walk into it.
    """

    name: str
    """
    Name of the surface's body in the world.
    """

    center: tuple[float, float]
    """
    Where the surface stands on the pad.
    """

    footprint: tuple[float, float]
    """
    Extents of the surface along x and y.
    """

    top_height: float
    """
    Height of the surface's working top above the world floor.
    """

    working_edge: WorkingEdge
    """
    The side the robot approaches from.
    """

    color: Color
    """
    Colour the surface is drawn in.
    """

    @property
    def scale(self) -> Scale:
        """
        :return: The extents of the box, which stands on the pad.
        """
        return Scale(
            self.footprint[0], self.footprint[1], self.top_height - PAD_SURFACE_HEIGHT
        )

    @property
    def box_center(self) -> tuple[float, float, float]:
        """
        :return: Where the box's center sits with the box resting on the pad.
        """
        return (*self.center, (self.top_height + PAD_SURFACE_HEIGHT) / 2)

    @property
    def working_edge_x(self) -> float:
        """
        :return: The x of the edge the robot reaches over.
        """
        reach_side = 1 if self.working_edge is WorkingEdge.EAST else -1
        return self.center[0] + reach_side * self.footprint[0] / 2

    @property
    def approach_yaw(self) -> float:
        """
        :return: The yaw the robot faces while working at this surface.
        """
        return np.pi if self.working_edge is WorkingEdge.EAST else 0.0

    @property
    def extent_y(self) -> tuple[float, float]:
        """
        :return: The stretch along y that items can be put down on.
        """
        return (
            self.center[1] - self.footprint[1] / 2,
            self.center[1] + self.footprint[1] / 2,
        )

    def inset_x(self, inset: float) -> float:
        """
        :param inset: How far inside the working edge to sit, in meters.
        :return: The x an item takes when it is set down that far in from the edge.
        """
        reach_side = 1 if self.working_edge is WorkingEdge.EAST else -1
        return self.working_edge_x - reach_side * inset


DELIVERY_TRAILER = ServiceSurface(
    name="delivery_trailer",
    center=(14.5, 0.0),
    footprint=(1.0, 2.4),
    top_height=1.00,
    working_edge=WorkingEdge.WEST,
    color=Color(0.45, 0.45, 0.48),
)
"""
The flatbed the service parts arrive on, parked on the pad away from the tower.
"""

SERVICE_BENCH = ServiceSurface(
    name="service_bench",
    center=(11.0, 0.0),
    footprint=(0.8, 2.0),
    top_height=1.10,
    working_edge=WorkingEdge.EAST,
    color=Color(0.35, 0.38, 0.42),
)
"""
The bench at the tower door the parts are laid out on, ready for the technicians.
"""

SERVICE_SURFACES = (DELIVERY_TRAILER, SERVICE_BENCH)
"""
Everything the demo stands on the pad, in the order it is spawned.
"""

WORKING_LANE_X = (
    SERVICE_BENCH.working_edge_x,
    DELIVERY_TRAILER.working_edge_x,
)
"""
The stretch along x between the two surfaces, which is where the robot works.
"""
