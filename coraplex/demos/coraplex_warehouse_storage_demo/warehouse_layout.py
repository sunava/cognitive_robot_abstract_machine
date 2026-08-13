"""
Where things stand in the storage warehouse.

Both the model converter and the demo depend on these measurements: the converter turns
them into the URDF's collision geometry, and the demo turns them into pick and place
poses. Keeping them here means a shelf cannot be moved in one place and missed in the
other.

Every value is in warehouse coordinates, with the floor surface at ``z = 0``. The values
that describe the scanned warehouse rather than added fittings were measured from the
source model; :mod:`test_warehouse_layout` re-derives them from the generated URDF.
"""

from __future__ import annotations

from dataclasses import dataclass

# %% the clear space the demo uses

WORKING_AISLE_X = (3.80, 6.30)
"""
Extent along x of the aisle between the east centre rack and the east wall rack.
"""

WORKING_AISLE_Y = (1.30, 10.50)
"""
Extent along y over which that aisle is free of stored goods and equipment.
"""

TARGET_SHELF_X = (6.45, 7.62)
"""
Extent along x of the east wall rack's deck, from its aisle-facing edge inwards.
"""

TARGET_SHELF_CLEAR_Y = (1.52, 3.02)
"""
Extent along y over which the east wall rack's 1.20 m deck carries no stored goods.

The only stretch of any deck that is simultaneously clear, long enough for the incoming
crates, and within the robot's reach.
"""

RACK_SHELF_HEIGHT = 1.20
"""
Top of the shelf the robot stows onto.

The source model's racks have decks at 0.14 m, 1.20 m, 2.25 m and 3.28 m; the 1.20 m
deck is the only reachable one, since the robot's grasps start at about 0.75 m and the
next deck up is at 2.25 m.
"""

# %% the pallet the goods arrive on

PALLET_CENTER = (4.40, 5.90)
"""
Where the incoming pallet stands in the working aisle.
"""

PALLET_FOOTPRINT = (0.80, 1.20)
"""
Footprint of a euro pallet, its long side along y.
"""

PALLET_BASE_HEIGHT = 0.144
"""
Height of a euro pallet.
"""

PALLET_LOAD_HEIGHT = 0.80
"""
Top of the load stacked on the pallet, which is the surface the crates are picked from.

The robot cannot grasp anything standing on the warehouse floor, so the goods have to
arrive at working height for the demo to be a pick at all.
"""

# %% the boxes the converter adds to the scanned model

SHELF_BOARD_THICKNESS = 0.05
"""
Thickness of the collision board standing in for the target shelf's deck.
"""


@dataclass(frozen=True)
class Fitting:
    """
    A box added on top of the scanned model, giving the robot a defined work surface.

    Every fitting becomes collision geometry. A fitting that also names a material is
    drawn as well; one that does not only backs a surface the scanned meshes already
    show, which is why the scanned warehouse itself stays visual-only.
    """

    link_name: str
    """
    Name of the link the box becomes.
    """

    size: tuple[float, float, float]
    """
    Extents of the box along x, y and z.
    """

    center: tuple[float, float, float]
    """
    Center of the box in warehouse coordinates.
    """

    material_name: str | None = None
    """
    Source material the box is drawn in, or ``None`` for collision geometry only.
    """


FITTINGS = (
    Fitting(
        link_name="target_shelf",
        size=(
            TARGET_SHELF_X[1] - TARGET_SHELF_X[0],
            TARGET_SHELF_CLEAR_Y[1] - TARGET_SHELF_CLEAR_Y[0],
            SHELF_BOARD_THICKNESS,
        ),
        center=(
            sum(TARGET_SHELF_X) / 2,
            sum(TARGET_SHELF_CLEAR_Y) / 2,
            RACK_SHELF_HEIGHT - SHELF_BOARD_THICKNESS / 2,
        ),
    ),
    Fitting(
        link_name="incoming_pallet_base",
        size=(*PALLET_FOOTPRINT, PALLET_BASE_HEIGHT),
        center=(*PALLET_CENTER, PALLET_BASE_HEIGHT / 2),
        material_name="kayu",
    ),
    Fitting(
        link_name="incoming_pallet_load",
        size=(*PALLET_FOOTPRINT, PALLET_LOAD_HEIGHT - PALLET_BASE_HEIGHT),
        center=(*PALLET_CENTER, (PALLET_LOAD_HEIGHT + PALLET_BASE_HEIGHT) / 2),
        material_name="kardus",
    ),
)
"""
The shelf the robot stows onto and the loaded pallet the crates arrive on.
"""
