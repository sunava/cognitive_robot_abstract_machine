"""
Placement geometry of the cutting board the tool-based-action demos put under a cut
object.

These pin the board's size and pose against a synthetic object of known dimensions,
independent of the PR2/apartment simulation stack the demos themselves run against.
"""

from __future__ import annotations

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from experiments.tool_based_actions.simple_demo.demo_world import (
    CUTTING_BOARD_MARGIN,
    CUTTING_BOARD_THICKNESS,
    add_cutting_board,
)

CUT_OBJECT_SCALE = Scale(0.1, 0.06, 0.05)
"""
Footprint of the synthetic object the board is sized against in these tests.
"""

CUT_OBJECT_POSITION_XYZ = (1.0, 2.0, 0.9)
"""
Position the synthetic object's origin would occupy resting on the counter without a
board. The object's origin is at its geometric center (like the real bread mesh, whose
origin is not at its bottom face either), so its bottom sits half its height below this.
"""

CUT_OBJECT_BOTTOM_Z_OFFSET = -CUT_OBJECT_SCALE.z / 2
"""
Offset from the synthetic object's origin down to its bottom face.
"""


@pytest.fixture()
def world_with_root() -> World:
    """A world with a single root body, playing the role of the counter."""
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("counter")))
    return world


@pytest.fixture()
def cut_object_world() -> World:
    """A not-yet-merged world containing one box-shaped stand-in for a cut object."""
    cut_object_world = World()
    with cut_object_world.modify_world():
        cut_object_world.add_body(
            Body(
                name=PrefixedName("cut_object"),
                collision=ShapeCollection([Box(scale=CUT_OBJECT_SCALE)]),
            )
        )
    return cut_object_world


def board_bounding_box(world: World, board):
    return board.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()


class TestAddCuttingBoard:
    def test_registers_the_board_under_its_name(
        self, world_with_root, cut_object_world
    ):
        board = add_cutting_board(
            world_with_root, cut_object_world, CUT_OBJECT_POSITION_XYZ
        )

        assert world_with_root.get_body_by_name("cutting_board") is board

    def test_footprint_is_the_object_footprint_plus_margin_on_each_side(
        self, world_with_root, cut_object_world
    ):
        board = add_cutting_board(
            world_with_root, cut_object_world, CUT_OBJECT_POSITION_XYZ
        )

        bounding_box = board_bounding_box(world_with_root, board)
        assert bounding_box.max_x - bounding_box.min_x == pytest.approx(
            CUT_OBJECT_SCALE.x + 2 * CUTTING_BOARD_MARGIN
        )
        assert bounding_box.max_y - bounding_box.min_y == pytest.approx(
            CUT_OBJECT_SCALE.y + 2 * CUTTING_BOARD_MARGIN
        )
        assert bounding_box.max_z - bounding_box.min_z == pytest.approx(
            CUTTING_BOARD_THICKNESS
        )

    def test_bottom_surface_sits_where_the_objects_bottom_would_otherwise_be(
        self, world_with_root, cut_object_world
    ):
        board = add_cutting_board(
            world_with_root, cut_object_world, CUT_OBJECT_POSITION_XYZ
        )

        bounding_box = board_bounding_box(world_with_root, board)
        object_bottom_z = CUT_OBJECT_POSITION_XYZ[2] + CUT_OBJECT_BOTTOM_Z_OFFSET
        assert bounding_box.min_z == pytest.approx(object_bottom_z)
        assert bounding_box.max_z == pytest.approx(
            object_bottom_z + CUTTING_BOARD_THICKNESS
        )

    def test_is_centered_on_the_requested_xy(self, world_with_root, cut_object_world):
        board = add_cutting_board(
            world_with_root, cut_object_world, CUT_OBJECT_POSITION_XYZ
        )

        bounding_box = board_bounding_box(world_with_root, board)
        assert (bounding_box.min_x + bounding_box.max_x) / 2 == pytest.approx(
            CUT_OBJECT_POSITION_XYZ[0]
        )
        assert (bounding_box.min_y + bounding_box.max_y) / 2 == pytest.approx(
            CUT_OBJECT_POSITION_XYZ[1]
        )
