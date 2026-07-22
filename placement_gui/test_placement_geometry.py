"""
Tests for the grid geometry in ``placement.py``.
"""

from __future__ import annotations

from placement import Recipe, Size, compute_pattern


def rail_recipe() -> Recipe:
    return Recipe(
        id="A", name="Part A", box=Size(600.0, 400.0),
        part=Size(480.0, 40.0), gap=12.0,
    )


class TestDefaultGrid:
    def test_fits_as_many_parts_as_the_box_allows(self):
        result = compute_pattern(rail_recipe())
        # (600 - 12) // (480 + 12) = 1 column, (400 - 12) // (40 + 12) = 7 rows
        assert result["columns"] == 1
        assert result["rows"] == 7
        assert result["count"] == 7
        assert len(result["placements"]) == 7

    def test_no_placements_when_the_part_is_larger_than_the_box(self):
        recipe = Recipe(id="X", name="Huge", box=Size(100.0, 100.0),
                        part=Size(200.0, 200.0))
        result = compute_pattern(recipe)
        assert result["count"] == 0
        assert result["offset_range"] == {"max_x": 0.0, "max_y": 0.0}

    def test_grid_is_centred_in_the_box(self):
        result = compute_pattern(rail_recipe())
        lowest = result["placements"][0]
        highest = result["placements"][-1]
        bottom_margin = lowest["y"]
        top_margin = result["box"]["height"] - (highest["y"] + highest["height"])
        assert bottom_margin > 0
        assert abs(bottom_margin - top_margin) < 1e-9


class TestOffset:
    def test_offset_is_applied_to_every_placement(self):
        centred = compute_pattern(rail_recipe())
        shifted = compute_pattern(rail_recipe(), offset_x=0.0, offset_y=5.0)
        for before, after in zip(centred["placements"],
                                 shifted["placements"]):
            assert after["y"] - before["y"] == 5.0

    def test_offset_is_clamped_to_the_box_walls(self):
        result = compute_pattern(rail_recipe(), offset_x=10_000.0,
                                 offset_y=-10_000.0)
        limits = result["offset_range"]
        assert result["offset"]["x"] == limits["max_x"]
        assert result["offset"]["y"] == -limits["max_y"]
        for placement in result["placements"]:
            assert placement["x"] >= 0
            assert placement["y"] >= 0
