"""
Tests for the demo recipe source in ``recipes.py``.
"""

from __future__ import annotations

import pytest

from recipes import DemoRecipes


@pytest.fixture
def demo_recipes() -> DemoRecipes:
    return DemoRecipes()


class TestRecipeSwitching:
    def test_there_are_two_recipes_as_required(self, demo_recipes):
        assert len(demo_recipes.list_recipes()) == 2

    def test_both_boxes_have_the_same_size(self, demo_recipes):
        first, second = demo_recipes.list_recipes()
        assert first.box == second.box

    def test_recipes_are_retrievable_by_id(self, demo_recipes):
        first = demo_recipes.list_recipes()[0]
        assert demo_recipes.get_recipe(first.id) == first

    def test_unknown_recipe_id_yields_none(self, demo_recipes):
        assert demo_recipes.get_recipe("does-not-exist") is None
