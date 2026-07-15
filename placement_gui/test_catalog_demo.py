"""
Tests for the in-memory demo catalog in ``catalog.py``.
"""

from __future__ import annotations

import pytest

from catalog import DemoCatalog, UnknownShapeError
from placement import Pattern, Shape, Size


@pytest.fixture
def demo_catalog() -> DemoCatalog:
    return DemoCatalog()


class TestShapeCatalog:
    def test_sample_shapes_are_available(self, demo_catalog):
        shapes = demo_catalog.list_shapes()
        assert len(shapes) >= 2
        assert all(shape.size.width > 0 and shape.size.height > 0
                   for shape in shapes)

    def test_shapes_are_retrievable_by_id(self, demo_catalog):
        first = demo_catalog.list_shapes()[0]
        assert demo_catalog.get_shape(first.id) == first

    def test_unknown_shape_id_yields_none(self, demo_catalog):
        assert demo_catalog.get_shape("does-not-exist") is None


class TestPatternStorage:
    def test_sample_patterns_reference_catalog_shapes(self, demo_catalog):
        shape_ids = {shape.id for shape in demo_catalog.list_shapes()}
        for pattern in demo_catalog.list_patterns():
            assert pattern.shape.id in shape_ids

    def test_saved_pattern_is_listed_and_retrievable(self, demo_catalog):
        shape = demo_catalog.list_shapes()[0]
        pattern = Pattern(id="my-pattern", name="My pattern",
                          box=Size(500.0, 300.0), shape=shape, rows=2)
        demo_catalog.save_pattern(pattern)
        assert demo_catalog.get_pattern("my-pattern") == pattern
        assert pattern in demo_catalog.list_patterns()

    def test_saving_with_an_existing_id_updates_the_pattern(self, demo_catalog):
        existing = demo_catalog.list_patterns()[0]
        updated = Pattern(id=existing.id, name="Renamed", box=existing.box,
                          shape=existing.shape, gap=99.0)
        demo_catalog.save_pattern(updated)
        assert demo_catalog.get_pattern(existing.id).gap == 99.0
        assert len([p for p in demo_catalog.list_patterns()
                    if p.id == existing.id]) == 1

    def test_saving_with_an_unknown_shape_raises(self, demo_catalog):
        stranger = Shape(id="not-in-catalog", name="Stranger",
                         size=Size(10.0, 10.0))
        pattern = Pattern(id="bad", name="Bad", box=Size(500.0, 300.0),
                          shape=stranger)
        with pytest.raises(UnknownShapeError):
            demo_catalog.save_pattern(pattern)
