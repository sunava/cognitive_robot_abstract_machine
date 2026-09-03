"""
The surfaces a plan can put an object on, shared by the ``/surfaces`` listing and a plan
that names one as its target.
"""

import pytest

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Drawer,
)

from cramera.live.placement_surface import (
    PLACEMENT_SURFACE_TYPES,
    UnknownPlacementSurface,
    placement_surface_type,
)


class TestNamingASurface:
    def test_a_surface_is_found_by_its_annotation_name(self):
        assert placement_surface_type(CounterTop.__name__) is CounterTop

    def test_a_container_counts_as_a_placement_surface(self):
        assert placement_surface_type(Drawer.__name__) is Drawer

    def test_an_annotation_nothing_can_be_placed_on_is_refused(self):
        with pytest.raises(UnknownPlacementSurface):
            placement_surface_type("Milk")

    def test_every_offered_surface_can_be_named(self):
        for surface in PLACEMENT_SURFACE_TYPES:
            assert placement_surface_type(surface.__name__) is surface
