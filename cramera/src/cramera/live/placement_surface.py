"""
The surfaces and containers a plan can put an object on or in.

The Plan Builder offers these in its "place on a surface" target mode, and a plan that
asks for such a target names one of them. Keeping the set in one place is what keeps the
offer and the request agreeing.
"""

from __future__ import annotations

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    CounterTop,
    Cupboard,
    Dishwasher,
    Drawer,
    Dresser,
    Floor,
    Fridge,
    ShelfLayer,
    Sofa,
    Table,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import Dict, Tuple, Type

PLACEMENT_SURFACE_TYPES: Tuple[Type[SemanticAnnotation], ...] = (
    CounterTop,
    Table,
    ShelfLayer,
    Floor,
    Sofa,
    Drawer,
    Fridge,
    Cabinet,
    Cupboard,
    Dresser,
    Dishwasher,
)
"""
Supporting surfaces to place *on* and case containers to place *in*, in the order the
Plan Builder offers them.

Both kinds answer ``sample_points_from_surface``, so resolving a place pose is the same
work for either.
"""

_BY_NAME: Dict[str, Type[SemanticAnnotation]] = {
    surface.__name__: surface for surface in PLACEMENT_SURFACE_TYPES
}


class UnknownPlacementSurface(Exception):
    """
    Raised when a plan names a surface nothing can be placed on.
    """


def placement_surface_type(name: str) -> Type[SemanticAnnotation]:
    """
    The annotation type a surface name stands for.

    :param name: The annotation's class name, as the viewer lists it.
    :raises UnknownPlacementSurface: If nothing can be placed on an annotation of that
        name.
    """
    if name not in _BY_NAME:
        raise UnknownPlacementSurface(
            "%r is not a surface a plan can place on; expected one of %s"
            % (name, sorted(_BY_NAME))
        )
    return _BY_NAME[name]
