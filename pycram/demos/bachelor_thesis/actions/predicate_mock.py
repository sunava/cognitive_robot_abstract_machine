from contextlib import contextmanager

from typing_extensions import Tuple

from semantic_digital_twin.reasoning.queries import semantic_annotations_on_surfaces
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Food, Cuttlery, Plate, Cup, Bowl, \
    Bottle, CounterTop, Table, ShelfLayer
from semantic_digital_twin.world import World
from semantic_digital_twin.reasoning.predicates import is_supported_by

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


def semantic_annotations_on_surface_cached(
    furniture: HasSupportingSurface,
    world: World,
    surface_cache: dict | None = None,
) -> list[SemanticAnnotation]:
    cache_key = str(furniture.name)
    if surface_cache is not None and cache_key in surface_cache:
        return surface_cache[cache_key]

    annotations = semantic_annotations_on_surfaces([furniture], world)

    if surface_cache is not None:
        surface_cache[cache_key] = annotations
    return annotations


def is_supported_by_surface_cached(
    obj: SemanticAnnotation,
    surface: HasSupportingSurface,
    support_cache: dict | None = None,
) -> bool:
    cache_key = (str(obj.name), str(surface.name))
    if support_cache is not None and cache_key in support_cache:
        return support_cache[cache_key]

    result = is_supported_by(obj.root, surface.root)

    if support_cache is not None:
        support_cache[cache_key] = result
    return result


def misplaced(
    obj: SemanticAnnotation,
    world: World,
    correct_location_tableware: SemanticAnnotation,
    correct_location_food: SemanticAnnotation,
    correct_location_drinks: SemanticAnnotation,
    correct_location_all_other_items: SemanticAnnotation,
    surface_cache: dict | None = None,
    support_cache: dict | None = None,
)-> Tuple[bool, SemanticAnnotation]:
    """
    returns true if object is misplaced and false if it is already at the correct location
    """

    if isinstance(obj, Food):
        correct_location = correct_location_food
    elif isinstance(obj, Bottle):
        correct_location = correct_location_drinks
    elif isinstance(obj, (Cuttlery, Plate, Bowl, Cup)):
        correct_location = correct_location_tableware
    else:
        correct_location = correct_location_all_other_items


    if isinstance(correct_location, (CounterTop, Table, ShelfLayer)):
        is_on_correct_location = is_supported_by_surface_cached(
            obj,
            correct_location,
            support_cache,
        )
        if is_on_correct_location:
            return False, correct_location
        else:
            return True, correct_location
    else:
        raise TypeError("Correct location is not of type HasSupportingSurface")


def is_empty(
    furniture: HasSupportingSurface,
    known_objects: list[SemanticAnnotation],
    world: World,
    surface_cache: dict | None = None,
) -> bool:
    """
    returns false, if the furniture has an already perceived object on it
    returns true, if it is empty (only acknowledged objects that are already perceived)
    """
    annotations_on_surface = semantic_annotations_on_surface_cached(
        furniture,
        world,
        surface_cache,
    )
    for ann in annotations_on_surface:
        if ann in known_objects:
            return False
    return True

def human_near() -> bool:
    # only useful in real scenario, no human in simulation
    return False

    


def reachable(object : SemanticAnnotation) -> bool:
    # debug, WIP for later
    return True

