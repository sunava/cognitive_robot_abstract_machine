"""
put knowledge queries here, like:
empty(supportingSurface) = true
def misplaced(object):
    get objects location
    if location = place where belongs -> return false
    else -> return true
"""
import os
from contextlib import contextmanager

from sqlalchemy.dialects.oracle.dictionary import all_objects

from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.reasoning.queries import semantic_annotations_on_surfaces
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Food, Fruit, Cuttlery, Plate, Cup, Bowl, \
    Bottle, CounterTop, Table, ShelfLayer, DrinkingContainer
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.reasoning.predicates import is_supported_by

from time import sleep


import math

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation, Body


def perf_print(message: str):
    pass


@contextmanager
def perf_step(label: str):
    yield


def semantic_annotations_on_surface_cached(
    furniture: HasSupportingSurface,
    world: World,
    surface_cache: dict | None = None,
):
    cache_key = str(furniture.name)
    if surface_cache is not None and cache_key in surface_cache:
        perf_print(f"CACHE HIT semantic_annotations_on_surfaces: {cache_key}")
        return surface_cache[cache_key]

    with perf_step(f"semantic_annotations_on_surfaces: {cache_key}"):
        annotations = semantic_annotations_on_surfaces([furniture], world)

    if surface_cache is not None:
        surface_cache[cache_key] = annotations
    return annotations


def is_supported_by_surface_cached(
    obj: SemanticAnnotation,
    surface: HasSupportingSurface,
    support_cache: dict | None = None,
):
    cache_key = (str(obj.name), str(surface.name))
    if support_cache is not None and cache_key in support_cache:
        perf_print(f"CACHE HIT is_supported_by: {cache_key[0]} on {cache_key[1]}")
        return support_cache[cache_key]

    with perf_step(f"is_supported_by: {obj.name} on {surface.name}"):
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
):
    """
    returns true if object is misplaced and false if it is already at the correct location
    """

    with perf_step(f"misplaced determine correct location: {obj.name}"):
        if isinstance(obj, Food):
            correct_location = correct_location_food
        elif isinstance(obj, Bottle):
            correct_location = correct_location_drinks
        elif isinstance(obj, (Cuttlery, Plate, Bowl, Cup)):
            correct_location = correct_location_tableware
        else:
            correct_location = correct_location_all_other_items


    if isinstance(correct_location, (CounterTop, Table, ShelfLayer)):
        with perf_step(f"misplaced direct support check: {obj.name} on {correct_location.name}"):
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
        TypeError("Correct location is not of type HasSupportingSurface")
        return True, correct_location


def is_empty(
    furniture: HasSupportingSurface,
    known_objects: list[SemanticAnnotation],
    world: World,
    surface_cache: dict | None = None,
):
    """
    returns false, if the furniture has an already perceived object on it
    returns true, if it is empty (only acknowledged objects that are already perceived)
    """
    with perf_step(f"is_empty query surface contents: {furniture.name}"):
        annotations_on_surface = semantic_annotations_on_surface_cached(
            furniture,
            world,
            surface_cache,
        )
    with perf_step(f"is_empty scan {len(annotations_on_surface)} annotations on {furniture.name}"):
        for ann in annotations_on_surface:
            if ann in known_objects:
                return False
        return True

def human_near():
    # only useful in real scenario, no human in simulation
    return False

    


def reachable(object : SemanticAnnotation):
    # debug, WIP for later
    return True

