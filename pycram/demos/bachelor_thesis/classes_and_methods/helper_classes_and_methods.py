from enum import Enum
import warnings

import os
from contextlib import contextmanager
from enum import Enum

from demos.bachelor_thesis.events.event_handler import EventDispatcher
from pycram.datastructures.dataclasses import Context
from pycram.plans.factories import execute_single
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table, SideTable, Wardrobe, Chair, Armchair, \
    Sofa, Cup, ShelfLayer
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body, SemanticAnnotation


class Environment(Enum):
    SuturoApartmentLab = 1
    Pr2ApartmentLab = 2


#-- METHODS ------------------------------------------------------------------------------------------------------------

def perf_print(message: str):
    pass


@contextmanager
def perf_step(label: str):
    yield


def timed_parse_stl(label: str, filename: str):
    with perf_step(f"parse STL: {label}"):
        return STLParser(
            os.path.join(
                os.path.dirname(__file__), "../..", "..", "resources", "objects", filename
            )
        ).parse()


def timed_plan(label: str, action, context: Context):
    with perf_step(f"build plan: {label}"):
        return execute_single(action, context).plan

def debug_task_list_for_demo(dispatcher: EventDispatcher):
    print("\n \n", "DEBUG")

    for task in dispatcher.activated_tasks:
        print("." * 110)
        print(task.name)
        print(task.required_objects)
        print(task.precondition())
        print(task.constraints())



def create_annotations_for_bodies_sage10k(world: World):
    bodies = world.bodies
    with (world.modify_world()):
        for bod in bodies:
            if body_name_contains_keyword(bod, "table"):
                world.add_semantic_annotation(Table(root=bod, name=bod.name))
            elif body_name_contains_keyword(bod, "sideboard"):
                world.add_semantic_annotation(SideTable(root=bod, name=bod.name))
            elif body_name_contains_keyword(bod, "coatstand"):
                world.add_semantic_annotation(Wardrobe(root=bod, name=bod.name))
            elif body_name_contains_keyword(bod, "chair") | body_name_contains_keyword(bod, "seat") | body_name_contains_keyword(bod, "pouf"):
                world.add_semantic_annotation(Sofa(root=bod, name=bod.name))
            elif body_name_contains_keyword(bod, "mug"):
                world.add_semantic_annotation(Cup(root=bod, name=bod.name))
            elif body_name_contains_keyword(bod, "bookshelf"):
                world.add_semantic_annotation(ShelfLayer(root=bod, name=bod.name))

            try:
                sem_anno = world.get_semantic_annotation_by_name(bod.name)
                if isinstance(sem_anno, HasSupportingSurface):
                    sem_anno.calculate_supporting_surface(upward_threshold=2)
            except WorldEntityNotFoundError:
                pass


def body_name_contains_keyword(body: Body, keyword: str):
    if keyword in body.name.name:
        return True

    if body.name.prefix is not None:
        if keyword in body.name.prefix:
            return True
    return False
