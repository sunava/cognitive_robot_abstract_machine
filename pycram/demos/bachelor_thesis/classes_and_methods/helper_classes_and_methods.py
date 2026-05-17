from enum import Enum
import warnings

import os
from contextlib import contextmanager
from enum import Enum
from typing import List, Any, Tuple

from demos.bachelor_thesis.classes_and_methods.tasks import Task, PutAwayObjectTask
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

def sort_tasks(tasks: List[Task], duration: float):
    """
    returns optimal order of tasks to execute if robot has duration amount of time
    """
    list_sort = []
    for task in tasks:
        list_sort = sorted_inserting(list_sort, task)

    list_return = []
    for task in list_sort:
        if duration > task.duration:
            list_return.append(task)
            duration = duration - task.duration

    return list_return

def print_sorted_task_list(tasks: List[Task], duration: float):
    print(f"OPTIMAL TASK ORDER FOR {duration / 60} minutes")
    print("-" * 120)

    header = f"{'Name':<60} | {'Feasibility':<15} | {'Score':<10} | {'Normalized Score':<18} | {'Duration':<10}"
    print(header)
    print("-" * 120)

    total_score = 0
    for task in tasks:
        name = task.name
        with perf_step(f"calculate feasibility: {name}"):
            feasibility = task.calculate_feasibility()
        score = task.reward * feasibility
        norm_score = score / task.duration
        duration = task.duration
        total_score += score

        line = f"{name:<60} | {feasibility:<15.3f} | {score:<10.3f} | {norm_score:<18.3f} | {duration/60:<10.3f}"
        print(line)

    print("-" * 120)
    print("TOTAL SCORE: ", total_score)




def sorted_inserting(list: List[Task], elem: Task) -> List[Task]:
    new_list = []
    found = False
    if not list:
        new_list.append(elem)
        return new_list

    for i in range (len(list)):
        if (list[i].calculate_current_score_normalized() > elem.calculate_current_score_normalized()) or found:
            new_list.append(list[i])
        else:
            new_list.append(elem)
            new_list.append(list[i])
            found = True

    if not found:
        new_list.append(elem)
    return new_list





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
