import os
from contextlib import contextmanager
from enum import Enum

from docutils.parsers.rst.directives import percentage
from typing_extensions import List

from demos.bachelor_thesis.classes_and_methods.tasks import Task
from demos.bachelor_thesis.events.event_handler import EventDispatcher
from pycram.datastructures.dataclasses import Context
from pycram.plans.factories import execute_single
from pycram.plans.plan import Plan
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table, SideTable, Wardrobe, \
    Sofa, Cup, ShelfLayer
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


class Environment(Enum):
    SuturoApartmentLab = 1
    Pr2ApartmentLab = 2


#-- METHODS ------------------------------------------------------------------------------------------------------------

def sort_tasks(tasks: List[Task], duration: float) -> list[Task]:
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

def print_sorted_task_list(tasks: List[Task], duration: float) -> None:
    print(f"OPTIMAL TASK ORDER FOR {duration / 60} minutes")
    print("-" * 120)

    header = f"{'Name':<60} | {'Feasibility':<15} | {'Score':<10} | {'Normalized Score':<18} | {'Duration':<10}"
    print(header)
    print("-" * 120)

    total_score = 0
    for task in tasks:
        name = task.name
        feasibility = task.calculate_feasibility()
        score = task.reward * feasibility
        norm_score = score / task.duration
        duration = task.duration
        total_score += score

        line = f"{name:<60} | {feasibility:<15.3f} | {score:<10.3f} | {norm_score:<18.3f} | {duration/60:<10.3f}"
        print(line)

    print("-" * 120)
    print("TOTAL SCORE: ", total_score)




def sorted_inserting(sorted_list: List[Task], elem: Task) -> List[Task]:
    new_list = []
    found = False
    if not sorted_list:
        new_list.append(elem)
        return new_list

    for i in range (len(sorted_list)):
        if (sorted_list[i].calculate_current_score_normalized() > elem.calculate_current_score_normalized()) or found:
            new_list.append(sorted_list[i])
        else:
            new_list.append(elem)
            new_list.append(sorted_list[i])
            found = True

    if not found:
        new_list.append(elem)
    return new_list


def timed_parse_stl(label: str, filename: str) -> World:
    return STLParser(
        os.path.join(
            os.path.dirname(__file__), "../..", "..", "resources", "objects", filename
        )
    ).parse()


def timed_plan(label: str, action, context: Context) -> Plan | None:
    return execute_single(action, context).plan

def debug_task_list_for_demo(dispatcher: EventDispatcher) -> None:
    print("\n \n", "DEBUG")

    for task in dispatcher.activated_tasks:
        print("." * 110)
        print(task.name)
        print(task.required_objects)
        print(task.precondition())
        print(task.constraints())



def create_annotations_for_bodies_sage10k(world: World) -> None:
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


def body_name_contains_keyword(body: Body, keyword: str) -> bool:
    if keyword in body.name.name:
        return True

    if body.name.prefix is not None:
        if keyword in body.name.prefix:
            return True
    return False


def compare_robot_world_with_real(dispatcher: EventDispatcher, world: World) -> list[list[float | None]]:
    real_world_dispatcher = EventDispatcher()
    real_world_dispatcher.correct_location_tableware_clean = dispatcher.correct_location_tableware_clean
    real_world_dispatcher.correct_location_tableware_dirty = dispatcher.correct_location_tableware_dirty
    real_world_dispatcher.correct_location_food = dispatcher.correct_location_food
    real_world_dispatcher.correct_location_drinks = dispatcher.correct_location_drinks
    real_world_dispatcher.correct_location_all_other_items = dispatcher.correct_location_all_other_items

    real_world_dispatcher.dining_table = dispatcher.dining_table
    real_world_dispatcher.dishwasher_exists = dispatcher.dishwasher_exists

    real_world_dispatcher.known_furniture = dispatcher.known_furniture

    # dispatcher gets all semantically annotated objects in the world -> same case as if robot has found all objects
    real_world_dispatcher.trigger_event(world.bodies, world)

    result = _print_task_comparison_robot_real(dispatcher, real_world_dispatcher)
    return result

def _print_task_comparison_robot_real(handler_robot : EventDispatcher, handler_world : EventDispatcher) \
        -> list[list[float | None]]:
    print("\n \n")
    print("COMPARE ROBOT WORLD VS REAL WORLD")
    print("-" * 110)

    header = f"{'Name':<60} | {'Feasibility':<15} | {'Score':<10} | {'Normalized Score':<18}"
    print(header)
    print("-" * 110)

    eval_array = []

    for task in handler_world.activated_tasks:
        # find task in robot handler
        robot_task = None
        for elem in handler_robot.activated_tasks:
            if elem.name == task.name:
                robot_task = elem



        if robot_task is None:
            print(f"{task.name: <60} not detected by robot")
            eval_array.append([None, None, None])
        else:
            name = task.name
            feasibility = task.calculate_feasibility()
            score = task.reward * feasibility
            norm_score = score / task.duration

            name_robot = robot_task.name
            feasibility_robot = robot_task.calculate_feasibility()
            score_robot = robot_task.reward * feasibility_robot
            norm_score_robot = score_robot / robot_task.duration

            line_robot = f"ROBOT:\n{name_robot:<60} | {feasibility_robot:<15.3f} | {score_robot:<10.3f} | {norm_score_robot:<18.3f}"
            print(line_robot)
            print("."*110)

            line_world = f"WORLD:\n{name:<60} | {feasibility:<15.3f} | {score:<10.3f} | {norm_score:<18.3f}"
            print(line_world)
            print("."*110)

            if feasibility == 0:
                percentage_feasibility = 0
            else:
                percentage_feasibility = feasibility_robot/feasibility

            line = f"ROBOT FOUND in percent:\n{name:<60} | {percentage_feasibility*100 :<15.3f} | {score_robot/score*100:<10.3f}% | {norm_score_robot/norm_score*100:<18.3f}%"
            print(line)
            print(":"*110)

            # duration and reward are compared through score and normalized score, required objects and preconditions and constraints through feasibility
            eval_array.append([feasibility_robot/feasibility, score_robot/score, norm_score_robot/norm_score])


    return eval_array


def print_object_locations(dispatcher: EventDispatcher, world: World) -> None:
    print("#"*110)
    for body in world.bodies:
        if body not in dispatcher.known_furniture:
            print(f"{body.name} at location ({body.global_pose.x}, {body.global_pose.y}, {body.global_pose.z})")
    print("#"*110)

