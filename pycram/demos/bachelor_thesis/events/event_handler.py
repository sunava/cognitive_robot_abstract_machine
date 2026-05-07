# source for base: https://medium.com/@idelossantosruiz/events-in-python-e2b3cb76ac2d
from typing import Tuple, Any

from poetry.console.commands import self

from demos.bachelor_thesis.classes.tasks import SetTableTask, CleanTableTask, PutAwayObjectTask, LoadDishwasherTask, \
    UnloadDishwasherTask
from semantic_digital_twin.reasoning.queries import semantic_annotations_on_surfaces
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Cuttlery, Plate, Cup
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body, SemanticAnnotation
from semantic_digital_twin.reasoning.predicates import reachable
from demos.bachelor_thesis.actions.predicate_mock import reachable, misplaced, human_near
import datetime
from time import time as tm


class EventDispatcher:
    def __init__(self):
        self._listeners = []

        ##### replaces direct use of predicate functions for faster runtime ############################################
        self.perceived_objects = []
        """
        objects detected via robot perception
        """

        self.reachable_objects = []
        """
        objects detected via robot perception, that are reachable from one of the observe positions
        """

        self.misplaced_objects = []
        """
        objects that are at the wrong location, that are misplaced
        """

        self.known_furniture = []
        """
        furniture and walls existing in the world
        """

        ################################################################################################################

        self.activated_tasks = []
        """
        all tasks that were triggered
        """

        self.used_by_tasks = []
        """
        objects that are used by a task
        """

        ########## DEBUG ###############################################################################################
        self.locations = []
        self.trigger_nr = 0

        self.add_listener(update_perceived_objects)
        self.add_listener(trigger_task)
        #self.add_listener(update_reachable_objects)

    def add_listener(self, listener):
        """Register a new listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener):
        """Unregister an existing listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def trigger_event(self, event_data : list[Body], world : World):
        """Fire the event, passing event_data to every listener."""
        for listener in self._listeners:
            listener(self, event_data, world)


# Usage example
def update_perceived_objects(handler : EventDispatcher, data : list[Body], world : World):
    print("#"*110 + "\n" + "#"*110)
    print(f"NEW UPDATE No. {handler.trigger_nr}\n \n")
    handler.trigger_nr += 1
    # print("I saw following new objects:")
    for obj in data:
        if (obj not in handler.perceived_objects) and (obj not in handler.known_furniture):
            handler.perceived_objects.append(obj)
            # print(f" - {obj.name}")

            out_misplaced = misplaced(obj, world)

            if reachable(world.get_semantic_annotation_by_name(obj.name)):
                handler.reachable_objects.append(obj)

            if out_misplaced[0]:
                handler.misplaced_objects.append(obj)

            handler.locations.append([obj, out_misplaced[1]])
    print_perceived_objects(handler)



# def update_reachable_objects(handler : EventDispatcher, data : list[Body], world : World):
#     robot = world.get_body_by_name("base_footprint")
#     print("I can reach following objects:")
#     for obj in data:
#         if (obj not in handler.reachable_objects) and (obj not in handler.known_furniture):
#             if reachable(obj.global_pose, robot.global_pose, world):
#                 handler.reachable_objects.append(obj)
#                 print(f" - {obj.name}")

def trigger_task(handler: EventDispatcher, data : list[Body], world : World):
    # trigger set the table
    ts = tm()
    # time = datetime.datetime.fromtimestamp(ts)
    time = datetime.datetime(year=2026, month=5, day=6, hour=9, minute=10) # for testing
    if (time.hour == 9 or time.hour == 13 or time.hour == 19) and not human_near():
        table_name = "table"
        exists = False
        for task in handler.activated_tasks:
            if task.name == ("set_table_task_" + table_name):
                exists = True
                task.update_to_current_world_state(world, handler.perceived_objects)
        if not exists:
            handler.activated_tasks.append(SetTableTask("set_table_task_"+table_name, table=world.get_semantic_annotation_by_name("table"), world=world, perceived_objects=handler.perceived_objects))


    # trigger clean the table
    if (time.hour != 9 and time.hour != 13 and time.hour != 19) and not human_near():
        table_name = "table"
        exists = False
        for task in handler.activated_tasks:
            if task.name == ("clean_table_task_" + table_name):
                exists = True
                task.update_to_current_world_state(world, handler.perceived_objects)
        if not exists:
            handler.activated_tasks.append(CleanTableTask("clean_table_task_"+table_name, table=world.get_semantic_annotation_by_name("table"), world=world, perceived_objects=handler.perceived_objects))

    # trigger put away object
    for obj in handler.misplaced_objects:
        task_name = "put_away_object_task_" + obj.name.name
        exists = False
        for task in handler.activated_tasks:
            if task.name == task_name:
                exists = True
                task.update_to_current_world_state(world, handler.perceived_objects)
        if not exists:
            handler.activated_tasks.append(PutAwayObjectTask(task_name, required_objects=[obj.name], world=world, perceived_objects=handler.perceived_objects))

    # trigger load dishwasher task
    objs_on_counter = semantic_annotations_on_surfaces([world.get_semantic_annotation_by_name("counterTop")], world)
    load_dishwasher_task = None
    for obj in objs_on_counter:
        obj_body = world.get_body_by_name(obj.name)
        if isinstance(obj, (Cuttlery, Plate, Bowl, Cup)) and handler.perceived_objects.__contains__(obj_body) and not human_near():
            load_dishwasher_task = "load_dishwasher_task"

    if load_dishwasher_task is not None:
        exists = False
        for task in handler.activated_tasks:
            if task.name == load_dishwasher_task:
                exists = True
                task.update_to_current_world_state(world, handler.perceived_objects)
        if not exists:
            handler.activated_tasks.append(LoadDishwasherTask(load_dishwasher_task, handler.perceived_objects, world))

    # trigger unload dishwasher task
    objs_on_dishwasher_rack = semantic_annotations_on_surfaces([world.get_semantic_annotation_by_name("dishwasher_rack")], world)
    unload_dishwasher_task = None
    for obj in objs_on_dishwasher_rack:
        obj_body = world.get_body_by_name(obj.name)
        if isinstance(obj, (Cuttlery, Plate, Bowl, Cup)) and handler.perceived_objects.__contains__(obj_body) and not human_near():
            unload_dishwasher_task = "unload_dishwasher_task"

    if unload_dishwasher_task is not None:
        exists = False
        for task in handler.activated_tasks:
            if task.name == unload_dishwasher_task:
                exists = True
                task.update_to_current_world_state(world, handler.perceived_objects)
        if not exists:
            handler.activated_tasks.append(
                UnloadDishwasherTask(unload_dishwasher_task, handler.perceived_objects, world))

    print_tasks(handler)


def print_tasks(handler : EventDispatcher):
    print("\n \n")
    print("ACTIVE TASKS")
    print("-" * 110)

    header = f"{'Name':<60} | {'Feasibility':<15} | {'Score':<10} | {'Normalized Score':<18}"
    print(header)
    print("-" * 110)

    for task in handler.activated_tasks:
        name = task.name
        feasibility = task.calculate_feasibility()
        score = task.calculate_current_score()
        norm_score = task.calculate_current_score_normalized()

        line = f"{name:<60} | {feasibility:<15.3f} | {score:<10.3f} | {norm_score:<18.3f}"
        print(line)


def print_perceived_objects(handler : EventDispatcher):
    print("PERCEIVED OBJECTS")
    print("-"*110)
    print(
        f"{'Name':<35} | "
        f"{'Reachable':<10} | "
        f"{'Misplaced':<10} | "
        f"{'Belongs at Location':<30} ")
    print("-"*110)
    
    for obj in handler.perceived_objects:
        is_reachable = False
        is_misplaced = False
        location = None

        if obj in handler.reachable_objects:
            is_reachable = True
        if obj in handler.misplaced_objects:
            is_misplaced = True
        for loc in handler.locations:
            if loc[0]==obj:
                location = loc[1].name.name

        print(
            f"{obj.name.name:<35} | "
            f"{'YES' if is_reachable else 'NO':<10} | "
            f"{'YES' if is_misplaced else 'NO':<10} | "
            f"{str(location):<30} ")