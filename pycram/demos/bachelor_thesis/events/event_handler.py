# source for base: https://medium.com/@idelossantosruiz/events-in-python-e2b3cb76ac2d
from contextlib import contextmanager

from rdflib.plugins.sparql.parser import PrefixedName

from demos.bachelor_thesis.classes_and_methods.tasks import SetTableTask, CleanTableTask, PutAwayObjectTask, \
    LoadDishwasherTask, UnloadDishwasherTask
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Cuttlery, Plate, Cup, Tableware
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body, SemanticAnnotation
from demos.bachelor_thesis.actions.predicate_mock import (
    reachable,
    misplaced,
    human_near,
    semantic_annotations_on_surface_cached,
    is_supported_by_surface_cached, is_empty,
)
import datetime
from time import sleep, time as tm


class EventDispatcher:
    def __init__(self):
        self._listeners = []

        #----- correct locations for items -----------------------------------------------------------------------------
        self.correct_location_tableware_dirty = None
        self.correct_location_tableware_clean = None
        self.correct_location_food = None
        self.correct_location_drinks = None
        self.correct_location_all_other_items = None

        self.dining_table = None
        self.dishwasher_exists = True

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

        # self.used_by_tasks = []
        # """
        # objects that are used by a task
        # """ removed, only useful if tasks are chosen and executed

        ########## DEBUG ###############################################################################################
        self.locations = []
        self.trigger_nr = 0
        self.surface_annotation_cache = {}
        self.support_relation_cache = {}
        self.perceived_objects_changed = False

        self.add_listener(update_perceived_objects)
        self.add_listener(trigger_task)
        #self.add_listener(update_reachable_objects)

    def add_listener(self, listener) -> None:
        """Register a new listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener) -> None:
        """Unregister an existing listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def trigger_event(self, event_data : list[Body], world : World) -> None:
        """Fire the event, passing event_data to every listener."""
        sem_annotations = []
        for data in event_data:
            if not data in self.known_furniture:
                try:
                    annotation = world.get_semantic_annotation_by_name(data.name)
                    sem_annotations.append(annotation)
                except WorldEntityNotFoundError:
                    print(f"Couldn't find Semantic Annotation for {data.name}")

        for listener in self._listeners:
            listener(self, sem_annotations, world)


# Usage example
def update_perceived_objects(handler : EventDispatcher, data : list[SemanticAnnotation], world : World) -> None:
    is_none = []
    if (handler.correct_location_drinks is None) or not isinstance(handler.correct_location_drinks, HasSupportingSurface):
        is_none.append("handler.correct_location_drinks")
    if (handler.correct_location_food is None)  or not isinstance(handler.correct_location_food, HasSupportingSurface):
        is_none.append("handler.correct_location_food")
    if (handler.correct_location_tableware_dirty is None)  or not isinstance(handler.correct_location_tableware_dirty, HasSupportingSurface):
        is_none.append("handler.correct_location_tableware_dirty")
    if (handler.correct_location_tableware_clean is None) or not isinstance(handler.correct_location_tableware_clean, HasSupportingSurface):
        is_none.append("handler.correct_location_tableware_clean")
    if (handler.correct_location_all_other_items is None)  or not isinstance(handler.correct_location_all_other_items, HasSupportingSurface):
        is_none.append("handler.correct_location_all_other_items")
    if (handler.dining_table is None) or not isinstance(handler.dining_table, HasSupportingSurface):
        is_none.append("handler.dining_table")

    if is_none:
        raise Exception(f"{is_none} is not set or is not a supporting surface.")
    print(world.root)
    print("#"*110 + "\n" + "#"*110)
    print(f"NEW UPDATE No. {handler.trigger_nr}\n \n")
    handler.trigger_nr += 1
    new_objects = 0

    for obj in data:
        if obj not in handler.perceived_objects: # furniture already filtered out
            new_objects += 1

            handler.perceived_objects.append(obj)

            if isinstance(obj, Tableware) and handler.dishwasher_exists and is_supported_by_surface_cached(obj, world.get_semantic_annotation_by_name("dishwasher_rack"), handler.support_relation_cache):
                obj.clean = True
                print(f"clean: {obj.name}")
            elif isinstance(obj, Tableware):
                obj.clean = False
                print(f"dirty: {obj.name}")


            out_misplaced = misplaced(
                obj,
                world,
                handler.correct_location_tableware_dirty,
                handler.correct_location_tableware_clean,
                handler.correct_location_food,
                handler.correct_location_drinks,
                handler.correct_location_all_other_items,
                handler.surface_annotation_cache,
                handler.support_relation_cache,
            )

            is_reachable = reachable(obj)

            if is_reachable:
                handler.reachable_objects.append(obj)

            if out_misplaced[0]:
                handler.misplaced_objects.append(obj)

            handler.locations.append([obj, out_misplaced[1]])
    handler.perceived_objects_changed = new_objects > 0
    print_perceived_objects(handler)


def trigger_task(handler: EventDispatcher, data : list[SemanticAnnotation], world : World) -> None:
    _trigger_set_table(handler, world)
    _trigger_clean_table(handler, data, world)
    _trigger_put_away_object(handler, world)
    _trigger_load_dishwasher(handler, world)
    _trigger_unload_dishwasher(handler, world)

    print_tasks(handler)

def _trigger_set_table(handler: EventDispatcher, world: World) -> None:
    # trigger set the table
    ts = tm()
    # time = datetime.datetime.fromtimestamp(ts)
    time = datetime.datetime(year=2026, month=5, day=6, hour=9, minute=10)  # for testing set the table
    # time = datetime.datetime(year=2026, month=5, day=6, hour=11, minute=10)  # for testing clean the table

    if (time.hour == 9 or time.hour == 13 or time.hour == 19) and not human_near():
        table_name = handler.dining_table.name.name
        exists = False
        for task in handler.activated_tasks:
            if task.name == ("set_table_task_" + table_name):
                exists = True
                if handler.perceived_objects_changed:
                    task.update_to_current_world_state(
                        world,
                        handler.perceived_objects,
                        handler.surface_annotation_cache,
                    )
                else:
                    # skip update task, perceived objects unchanged
                    pass
        if not exists:
            handler.activated_tasks.append(
                SetTableTask(
                    "set_table_task_" + table_name,
                    handler.dining_table,
                    world=world,
                    perceived_objects=handler.perceived_objects,
                    surface_cache=handler.surface_annotation_cache,
                )
            )

def _trigger_clean_table(handler: EventDispatcher, data: list[SemanticAnnotation], world: World) -> None:
    # trigger clean the table
    ts = tm()
    # time = datetime.datetime.fromtimestamp(ts)
    # time = datetime.datetime(year=2026, month=5, day=6, hour=9, minute=10)  # for testing set the table
    time = datetime.datetime(year=2026, month=5, day=6, hour=11, minute=10)  # for testing clean the table
    if (time.hour != 9 and time.hour != 13 and time.hour != 19) and not human_near() \
            and not is_empty(handler.dining_table, data, world, handler.surface_annotation_cache):
        table_name = handler.dining_table.name.name
        exists = False
        for task in handler.activated_tasks:
            if task.name == ("clean_table_task_" + table_name):
                exists = True
                if handler.perceived_objects_changed:
                    task.update_to_current_world_state(world, handler.perceived_objects)
                else:
                    # skip update task {task.name}: perceived objects unchanged
                    pass

        if not exists:
            handler.activated_tasks.append(
                CleanTableTask("clean_table_task_" + table_name, handler.dining_table, world=world,
                               perceived_objects=handler.perceived_objects))

def _trigger_put_away_object(handler: EventDispatcher, world: World) -> None:
    # trigger put away object
    for obj in handler.misplaced_objects:
        task_name = "put_away_object_task_" + obj.name.name
        exists = False
        for task in handler.activated_tasks:
            if task.name == task_name:
                exists = True
                if handler.perceived_objects_changed:
                    task.update_to_current_world_state(world, handler.perceived_objects)
                else:
                    # skip update task, perceived objects unchanged
                    pass
        if not exists:
            handler.activated_tasks.append(PutAwayObjectTask(task_name, required_objects=[obj], world=world,
                                                             perceived_objects=handler.perceived_objects))

def _perceive_dishware(handler: EventDispatcher) -> list[SemanticAnnotation]:
    perceived_dishware = []
    for obj in handler.perceived_objects:
        if isinstance(obj, (Cuttlery, Plate, Bowl, Cup)):
            perceived_dishware.append(obj)

    return perceived_dishware

def _trigger_load_dishwasher(handler: EventDispatcher, world: World) -> None:
    perceived_dishware = _perceive_dishware(handler)

    # trigger load dishwasher task
    load_dishwasher_task = None
    load_dishwasher_objects = []
    if len(perceived_dishware) > 0:
        for obj in perceived_dishware:
            if obj not in handler.misplaced_objects and not human_near():
                load_dishwasher_task = "load_dishwasher_task"
                load_dishwasher_objects.append(obj)
    else:
        # skip counterTop query: no perceived dishware/cutlery
        pass

    if load_dishwasher_task is not None and handler.dishwasher_exists:
        exists = False
        for task in handler.activated_tasks:
            if task.name == load_dishwasher_task:
                exists = True
                if handler.perceived_objects_changed:
                    task.update_to_current_world_state(
                        world,
                        handler.perceived_objects,
                        surface_cache=handler.surface_annotation_cache,
                        support_cache=handler.support_relation_cache,
                        required_objects=load_dishwasher_objects,
                    )
                else:
                    # skip update task {task.name}: perceived objects unchanged
                    pass
        if not exists:
            handler.activated_tasks.append(
                LoadDishwasherTask(
                    load_dishwasher_task,
                    handler.perceived_objects,
                    handler.correct_location_tableware_dirty,
                    world,
                    surface_cache=handler.surface_annotation_cache,
                    support_cache=handler.support_relation_cache,
                    required_objects=load_dishwasher_objects,
                )
            )

def _trigger_unload_dishwasher(handler: EventDispatcher, world: World) -> None:
    perceived_dishware = _perceive_dishware(handler)
    # trigger unload dishwasher task
    unload_dishwasher_task = None
    unload_dishwasher_objects = []
    if len(perceived_dishware) > 0:
        dishwasher_rack = world.get_semantic_annotation_by_name("dishwasher_rack")
        for annotation in perceived_dishware:
            if is_supported_by_surface_cached(
                    annotation,
                    dishwasher_rack,
                    handler.support_relation_cache,
            ) and not human_near():
                unload_dishwasher_task = "unload_dishwasher_task"
                unload_dishwasher_objects.append(annotation)
    else:
        # skip dishwasher_rack query: no perceived dishware/cutlery
        pass

    if unload_dishwasher_task is not None and handler.dishwasher_exists:
        exists = False
        for task in handler.activated_tasks:
            if task.name == unload_dishwasher_task:
                exists = True
                if handler.perceived_objects_changed:
                    task.update_to_current_world_state(
                        world,
                        handler.perceived_objects,
                        surface_cache=handler.surface_annotation_cache,
                        required_objects=unload_dishwasher_objects,
                    )
                else:
                    # skip update task {task.name}: perceived objects unchanged
                    pass
        if not exists:
            handler.activated_tasks.append(
                UnloadDishwasherTask(
                    unload_dishwasher_task,
                    handler.perceived_objects,
                    world,
                    surface_cache=handler.surface_annotation_cache,
                    required_objects=unload_dishwasher_objects,
                )
            )

def print_tasks(handler : EventDispatcher) -> None:
    print("\n \n")
    print("ACTIVE TASKS")
    print("-" * 110)

    header = f"{'Name':<60} | {'Feasibility':<15} | {'Score':<10} | {'Normalized Score':<18}"
    print(header)
    print("-" * 110)

    for task in handler.activated_tasks:
        name = task.name
        feasibility = task.calculate_feasibility()
        score = task.reward * feasibility
        if task.duration == 0:
            norm_score = 0
        else:
            norm_score = score / task.duration

        line = f"{name:<60} | {feasibility:<15.3f} | {score:<10.3f} | {norm_score:<18.3f}"
        print(line)


def print_perceived_objects(handler : EventDispatcher) -> None:
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
