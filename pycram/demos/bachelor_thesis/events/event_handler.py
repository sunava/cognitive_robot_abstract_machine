# source for base: https://medium.com/@idelossantosruiz/events-in-python-e2b3cb76ac2d
from poetry.console.commands import self

from demos.bachelor_thesis.classes.tasks import SetTableTask, CleanTableTask, PutAwayObjectTask
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
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
    print("I saw following new objects:")
    for obj in data:
        if (obj not in handler.perceived_objects) and (obj not in handler.known_furniture):
            handler.perceived_objects.append(obj)
            print(f" - {obj.name}")

            if reachable(world.get_semantic_annotation_by_name(obj.name)):
                handler.reachable_objects.append(obj)

            if misplaced(obj, world)[0]:
                handler.misplaced_objects.append(obj)



# def update_reachable_objects(handler : EventDispatcher, data : list[Body], world : World):
#     robot = world.get_body_by_name("base_footprint")
#     print("I can reach following objects:")
#     for obj in data:
#         if (obj not in handler.reachable_objects) and (obj not in handler.known_furniture):
#             if reachable(obj.global_pose, robot.global_pose, world):
#                 handler.reachable_objects.append(obj)
#                 print(f" - {obj.name}")

def trigger_task(handler: EventDispatcher, data : list[Body], world : World):
    print("task triggered:")
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

    for obj in handler.misplaced_objects:
        task_name = "put_away_object_task_" + obj.name.name
        exists = False
        for task in handler.activated_tasks:
            if task.name == task_name:
                exists = True
                task.update_to_current_world_state(world, handler.perceived_objects)
        if not exists:
            handler.activated_tasks.append(PutAwayObjectTask(task_name, required_objects=[obj.name], world=world, perceived_objects=handler.perceived_objects))

    print_tasks(handler)


def print_tasks(handler : EventDispatcher):
    print("ACTIVE TASKS")
    print("----------------------------------------------------------------------------------------------------------")
    print(" name                         | feasibility     | Score     | Normalized Score")
    print("----------------------------------------------------------------------------------------------------------")
    for task in handler.activated_tasks:
        print(f"{task.name}         {task.calculate_feasibility()}      {task.calculate_current_score()}         {task.calculate_current_score_normalized()}")
