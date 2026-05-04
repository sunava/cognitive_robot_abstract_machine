# source for base: https://medium.com/@idelossantosruiz/events-in-python-e2b3cb76ac2d
from poetry.console.commands import self

from demos.bachelor_thesis.classes.tasks import Task
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
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

            if reachable(world.get_semantic_annotation_by_name(obj.name), world):
                handler.reachable_objects.append(obj)

            if misplaced(obj, world):
                handler.misplaced_objects.append(obj)



# def update_reachable_objects(handler : EventDispatcher, data : list[Body], world : World):
#     robot = world.get_body_by_name("base_footprint")
#     print("I can reach following objects:")
#     for obj in data:
#         if (obj not in handler.reachable_objects) and (obj not in handler.known_furniture):
#             if reachable(obj.global_pose, robot.global_pose, world):
#                 handler.reachable_objects.append(obj)
#                 print(f" - {obj.name}")

def trigger_task(handler: EventDispatcher, data):
    # trigger set the table
    ts = tm()
    time = datetime.datetime.fromtimestamp(ts)#.strftime('%d-%m-%Y %H:%M:%S')
    if (9 < time.hour < 10 or 13 < time.hour < 14 or 19 < time.hour < 20) and not human_near():
        handler.activated_tasks.append(TableTask)


