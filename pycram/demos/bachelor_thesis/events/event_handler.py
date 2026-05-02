# source for base: https://medium.com/@idelossantosruiz/events-in-python-e2b3cb76ac2d
from poetry.console.commands import self

from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.reasoning.predicates import reachable
from demos.bachelor_thesis.actions.predicate_mock import reachable, misplaced


class EventDispatcher:
    def __init__(self):
        self._listeners = []

        self.perceived_objects = []
        """
        objects detected via robot perception
        """

        self.reachable_objects = []
        """
        objects detected via robot perception, that are reachable from one of the observe positions
        """

        self.known_furniture = []
        """
        furniture and walls existing in the world
        """

        self.activated_tasks = []
        """
        all tasks that were triggered
        """

        self.add_listener(update_perceived_objects)
        self.add_listener(misplaced_debug_test)
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

def misplaced_debug_test(handler : EventDispatcher, data: list[Body], world : World):
    for obj in data:
        print(misplaced(obj, world))


# def update_reachable_objects(handler : EventDispatcher, data : list[Body], world : World):
#     robot = world.get_body_by_name("base_footprint")
#     print("I can reach following objects:")
#     for obj in data:
#         if (obj not in handler.reachable_objects) and (obj not in handler.known_furniture):
#             if reachable(obj.global_pose, robot.global_pose, world):
#                 handler.reachable_objects.append(obj)
#                 print(f" - {obj.name}")

def trigger_task(handler: EventDispatcher, data):
    print("Nothing to see here")

