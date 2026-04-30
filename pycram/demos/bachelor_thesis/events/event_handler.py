# source for base: https://medium.com/@idelossantosruiz/events-in-python-e2b3cb76ac2d
from poetry.console.commands import self

from semantic_digital_twin.world_description.world_entity import Body


class EventDispatcher:
    def __init__(self):
        self._listeners = []

        self.perceived_objects = []
        """
        objects detected via robot perception
        """

        self.known_furniture = []
        """
        furniture and walls existing in the world
        """

        self.activated_tasks = []
        """
        all tasks that were triggered
        """

    def add_listener(self, listener):
        """Register a new listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener):
        """Unregister an existing listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def trigger_event(self, event_data : list[Body]):
        """Fire the event, passing event_data to every listener."""
        for listener in self._listeners:
            listener(self, event_data)


# Usage example
def update_perceived_objects(handler : EventDispatcher, data : list[Body]):
    print("I saw following new objects:")
    for obj in data:
        if (obj not in handler.perceived_objects) and (obj not in handler.known_furniture):
            handler.perceived_objects.append(obj)
            print(f" - {obj.name}")

def trigger_task(handler: EventDispatcher, data):
    print("Nothing to see here")

