# source for base: https://medium.com/@idelossantosruiz/events-in-python-e2b3cb76ac2d


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

    def add_listener(self, listener):
        """Register a new listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener):
        """Unregister an existing listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def trigger_event(self, event_data=None):
        """Fire the event, passing event_data to every listener."""
        for listener in self._listeners:
            listener(event_data)

# Usage example
def my_listener(data):
    print("Received event with data:", data)

dispatcher = EventDispatcher()
dispatcher.add_listener(my_listener)
dispatcher.trigger_event("Hello, World!")