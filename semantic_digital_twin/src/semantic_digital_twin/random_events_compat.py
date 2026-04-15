from random_events.interval import SimpleInterval


def make_simple_interval(lower=0, upper=0, left=None, right=None):
    """
    Support both SimpleInterval.from_data(...) and direct-constructor APIs.
    """
    factory = getattr(SimpleInterval, "from_data", None)
    if callable(factory):
        return factory(lower, upper, left, right)
    return SimpleInterval(lower, upper, left, right)
