import asyncio
from typing import Callable, Coroutine, Optional, Any


def run_on_own_event_loop(coroutine: Coroutine) -> Any:
    """
    Run a coroutine to completion on an event loop that belongs to this call alone.

    An event loop can only be run by one thread at a time, so a loop shared by the whole
    process makes every thread but the first fail with "This event loop is already
    running". A loop per call lets threads that talk to different servers wait side by
    side, and behaves like a shared loop for a caller that only ever waits in one
    thread.

    :param coroutine: coroutine to run
    :return: whatever the coroutine returned
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


async def wait_until_not_none(
    variable_getter: Callable[[], Optional[Any]], check_interval: float = 0.1
) -> Any:
    while variable_getter() is None:
        await asyncio.sleep(check_interval)
    return variable_getter()


async def wait_until_none(
    variable_getter: Callable[[], Optional[Any]], check_interval: float = 0.1
) -> Any:
    while variable_getter() is not None:
        await asyncio.sleep(check_interval)
    return variable_getter()
