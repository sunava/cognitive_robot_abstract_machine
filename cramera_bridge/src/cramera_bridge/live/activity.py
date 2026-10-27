"""
Whether the simulation thread is moving the world right now.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from typing_extensions import Callable, Optional

IDLE_SECONDS = 0.5
"""
How long after its last motion tick the simulation counts as idle; motion ticks come at
least ten times a second while a plan moves the robot.
"""

# %% activity


@dataclass
class SimulationActivity:
    """
    Remembers the last motion tick, so requests from the viewer know whether the
    simulation thread will apply them or whether nothing is moving the world.
    """

    clock: Callable[[], float] = time.monotonic
    """
    The monotonic clock ticks are timed with.
    """

    idle_seconds: float = IDLE_SECONDS
    """
    How long after its last tick the simulation counts as idle.
    """

    last_tick: Optional[float] = field(default=None, init=False)
    """
    When the last motion tick happened, or None before the first.
    """

    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    """
    Guards :attr:`last_tick`.
    """

    def tick(self) -> None:
        """
        Note that the simulation thread just ticked.
        """
        with self.lock:
            self.last_tick = self.clock()

    def is_busy(self) -> bool:
        """
        :return: Whether a motion ticked within the last :attr:`idle_seconds`.
        """
        with self.lock:
            last_tick = self.last_tick
        return last_tick is not None and self.clock() - last_tick < self.idle_seconds
