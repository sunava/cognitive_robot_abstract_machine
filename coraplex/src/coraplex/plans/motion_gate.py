from __future__ import annotations

import threading
from contextlib import contextmanager

from typing_extensions import Iterator


class MotionAndModelChangeGate:
    """
    A readers/writer gate that keeps world model changes out of running motions.

    Several plans can be performed at the same time, one thread per robot, but they all
    share one semantic digital twin world. A
    :class:`~semantic_digital_twin.world_synchronizer.WorldSynchronizer` broadcasts every
    modification of that world to *all* giskard processes, and a giskard that receives a
    model change while it is executing a goal aborts that goal with a
    ``WorldModelModifiedDuringMotionError``. So a model change made by one robot's plan
    (typically attaching or detaching the object it just grasped) kills the motion of an
    unrelated robot that happens to be moving at that moment.

    The gate makes the two mutually exclusive without serializing the motions against
    each other:

    - a motion is a *reader*: any number of robots may have a goal in flight at once,
    - a model change is the *writer*: it waits until every running motion ended and no
      new motion starts until it (and the settling time giskard needs to apply it) is
      over.

    A plain mutex around the goals does not help: it prevents the robots from moving at
    the same time and the model change can still land in the middle of a goal.

    Writers are not starved: while one is waiting, arriving readers queue behind it
    instead of overtaking it, so a stream of motions cannot keep a model change out
    forever.

    The gate only matters for :attr:`~coraplex.datastructures.enums.ExecutionType.REAL`
    execution, where the motions run in remote giskard processes. The simulated path
    ticks the state chart in this process and never sends a world update to a remote
    giskard, so it does not take the gate at all (taking it there would only risk a
    deadlock in a single threaded simulation).

    The gate is not re-entrant: a thread that holds it as a motion must not ask for a
    model change (or the other way around) before it left. Plans never do, because a
    motion and a model change are separate executables that a plan runs one after the
    other.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._running_motions = 0
        """
        Number of motions currently holding the gate as readers.
        """
        self._model_change_running = False
        """
        Whether a model change currently holds the gate exclusively.
        """
        self._model_changes_waiting = 0
        """
        Number of model changes waiting, which blocks newly arriving motions.
        """

    @contextmanager
    def motion(self) -> Iterator[None]:
        """
        Hold the gate for a motion, next to any other motion that is running.

        Blocks while a model change is being applied or is waiting to be applied.
        """
        with self._condition:
            while self._model_change_running or self._model_changes_waiting:
                self._condition.wait()
            self._running_motions += 1
        try:
            yield
        finally:
            with self._condition:
                self._running_motions -= 1
                self._condition.notify_all()

    @contextmanager
    def model_change(self) -> Iterator[None]:
        """
        Hold the gate exclusively for a change of the shared world model.

        Blocks until every running motion ended, and keeps new motions out until the
        block is left.
        """
        with self._condition:
            self._model_changes_waiting += 1
            while self._model_change_running or self._running_motions:
                self._condition.wait()
            self._model_changes_waiting -= 1
            self._model_change_running = True
        try:
            yield
        finally:
            with self._condition:
                self._model_change_running = False
                self._condition.notify_all()


motion_and_model_change_gate = MotionAndModelChangeGate()
"""
The process wide gate between the motions and the model changes of all plans performed
in this process.

It has to be a single object for the whole process because the world model it protects
is shared by every plan performed here.
"""
