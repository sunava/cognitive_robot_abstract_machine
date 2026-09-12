"""
Tests for the gate between motions and world model changes (see
``coraplex/src/coraplex/plans/motion_gate.py``).

The gate lets several robots move at the same time while keeping every change of the
shared world model out of those motions, because the giskard processes that mirror the
world abort a running goal when the world is modified under them.
"""

import threading

import pytest

from coraplex.plans.motion_gate import MotionAndModelChangeGate

TIMEOUT = 5.0
"""
Generous upper bound for every wait in this module, so a broken gate fails instead of
hanging the test session.
"""


def _start(target, *args) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


def test_motions_run_concurrently():
    """
    Two motions must be able to hold the gate at the same time.
    """
    gate = MotionAndModelChangeGate()
    both_inside = threading.Barrier(2)
    overlapped = []

    def motion():
        with gate.motion():
            # Only returns if the other motion also got in, so the two really overlap.
            both_inside.wait(timeout=TIMEOUT)
            overlapped.append(True)

    threads = [_start(motion), _start(motion)]
    for thread in threads:
        thread.join(timeout=TIMEOUT)
        assert not thread.is_alive()
    assert overlapped == [True, True]


def test_model_change_waits_for_running_motions():
    """
    A model change must not start before the last running motion left the gate.
    """
    gate = MotionAndModelChangeGate()
    events = []
    events_lock = threading.Lock()
    motion_entered = threading.Event()
    model_change_started = threading.Event()

    def motion():
        with gate.motion():
            motion_entered.set()
            # Give the model change ample time to (wrongly) barge in.
            model_change_started.wait(timeout=0.2)
            with events_lock:
                events.append("motion left")

    def model_change():
        with gate.model_change():
            with events_lock:
                events.append("model change entered")
            model_change_started.set()

    motion_thread = _start(motion)
    assert motion_entered.wait(timeout=TIMEOUT)
    model_change_thread = _start(model_change)

    for thread in (motion_thread, model_change_thread):
        thread.join(timeout=TIMEOUT)
        assert not thread.is_alive()
    assert events == ["motion left", "model change entered"]


def test_waiting_model_change_holds_back_new_motions():
    """
    A motion arriving while a model change waits must queue behind it.

    Otherwise a stream of motions from several robots would starve the model change
    forever, and the model change would eventually land in the middle of one of them.
    """
    gate = MotionAndModelChangeGate()
    events = []
    events_lock = threading.Lock()
    first_motion_entered = threading.Event()
    release_first_motion = threading.Event()
    model_change_entered = threading.Event()

    def first_motion():
        with gate.motion():
            first_motion_entered.set()
            assert release_first_motion.wait(timeout=TIMEOUT)

    def model_change():
        with gate.model_change():
            with events_lock:
                events.append("model change entered")
            model_change_entered.set()

    def late_motion():
        with gate.motion():
            with events_lock:
                events.append("late motion entered")

    first_motion_thread = _start(first_motion)
    assert first_motion_entered.wait(timeout=TIMEOUT)

    model_change_thread = _start(model_change)
    # Let the model change reach the gate and register itself as waiting.
    model_change_entered.wait(timeout=0.2)
    assert not model_change_entered.is_set()

    late_motion_thread = _start(late_motion)
    # The late motion must not overtake the waiting model change, even though other
    # motions are still running.
    model_change_entered.wait(timeout=0.2)
    with events_lock:
        assert events == []

    release_first_motion.set()
    for thread in (first_motion_thread, model_change_thread, late_motion_thread):
        thread.join(timeout=TIMEOUT)
        assert not thread.is_alive()
    assert events == ["model change entered", "late motion entered"]


def test_model_changes_are_exclusive():
    """
    Two model changes must never be inside the gate at the same time.
    """
    gate = MotionAndModelChangeGate()
    inside = 0
    maximum_inside = 0
    counter_lock = threading.Lock()
    errors = []
    start = threading.Barrier(4)

    def model_change():
        nonlocal inside, maximum_inside
        try:
            start.wait(timeout=TIMEOUT)
            for _ in range(50):
                with gate.model_change():
                    with counter_lock:
                        inside += 1
                        maximum_inside = max(maximum_inside, inside)
                    with counter_lock:
                        inside -= 1
        except BaseException as e:  # noqa: BLE001 - reported to the main thread
            errors.append(e)

    threads = [_start(model_change) for _ in range(4)]
    for thread in threads:
        thread.join(timeout=TIMEOUT)
        assert not thread.is_alive()
    assert errors == []
    assert maximum_inside == 1


@pytest.mark.parametrize("holder", ["motion", "model_change"])
def test_gate_is_released_on_exception(holder: str):
    """
    An exception inside the block must not leave the gate held.
    """
    gate = MotionAndModelChangeGate()

    with pytest.raises(RuntimeError):
        with getattr(gate, holder)():
            raise RuntimeError("boom")

    # Both kinds of holders get in again immediately, so nothing stayed held.
    with gate.model_change():
        pass
    with gate.motion():
        pass
