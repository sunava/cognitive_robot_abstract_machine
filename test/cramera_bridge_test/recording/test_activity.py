"""
Whether the simulation thread is moving the world.
"""

from __future__ import annotations

from cramera_bridge.live.activity import IDLE_SECONDS, SimulationActivity

from .clock import ManualClock


def test_the_simulation_is_idle_before_its_first_tick():
    assert SimulationActivity(clock=ManualClock()).is_busy() is False


def test_a_tick_keeps_the_simulation_busy_for_the_idle_time():
    clock = ManualClock()
    activity = SimulationActivity(clock=clock)
    activity.tick()
    clock.now = IDLE_SECONDS / 2
    busy = activity.is_busy()
    clock.now = IDLE_SECONDS
    assert busy is True
    assert activity.is_busy() is False
