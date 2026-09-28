"""
Physical grasp behavior of the native cable laboratory.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Thread

import mujoco
import numpy as np
import pytest

from cramera.cable import Cable, CableConfig
from cramera.cable_physics import CableObservation, CablePhysics
from physics_simulators.base_simulator import SimulatorState


# %% friction grasp
@pytest.fixture
def loose_cable():
    """
    Provide an unanchored cable and stop its simulator after each experiment.
    """
    physics = CablePhysics(CableConfig(anchor_to_parent=False))
    yield physics
    physics.stop()


def test_closing_and_lifting_holds_cable_through_contact(loose_cable):
    """
    Opposing pad friction transfers the carriage's upward motion to the cable.
    """
    loose_cable.command("pinch")
    loose_cable.step(800)
    initial_height = np.mean(loose_cable.segment_positions[:, 2])
    loose_cable.command("lift")
    loose_cable.step(1800)
    assert np.mean(loose_cable.segment_positions[:, 2]) > initial_height + 0.03
    assert loose_cable.model.neq == 0


def test_near_zero_friction_cannot_hold_cable():
    """
    A weak friction contact lets the cable remain on its support as pads rise.
    """
    physics = CablePhysics(
        CableConfig(anchor_to_parent=False, friction=(0.00001, 0.0, 0.0))
    )
    physics.command("pinch")
    physics.step(800)
    initial_height = np.mean(physics.segment_positions[:, 2])
    physics.command("lift")
    physics.step(1800)
    assert np.mean(physics.segment_positions[:, 2]) < initial_height + 0.01
    physics.stop()


def test_releasing_a_lifted_cable_allows_gravity_to_lower_it(loose_cable):
    """
    Opening the physical pads removes the cable's frictional support.
    """
    loose_cable.command("pinch")
    loose_cable.step(800)
    loose_cable.command("lift")
    loose_cable.step(1800)
    held_height = np.mean(loose_cable.segment_positions[:, 2])
    loose_cable.command("release")
    loose_cable.step(1200)
    assert np.mean(loose_cable.segment_positions[:, 2]) < held_height - 0.03


# %% simulator integration
def test_snapshot_reports_native_capsules_and_semantic_points(loose_cable):
    """
    The visual scene and semantic placeholders follow the same simulated cable.
    """
    loose_cable.step(20)
    snapshot = loose_cable.snapshot()
    assert snapshot["physical"] is True
    assert len(snapshot["segments"]) == loose_cable.config.segment_count
    for placeholder, position in zip(
        loose_cable.cable.placeholders, loose_cable.segment_positions
    ):
        np.testing.assert_allclose(
            loose_cable.world.compute_forward_kinematics_np(
                loose_cable.world.root, placeholder
            )[:3, 3],
            position,
        )


def test_start_and_stop_own_the_headless_physics_thread(loose_cable):
    """
    The session exposes the actual background simulator lifecycle.
    """
    loose_cable.start()
    assert loose_cable.running is True
    assert loose_cable.simulation.simulator.headless is True
    loose_cable.stop()
    assert loose_cable.running is False
    assert not loose_cable.simulation.simulator.simulation_thread.is_alive()


@pytest.mark.parametrize("position", [(0.0, 0.0), (np.nan, 0.0, 0.0)])
def test_invalid_move_target_is_rejected(loose_cable, position):
    """
    Invalid carriage coordinates cannot enter the actuator controls.
    """
    with pytest.raises(ValueError):
        loose_cable.command("move", position)


def test_background_steps_preserve_the_initial_cable_height(loose_cable):
    """
    Framework startup keeps new cable coordinates intact while time advances.
    """
    initial_height = np.mean(loose_cable.segment_positions[:, 2])
    loose_cable.start()
    deadline = time.monotonic() + 2.0
    while (
        loose_cable.snapshot()[CableObservation.SIMULATION_TIME] < 0.03
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    loose_cable.stop()
    assert loose_cable.snapshot()[CableObservation.SIMULATION_TIME] >= 0.03
    np.testing.assert_allclose(
        loose_cable.segment_positions[:, 2], initial_height, atol=0.001
    )


def test_native_failure_retains_visible_semantic_points_without_a_worker(monkeypatch):
    """
    Compilation failure leaves a readable placeholder-only laboratory.
    """

    def unavailable_native_spec(cable: Cable) -> mujoco.MjSpec:
        """
        Reproduce an unavailable native cable implementation.
        """
        raise RuntimeError("Native cable unavailable in this runtime")

    monkeypatch.setattr(Cable, "_native_spec", unavailable_native_spec)
    physics = CablePhysics()
    physics.start()
    snapshot = physics.snapshot()
    assert physics.running is False
    assert snapshot[CableObservation.PHYSICAL] is False
    assert len(snapshot[CableObservation.POINTS]) == physics.config.segment_count
    assert snapshot[CableObservation.WARNINGS] == [physics.cable.error]
    physics.stop()
    physics.stop()


def test_grasp_and_release_have_no_numerical_recovery(loose_cable):
    """
    Visible falling behavior never relies on MuJoCo resetting unstable dynamics.
    """
    loose_cable.command("pinch")
    loose_cable.step(800)
    loose_cable.command("lift")
    loose_cable.step(1800)
    loose_cable.command("release")
    loose_cable.step(1200)
    assert loose_cable.snapshot()[CableObservation.WARNINGS] == []


def test_numerical_recovery_warnings_are_visible(loose_cable):
    """
    A solver recovery cannot silently masquerade as ordinary cable dynamics.
    """
    warning = mujoco.mjtWarning.mjWARN_BADQACC
    loose_cable.simulation.simulator._mj_data.warning[warning].number = 1
    [message] = loose_cable.snapshot()[CableObservation.WARNINGS]
    assert warning.name in message


def test_a_dead_worker_is_not_reported_as_running(loose_cable):
    """
    An exception-terminated worker cannot retain a healthy lifecycle status.
    """
    simulator = loose_cable.simulation.simulator
    simulator._state = SimulatorState.RUNNING
    simulator.simulation_thread = Thread()
    assert loose_cable.running is False


def test_failed_apparatus_construction_releases_native_resources(monkeypatch):
    """
    An exception before factory return still disconnects the provisional scene.
    """
    constructed = []

    def unavailable_apparatus(physics: CablePhysics, spec: mujoco.MjSpec) -> None:
        """
        Stop construction after the framework has acquired scene resources.
        """
        constructed.append(physics)
        raise RuntimeError("Apparatus cannot be compiled")

    monkeypatch.setattr(CablePhysics, "_build_apparatus", unavailable_apparatus)
    with pytest.raises(RuntimeError):
        CablePhysics()
    [physics] = constructed
    assert physics.simulation.simulator.renderer.is_running() is False
    assert Path(physics._directory.name).exists() is False


def test_sustained_hold_and_release_do_not_reset_simulation_time(loose_cable):
    """
    A long interactive pinch remains stable when released after minutes of physics.
    """
    loose_cable.step(10000)
    loose_cable.command("pinch")
    loose_cable.step(30000)
    loose_cable.command("lift")
    loose_cable.step(80000)
    loose_cable.command("release")
    loose_cable.step(30000)
    snapshot = loose_cable.snapshot()
    assert snapshot[CableObservation.WARNINGS] == []
    assert snapshot[CableObservation.SIMULATION_TIME] == pytest.approx(
        150000 * loose_cable.parameters.timestep
    )
