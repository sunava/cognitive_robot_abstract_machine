"""
Physical support, manipulation resistance and release in the authored laboratory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cramera.laboratory_physics import LaboratoryPhysics, InvalidPhysicsTarget
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory


# %% real contact fixture
@pytest.fixture
def physics(laboratory_directory: Path) -> LaboratoryPhysics:
    """
    Run the generated laboratory collisions without visual assets or a display.
    """
    return LaboratoryPhysics(bundle_directory=laboratory_directory)


class TestPhysicalSupport:
    """
    Loose objects settle on authored support surfaces under gravity.
    """

    def test_tubes_and_stopper_settle_without_falling_through(
        self, physics: LaboratoryPhysics
    ) -> None:
        before = physics.snapshot()["objects"]
        physics.step(1500)
        after = physics.snapshot()
        for key, pose in before.items():
            np.testing.assert_allclose(after["objects"][key][:3], pose[:3], atol=0.001)
        assert after["contacts"]
        assert all(contact["force"] >= 0 for contact in after["contacts"])
        assert np.isfinite(after["energy"])


class TestCompliantManipulation:
    """
    Cursor targets exert bounded forces without teleporting objects.
    """

    def test_lateral_target_stops_at_rack_rim(self, physics: LaboratoryPhysics) -> None:
        key = LaboratoryBody.CLEAR_TUBE
        before = physics.snapshot()["objects"][key]
        target = [before[0] + 0.03, before[1], before[2]]
        physics.set_target(key, target)
        assert physics.snapshot()["objects"][key] == before
        physics.step(1500)
        after = physics.snapshot()
        assert after["objects"][key][0] < before[0] + 0.008
        assert np.linalg.norm(np.asarray(target) - after["objects"][key][:3]) > 0.02
        assert any(
            {contact["bodyA"], contact["bodyB"]} == {key, LaboratoryBody.RACK}
            and contact["force"] > 0
            for contact in after["contacts"]
        )
        assert after["maximumAppliedForce"] <= physics.parameters.maximum_force

    def test_open_slot_allows_withdrawal_and_reinsertion(
        self, physics: LaboratoryPhysics
    ) -> None:
        key = LaboratoryBody.CLEAR_TUBE
        start = physics.snapshot()["objects"][key][:3]
        target = [start[0], start[1], start[2] + 0.18]
        physics.set_target(key, target)
        physics.step(2000)
        np.testing.assert_allclose(
            physics.snapshot()["objects"][key][:3], target, atol=0.002
        )
        physics.set_target(key, start)
        physics.step(2000)
        physics.release()
        physics.step(1000)
        np.testing.assert_allclose(
            physics.snapshot()["objects"][key][:3], start, atol=0.001
        )

    def test_release_removes_support_and_glass_falls(
        self, physics: LaboratoryPhysics
    ) -> None:
        key = LaboratoryBody.CLEAR_TUBE
        start = physics.snapshot()["objects"][key][:3]
        physics.set_target(key, [start[0], start[1], start[2] + 0.20])
        physics.step(2000)
        raised = physics.snapshot()["objects"][key][2]
        physics.release()
        physics.step(100)
        snapshot = physics.snapshot()
        assert snapshot["target"] is None
        assert snapshot["objects"][key][2] < raised - 0.02

    def test_remote_target_force_is_bounded_and_state_stays_finite(
        self, physics: LaboratoryPhysics
    ) -> None:
        physics.set_target(LaboratoryBody.CLEAR_TUBE, [1.0, -0.3, 1.2])
        physics.step(1000)
        snapshot = physics.snapshot()
        assert 0 < snapshot["maximumAppliedForce"] <= physics.parameters.maximum_force
        assert np.isfinite(snapshot["energy"])
        assert np.isfinite(list(snapshot["objects"].values())).all()


class TestPhysicsSessionState:
    """
    Reset and malformed input cannot leak targets into another interaction.
    """

    def test_reset_restores_authored_poses_and_clears_target(
        self, physics: LaboratoryPhysics
    ) -> None:
        original = physics.snapshot()["objects"]
        physics.set_target(LaboratoryBody.CLEAR_TUBE, [0.16, -0.065, 1.1])
        physics.step(1000)
        physics.reset()
        snapshot = physics.snapshot()
        assert snapshot["objects"] == original
        assert snapshot["target"] is None
        assert snapshot["time"] == 0
        assert snapshot["maximumAppliedForce"] == 0

    @pytest.mark.parametrize("position", [[float("nan"), 0, 1], [0, 1], [True, 0, 1]])
    def test_invalid_position_is_rejected(
        self, physics: LaboratoryPhysics, position: list[float]
    ) -> None:
        with pytest.raises(InvalidPhysicsTarget):
            physics.set_target(LaboratoryBody.CLEAR_TUBE, position)

    def test_static_environment_cannot_be_dragged(
        self, physics: LaboratoryPhysics
    ) -> None:
        with pytest.raises(InvalidPhysicsTarget):
            physics.set_target(LaboratoryBody.RACK, [0, 0, 1])
