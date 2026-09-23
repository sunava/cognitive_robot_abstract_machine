"""
Physical rotor motion, measured seating and centrifuge interlocks.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from cramera.laboratory_centrifuge import CentrifugeField, InvalidCentrifugeOperation
from cramera.laboratory_physics import LaboratoryPhysics
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics import physics


# %% measured holder fixtures
def seat_tube(physics: LaboratoryPhysics, key: str, slot: str) -> None:
    """
    Initialize a glass above a pocket and let gravity establish contact.
    """
    position = physics.centrifuge.slot_position(slot)
    body = physics.objects[key].body_id
    joint = physics.model.body_jntadr[body]
    coordinate = physics.model.jnt_qposadr[joint]
    velocity = physics.model.jnt_dofadr[joint]
    physics.data.qpos[coordinate : coordinate + 3] = position + [0, 0, 0.002]
    physics.data.qpos[coordinate + 3 : coordinate + 7] = [1, 0, 0, 0]
    physics.data.qvel[velocity : velocity + 6] = 0
    mujoco.mj_forward(physics.model, physics.data)
    physics.step(1800)


def load_pair(physics: LaboratoryPhysics, second_volume: float = 5.0) -> None:
    """
    Seat and explicitly secure an opposing pair with known liquid masses.
    """
    for key, slot, volume in (
        (LaboratoryBody.CLEAR_TUBE, "C1", 5.0),
        (LaboratoryBody.AMBER_TUBE, "C3", second_volume),
    ):
        physics.fill_liquid(key, volume)
        seat_tube(physics, key, slot)
        physics.centrifuge_command("secure", slot=slot, key=key)


class TestCentrifugeInterlocks:
    """
    The drive is permitted only for a closed and balanced fixture.
    """

    def test_authored_instrument_starts_open_and_stopped(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        A newly reset instrument is ready for loading its four holders.
        """
        state = physics.snapshot()[CentrifugeField.CENTRIFUGE]
        assert state[CentrifugeField.AVAILABLE] is True
        assert state[CentrifugeField.LID_OPEN] is True
        assert state[CentrifugeField.RPM] == 0
        assert len(state[CentrifugeField.SLOTS]) == 4

    def test_unseated_glass_cannot_be_secured(self, physics: LaboratoryPhysics) -> None:
        """
        Clamp engagement never transports a glass from elsewhere in the room.
        """
        before = physics.data.qpos.copy()
        with pytest.raises(InvalidCentrifugeOperation):
            physics.centrifuge_command(
                "secure", slot="C1", key=LaboratoryBody.CLEAR_TUBE
            )
        np.testing.assert_array_equal(physics.data.qpos, before)

    def test_empty_rotor_cannot_start(self, physics: LaboratoryPhysics) -> None:
        """
        A cycle requires samples in opposing holders.
        """
        physics.centrifuge_command("close")
        physics.step(5000)
        with pytest.raises(InvalidCentrifugeOperation):
            physics.centrifuge_command("start", rpm=120)

    def test_mass_imbalance_prevents_start(self, physics: LaboratoryPhysics) -> None:
        """
        Actual contained liquid mass participates in the start interlock.
        """
        load_pair(physics, second_volume=7.0)
        physics.centrifuge_command("close")
        physics.step(5000)
        assert physics.centrifuge.snapshot()[CentrifugeField.BALANCED] is False
        with pytest.raises(InvalidCentrifugeOperation):
            physics.centrifuge_command("start", rpm=120)

    @pytest.mark.parametrize("rpm", [True, float("nan"), -1, 301])
    def test_invalid_speed_is_rejected(
        self, physics: LaboratoryPhysics, rpm: float
    ) -> None:
        """
        Malformed or out-of-range speed requests cannot energize the drive.
        """
        with pytest.raises(InvalidCentrifugeOperation):
            physics.centrifuge_command("start", rpm=rpm)

    def test_unsecured_third_glass_prevents_start(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        An opposing secured pair cannot conceal a loose glass inside the rotor.
        """
        load_pair(physics)
        seat_tube(physics, LaboratoryBody.TEAL_TUBE, "C2")
        physics.centrifuge_command("close")
        physics.step(5000)
        with pytest.raises(InvalidCentrifugeOperation):
            physics.centrifuge_command("start", rpm=120)


class TestCentrifugeDynamics:
    """
    MuJoCo actuators rotate secured vessels and brake before release.
    """

    def test_balanced_pair_rotates_with_bounded_motor_and_stops_before_opening(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        A powered cycle obeys fixture, cover and bounded-force motor constraints.
        """
        load_pair(physics)
        with pytest.raises(InvalidCentrifugeOperation):
            physics.centrifuge_command("start", rpm=120)
        with pytest.raises(InvalidCentrifugeOperation):
            physics.set_target(LaboratoryBody.CLEAR_TUBE, [0, 0, 1.5])
        physics.centrifuge_command("close")
        physics.step(5000)
        assert physics.centrifuge.snapshot()[CentrifugeField.LID_CLOSED] is True
        before = physics.data.qpos.copy()
        physics.centrifuge_command("start", rpm=120)
        np.testing.assert_array_equal(physics.data.qpos, before)
        physics.step(500)
        assert 0 < physics.centrifuge.rpm < 120
        with pytest.raises(InvalidCentrifugeOperation):
            physics.centrifuge_command("open")
        with pytest.raises(InvalidCentrifugeOperation):
            physics.centrifuge_command("release", slot="C1")
        physics.step(4500)
        state = physics.centrifuge.snapshot()
        assert state[CentrifugeField.RPM] == pytest.approx(120, abs=2)
        radius = np.linalg.norm(physics.centrifuge.slots["C1"].position[:2])
        expected_force = (physics.centrifuge.rpm * math.tau / 60) ** 2 * radius / 9.81
        assert state[CentrifugeField.RELATIVE_FORCE] == pytest.approx(
            expected_force, abs=0.001
        )
        for slot in ("C1", "C3"):
            key = physics.centrifuge.slots[slot].key
            np.testing.assert_allclose(
                physics.data.xpos[physics.objects[key].body_id],
                physics.centrifuge.slot_position(slot),
                atol=0.003,
            )
        actuator = physics.centrifuge.rotor_actuator
        assert (
            abs(physics.data.actuator_force[actuator])
            <= physics.centrifuge.parameters.motor_torque
        )
        np.testing.assert_array_equal(physics.data.xfrc_applied, 0)
        physics.centrifuge_command("stop")
        with pytest.raises(InvalidCentrifugeOperation):
            physics.centrifuge_command("open")
        physics.step(6000)
        assert abs(physics.centrifuge.rpm) < physics.centrifuge.parameters.stopped_rpm
        physics.centrifuge_command("open")
        physics.step(5000)
        physics.centrifuge_command("release", slot="C1")
        assert physics.centrifuge.slots["C1"].key is None
        position = physics.data.xpos[
            physics.objects[LaboratoryBody.CLEAR_TUBE].body_id
        ].copy()
        physics.set_target(
            LaboratoryBody.CLEAR_TUBE, (position + [0, 0, 0.20]).tolist()
        )
        physics.step(2500)
        assert (
            physics.data.xpos[physics.objects[LaboratoryBody.CLEAR_TUBE].body_id][2]
            > position[2] + 0.18
        )

    def test_reset_releases_clamps_and_restores_open_lid(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        Reset restores free glasses and the instrument's loading configuration.
        """
        load_pair(physics)
        physics.reset()
        assert all(slot.key is None for slot in physics.centrifuge.slots.values())
        assert physics.centrifuge.snapshot()[CentrifugeField.LID_OPEN] is True
        assert physics.centrifuge.rpm == 0

    def test_timed_cycle_brakes_after_time_at_measured_speed(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        Acceleration time does not consume the requested processing duration.
        """
        load_pair(physics)
        physics.centrifuge_command("close")
        physics.step(5000)
        physics.centrifuge_command("start", rpm=120, duration=1)
        physics.step(500)
        assert physics.centrifuge.remaining_seconds == 1
        physics.step(7500)
        assert physics.centrifuge.remaining_seconds == 0
        assert physics.centrifuge.target_rpm == 0
        assert physics.centrifuge.stopped is True

    def test_configured_maximum_speed_retains_the_secured_pair(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        The advertised upper speed preserves finite contacts and fixture alignment.
        """
        load_pair(physics)
        physics.centrifuge_command("close")
        physics.step(5000)
        physics.centrifuge_command(
            "start", rpm=physics.centrifuge.maximum_rpm, duration=10
        )
        physics.step(7000)
        assert physics.centrifuge.rpm == pytest.approx(
            physics.centrifuge.maximum_rpm, abs=2
        )
        assert np.isfinite(physics.data.qpos).all()
        assert np.isfinite(physics.data.qvel).all()
        for slot in ("C1", "C3"):
            key = physics.centrifuge.slots[slot].key
            np.testing.assert_allclose(
                physics.data.xpos[physics.objects[key].body_id],
                physics.centrifuge.slot_position(slot),
                atol=0.003,
            )

    def test_resecuring_uses_the_current_rotor_frame(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        A released glass can be loaded again after the rotor changes angle.
        """
        load_pair(physics)
        physics.centrifuge_command("close")
        physics.step(5000)
        physics.centrifuge_command("start", rpm=120, duration=1)
        physics.step(8000)
        physics.centrifuge_command("open")
        physics.step(5000)
        physics.centrifuge_command("release", slot="C1")
        position = physics.centrifuge.slot_position("C1")
        physics.set_target(
            LaboratoryBody.CLEAR_TUBE, (position + [0, 0, 0.20]).tolist()
        )
        physics.step(2500)
        physics.set_target(LaboratoryBody.CLEAR_TUBE, position.tolist())
        physics.step(2500)
        physics.release()
        physics.step(1000)
        physics.centrifuge_command("secure", slot="C1", key=LaboratoryBody.CLEAR_TUBE)
        physics.step(500)
        np.testing.assert_allclose(
            physics.data.xpos[physics.objects[LaboratoryBody.CLEAR_TUBE].body_id],
            physics.centrifuge.slot_position("C1"),
            atol=0.001,
        )
