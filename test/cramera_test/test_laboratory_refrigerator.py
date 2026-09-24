"""
Measured refrigerator articulation, supported samples and reduced cooling.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from cramera.laboratory_physics import LaboratoryPhysics
from cramera.laboratory_refrigerator import (
    InvalidRefrigeratorOperation,
    LaboratoryRefrigerator,
)
from cramera.laboratory_refrigerator_command import (
    RefrigeratorAction,
    RefrigeratorField,
)
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics import physics


# %% bounded physical door
class TestRefrigeratorDoor:
    """
    Door requests advance an actual bounded hinge without changing object poses.
    """

    def test_door_opens_physically_and_reports_measured_joint(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        A door request changes its actuator reference while poses remain engine-owned.
        """
        refrigerator = physics.refrigerator
        initial = physics.snapshot()[RefrigeratorField.REFRIGERATOR]
        assert initial[RefrigeratorField.AVAILABLE] is True
        assert initial[RefrigeratorField.DOOR_CLOSED] is True
        before = physics.data.qpos.copy()
        physics.refrigerator_command(RefrigeratorAction.OPEN)
        np.testing.assert_array_equal(physics.data.qpos, before)
        physics.step(1000)
        assert 0 < refrigerator.door_angle < refrigerator.open_angle
        physics.step(5000)
        state = refrigerator.snapshot()
        assert state[RefrigeratorField.DOOR_OPEN] is True
        assert state[RefrigeratorField.JOINTS] == {
            refrigerator.metadata[RefrigeratorField.DOOR_JOINT]: refrigerator.door_angle
        }
        assert (
            abs(physics.data.actuator_force[refrigerator.actuator])
            <= refrigerator.parameters.maximum_torque
        )
        physics.refrigerator_command(RefrigeratorAction.CLOSE)
        physics.step(6000)
        assert refrigerator.snapshot()[RefrigeratorField.DOOR_CLOSED] is True

    def test_reset_restores_closed_cooled_cabinet(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        Reset discards cabinet heat and restores the authored closed configuration.
        """
        refrigerator = physics.refrigerator
        physics.refrigerator_command(RefrigeratorAction.OPEN)
        physics.step(5000)
        assert refrigerator.temperature > refrigerator.target_temperature
        physics.reset()
        state = refrigerator.snapshot()
        assert state[RefrigeratorField.DOOR_CLOSED] is True
        assert (
            state[RefrigeratorField.TEMPERATURE]
            == state[RefrigeratorField.TARGET_TEMPERATURE]
        )
        assert all(
            slot[RefrigeratorField.KEY] is None
            for slot in state[RefrigeratorField.SLOTS]
        )

    def test_older_bundle_without_metadata_remains_usable(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        Optional equipment is unavailable without preventing older worlds from stepping.
        """
        refrigerator = LaboratoryRefrigerator(
            physics.model, physics.data, {}, tuple(physics.liquid.tubes)
        )
        refrigerator.reset()
        refrigerator.advance(physics.timestep)
        assert refrigerator.snapshot()[RefrigeratorField.AVAILABLE] is False
        assert refrigerator.snapshot()[RefrigeratorField.JOINTS] == {}
        with pytest.raises(InvalidRefrigeratorOperation):
            refrigerator.command(RefrigeratorAction.OPEN)


# %% simulated cooling and contact-grounded inventory
class TestCabinetTemperature:
    """
    Temperature follows simulated time and the measured door aperture.
    """

    def test_open_door_warms_and_closed_door_cools(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        The cabinet temperature responds to simulated exposure and cooling over time.
        """
        refrigerator = physics.refrigerator
        physics.refrigerator_command(RefrigeratorAction.OPEN)
        physics.step(6000)
        warmed = refrigerator.temperature
        assert (
            refrigerator.target_temperature
            < warmed
            < refrigerator.parameters.ambient_temperature
        )
        physics.refrigerator_command(RefrigeratorAction.CLOSE)
        physics.step(6000)
        closed = refrigerator.temperature
        physics.step(2000)
        assert refrigerator.target_temperature < refrigerator.temperature < closed
        assert (
            refrigerator.snapshot()[RefrigeratorField.TEMPERATURE_MODEL] == "simulated"
        )

    def test_repeated_snapshots_do_not_advance_temperature(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        Reading observations never advances the reduced thermal model.
        """
        before = physics.refrigerator.temperature
        for _ in range(10):
            physics.refrigerator.snapshot()
        assert physics.refrigerator.temperature == before


class TestSupportedRefrigeratorSamples:
    """
    A slot contains a sample only when actual shelf contact supports it upright.
    """

    def test_occupancy_requires_support_and_clears_after_removal(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        Proximity alone cannot publish stored inventory before a shelf supports the
        glass.
        """
        refrigerator = physics.refrigerator
        slot = refrigerator.metadata[RefrigeratorField.SLOTS][0]
        key = LaboratoryBody.CLEAR_TUBE
        body = physics.objects[key].body_id
        joint = physics.model.body_jntadr[body]
        coordinate = physics.model.jnt_qposadr[joint]
        velocity = physics.model.jnt_dofadr[joint]
        physics.data.qpos[coordinate : coordinate + 3] = np.asarray(
            slot[RefrigeratorField.POSITION]
        ) + [0, 0, 0.002]
        physics.data.qpos[coordinate + 3 : coordinate + 7] = [1, 0, 0, 0]
        physics.data.qvel[velocity : velocity + 6] = 0
        mujoco.mj_forward(physics.model, physics.data)
        assert (
            refrigerator.snapshot()[RefrigeratorField.SLOTS][0][RefrigeratorField.KEY]
            is None
        )
        physics.step(1800)
        assert (
            refrigerator.snapshot()[RefrigeratorField.SLOTS][0][RefrigeratorField.KEY]
            == key
        )
        physics.data.qpos[coordinate + 2] += 0.04
        physics.data.qvel[velocity : velocity + 6] = 0
        mujoco.mj_forward(physics.model, physics.data)
        assert (
            refrigerator.snapshot()[RefrigeratorField.SLOTS][0][RefrigeratorField.KEY]
            is None
        )

    def test_tilted_sample_is_not_reported_as_stored(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        A horizontal glass cannot masquerade as upright supported sample storage.
        """
        refrigerator = physics.refrigerator
        slot = refrigerator.metadata[RefrigeratorField.SLOTS][0]
        body = physics.objects[LaboratoryBody.CLEAR_TUBE].body_id
        coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[body]]
        physics.data.qpos[coordinate : coordinate + 3] = slot[
            RefrigeratorField.POSITION
        ]
        physics.data.qpos[coordinate + 3 : coordinate + 7] = [
            math.sqrt(0.5),
            math.sqrt(0.5),
            0,
            0,
        ]
        mujoco.mj_forward(physics.model, physics.data)
        assert (
            refrigerator.snapshot()[RefrigeratorField.SLOTS][0][RefrigeratorField.KEY]
            is None
        )


# %% stored sample protection
class TestStoredSampleAccess:
    """
    Door geometry and existing sample occupancy constrain manual access.
    """

    def test_closed_cabinet_blocks_manual_retrieval_before_opening(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        Stored samples remain inaccessible until the measured door aperture is clear.
        """
        refrigerator = physics.refrigerator
        slot = refrigerator.metadata[RefrigeratorField.SLOTS][0]
        key = LaboratoryBody.CLEAR_TUBE
        body = physics.objects[key].body_id
        coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[body]]
        physics.data.qpos[coordinate : coordinate + 3] = np.asarray(
            slot[RefrigeratorField.POSITION]
        ) + [0, 0, 0.002]
        mujoco.mj_forward(physics.model, physics.data)
        physics.step(1500)
        with pytest.raises(InvalidRefrigeratorOperation):
            physics.set_target(
                key,
                (np.asarray(slot[RefrigeratorField.POSITION]) + [0, 0, 0.05]).tolist(),
            )
        assert physics.target is None
        physics.refrigerator_command(RefrigeratorAction.OPEN)
        physics.step(6000)
        physics.set_target(
            key, (np.asarray(slot[RefrigeratorField.POSITION]) + [0, 0, 0.05]).tolist()
        )
        assert physics.target.key == key


class TestGlassDoorContact:
    """
    Glazing blocks a force-driven sample until the physical door opens.
    """

    def test_closed_glass_blocks_force_driven_sample_entry(
        self, physics: LaboratoryPhysics
    ) -> None:
        """
        An inward force target produces door contacts rather than passing through
        glazing.
        """
        refrigerator = physics.refrigerator
        slot = refrigerator.metadata[RefrigeratorField.SLOTS][1]
        key = LaboratoryBody.CLEAR_TUBE
        body = physics.objects[key].body_id
        coordinate = physics.model.jnt_qposadr[physics.model.body_jntadr[body]]
        start = np.asarray(slot[RefrigeratorField.POSITION]) + [0, -0.35, 0.06]
        target = np.asarray(slot[RefrigeratorField.POSITION]) + [0, 0, 0.06]
        physics.data.qpos[coordinate : coordinate + 3] = start
        mujoco.mj_forward(physics.model, physics.data)
        physics.set_target(key, target.tolist())
        physics.step(2500)
        blocked = physics.snapshot()
        assert np.linalg.norm(np.asarray(blocked["objects"][key][:3]) - target) > 0.1
        assert any(
            {contact["bodyA"], contact["bodyB"]}
            == {key, refrigerator.metadata[RefrigeratorField.DOOR_BODY]}
            and contact["force"] > 0
            for contact in blocked["contacts"]
        )
