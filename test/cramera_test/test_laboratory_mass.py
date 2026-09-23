"""
Contained liquid contributes a real load without resetting robot motion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from cramera.laboratory_liquid import LaboratoryLiquid, TubeKey
from cramera.laboratory_mass import LaboratoryLiquidMass
from cramera.laboratory_physics import (
    LaboratoryCollisionModel,
    LaboratoryPhysicsParameters,
)

from .test_laboratory_bundle import laboratory_directory


# %% independent liquid and rigid-body fixture
@dataclass
class LiquidLoad:
    """
    An authored laboratory before liquid mass is coupled into its bodies.
    """

    model: mujoco.MjModel
    """Compiled dry-body collisions."""
    data: mujoco.MjData
    """
    Live contact integration state.
    """

    liquid: LaboratoryLiquid
    """Volume-conserving liquid state."""
    masses: LaboratoryLiquidMass
    """
    Coupling that changes the bodies' inertial properties.
    """

    def settle(self) -> None:
        """
        Allow the supported glasses to reach static contact equilibrium.
        """
        for _ in range(1000):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def supported_mass(self, key: str) -> float:
        """
        Read the vertical contact reaction on one supported glass in kilograms.
        """
        body_id = self.model.body(key).id
        load = np.zeros(3)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = self.model.geom_bodyid[contact.geom1]
            second = self.model.geom_bodyid[contact.geom2]
            if body_id not in (first, second) or contact.efc_address < 0:
                continue
            effort = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, effort)
            world_force = contact.frame.reshape(3, 3).T @ effort[:3]
            load += world_force * (1 if body_id == second else -1)
        return float(load[2] / -self.model.opt.gravity[2])


@pytest.fixture
def liquid_load(laboratory_directory: Path) -> LiquidLoad:
    """
    Create separate authored liquid and dry-body physical states.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    model = LaboratoryCollisionModel(
        laboratory_directory, scene, LaboratoryPhysicsParameters()
    ).build()
    data = mujoco.MjData(model)
    liquid = LaboratoryLiquid(scene)
    masses = LaboratoryLiquidMass(
        model, {key: model.body(key).id for key in liquid.tubes}
    )
    return LiquidLoad(model, data, liquid, masses)


# %% force and state contracts
class TestContainedLiquidWeight:
    """
    The solver carries contained liquid as a load on the glass.
    """

    def test_five_milliliters_increases_actual_support_reaction(
        self, liquid_load: LiquidLoad
    ) -> None:
        key = TubeKey.CLEAR
        liquid_load.masses.synchronize(liquid_load.liquid)
        liquid_load.settle()
        before = liquid_load.supported_mass(key)
        added_volume = 5.0
        liquid_load.liquid.fill(key, added_volume)
        liquid_load.masses.synchronize(liquid_load.liquid)
        liquid_load.settle()
        added_mass = (
            added_volume
            * liquid_load.masses.density_grams_per_milliliter
            / liquid_load.masses.grams_per_kilogram
        )
        assert liquid_load.supported_mass(key) - before == pytest.approx(
            added_mass, abs=1e-8
        )

    def test_draining_restores_dry_mass_and_inertia(
        self, liquid_load: LiquidLoad
    ) -> None:
        key = TubeKey.CLEAR
        body_id = liquid_load.model.body(key).id
        dry_mass = liquid_load.model.body_mass[body_id]
        dry_inertia = liquid_load.model.body_inertia[body_id].copy()
        liquid_load.liquid.fill(key, 5.0)
        liquid_load.masses.synchronize(liquid_load.liquid)
        liquid_load.liquid.fill(key, 0.0)
        liquid_load.masses.synchronize(liquid_load.liquid)
        assert liquid_load.model.body_mass[body_id] == dry_mass
        np.testing.assert_array_equal(
            liquid_load.model.body_inertia[body_id], dry_inertia
        )

    def test_synchronization_preserves_live_integrator_state(
        self, liquid_load: LiquidLoad
    ) -> None:
        liquid_load.settle()
        liquid_load.data.qvel[:] = np.arange(liquid_load.model.nv) * 1e-4
        positions_before = liquid_load.data.qpos.copy()
        velocities_before = liquid_load.data.qvel.copy()
        controls_before = liquid_load.data.ctrl.copy()
        time_before = liquid_load.data.time
        liquid_load.liquid.fill(TubeKey.CLEAR, 5.0)
        liquid_load.masses.synchronize(liquid_load.liquid)
        np.testing.assert_array_equal(liquid_load.data.qpos, positions_before)
        np.testing.assert_array_equal(liquid_load.data.qvel, velocities_before)
        np.testing.assert_array_equal(liquid_load.data.ctrl, controls_before)
        assert liquid_load.data.time == time_before

    def test_unchanged_liquid_does_not_recompute_model_constants(
        self, liquid_load: LiquidLoad
    ) -> None:
        liquid_load.masses.synchronize(liquid_load.liquid)
        with patch.object(mujoco, "mj_setConst", wraps=mujoco.mj_setConst) as refresh:
            assert liquid_load.masses.synchronize(liquid_load.liquid) is False
        refresh.assert_not_called()
