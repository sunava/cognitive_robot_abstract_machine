"""
Station dosing changes real MuJoCo vessel mass without writing receiver poses.
"""

from pathlib import Path
import json
import shutil

import numpy as np
import pytest

from cramera.laboratory_physics import LaboratoryPhysics
from cramera.laboratory_pipetting import (
    InvalidPipettingCommand,
    PipettingAction,
    PipettingField,
    PipettingSource,
    PipettingState,
)
from cramera.laboratory_liquid import TubeKey

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_pipetting_transport import position_receiver


# %% force-driven receiver positioning
@pytest.fixture
def dispensing_physics(laboratory_directory: Path) -> LaboratoryPhysics:
    """
    Create a station-enabled contact world from the generated lab fixtures.
    """
    scene_path = laboratory_directory / "scene.json"
    scene = json.loads(scene_path.read_text())
    scene["laboratory"][PipettingField.PIPETTING] = {
        PipettingField.AVAILABLE: True,
        PipettingField.TARGET_POSITION: [-0.10, -0.11, 0.916],
        PipettingField.NOZZLE_POSITION: [-0.10, -0.11, 1.080],
    }
    scene_path.write_text(json.dumps(scene))
    return LaboratoryPhysics(laboratory_directory)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_pump_uses_measured_receiver_pose_and_updates_rigid_mass(dispensing_physics):
    """
    A desired target alone cannot dose, while an aligned measured vessel gains mass.
    """
    physics = dispensing_physics
    key = TubeKey.CLEAR
    pose = physics.snapshot()["objects"][key]
    physics.set_target(key, [pose[0], pose[1], 1.17])
    with pytest.raises(InvalidPipettingCommand):
        physics.pipetting_command(
            PipettingAction.DISPENSE, key=key, source=PipettingSource.AMBER, volume_ml=1
        )
    position_receiver(physics, key)
    before_pose = physics.data.qpos.copy()
    body = physics.objects[key].body_id
    before_mass = physics.model.body_mass[body]
    physics.pipetting_command(
        PipettingAction.DISPENSE, key=key, source=PipettingSource.AMBER, volume_ml=1
    )
    np.testing.assert_array_equal(physics.data.qpos, before_pose)
    physics.step(1100)
    assert (
        physics.snapshot()[PipettingField.PIPETTING][PipettingField.STATE]
        == PipettingState.COMPLETE
    )
    expected_added_mass = (
        physics.liquid_mass.density_grams_per_milliliter
        / physics.liquid_mass.grams_per_kilogram
    )
    assert physics.model.body_mass[body] - before_mass == pytest.approx(
        expected_added_mass
    )
    assert physics.liquid.tubes[key].volume_ml == pytest.approx(1)
    physics.reset()
    assert physics.model.body_mass[body] == before_mass
    assert physics.pipetting.state == PipettingState.IDLE
    assert all(
        reservoir.volume_ml == physics.pipetting.parameters.reservoir_ml
        for reservoir in physics.pipetting.reservoirs.values()
    )


def test_explicit_fill_cannot_replace_a_receiver_during_active_dosing(
    dispensing_physics,
):
    """
    Manual liquid editing cannot overwrite the pump's active receiver accounting.
    """
    physics = dispensing_physics
    key = TubeKey.CLEAR
    physics.pipetting.key = key
    physics.pipetting.state = PipettingState.DISPENSING
    before = physics.liquid.snapshot()
    with pytest.raises(InvalidPipettingCommand):
        physics.fill_liquid(key, 3)
    assert physics.liquid.snapshot() == before
