"""
The production browser controller approaches the station through its open front.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from cramera.laboratory_liquid import TubeKey
from cramera.laboratory_physics import LaboratoryPhysics
from cramera.laboratory_physics_session import LaboratoryPhysicsSession
from cramera.laboratory_pipetting import (
    PipettingAction,
    PipettingSource,
    PipettingState,
)

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics import physics
from .test_laboratory_scale_transport import BrowserTransport, TransferField


# %% controller transport
@dataclass
class PipettingTransport(BrowserTransport):
    """
    Connect production browser force commands to measured MuJoCo poses.
    """

    def __enter__(self) -> PipettingTransport:
        """
        Start the station-specific controller with the existing line protocol.
        """
        script = Path(__file__).parent / "dataset/laboratory_pipetting_controller.js"
        self.process = subprocess.Popen(
            ["node", str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return self


def position_receiver(physics: LaboratoryPhysics, key: str = TubeKey.CLEAR) -> None:
    """
    Follow browser-generated waypoints without changing authoritative poses.
    """
    configuration = {
        TransferField.OBJECTS: [{TransferField.KEY: item} for item in physics.objects],
        TransferField.BOUNDS: {
            TransferField.MINIMUM: list(LaboratoryPhysicsSession.TARGET_MINIMUM),
            TransferField.MAXIMUM: list(LaboratoryPhysicsSession.TARGET_MAXIMUM),
        },
        "pipetting": physics.scene[TransferField.LABORATORY]["pipetting"],
    }
    issued = []
    with PipettingTransport() as browser:
        for _ in range(300):
            state = physics.snapshot()
            state[TransferField.STATE] = TransferField.RUNNING
            answer = browser.observe(
                {
                    TransferField.STATE: state,
                    TransferField.CONFIG: configuration,
                    TransferField.SELECTED: key,
                }
            )
            assert answer[TransferField.ERROR] is None, answer
            for command in answer[TransferField.COMMANDS]:
                issued.append(command.copy())
                before = physics.data.qpos.copy()
                kind = command.pop(TransferField.TYPE)
                assert kind == TransferField.TARGET
                physics.set_target(
                    command[TransferField.KEY],
                    command[TransferField.POSITION],
                    command.get(TransferField.ORIENTATION),
                )
                np.testing.assert_array_equal(physics.data.qpos, before)
            if not answer["moving"]:
                break
            physics.step(100)
    assert issued
    assert not answer["moving"], answer
    assert answer["canDispense"] is True, answer
    physics.step(500)


# %% physically seated receiver
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_positions_and_doses_both_reagents(physics: LaboratoryPhysics):
    """
    The normal UI path places a real glass where both reagent channels can dose.
    """
    physics.step(1000)
    position_receiver(physics)
    for source in PipettingSource:
        physics.pipetting_command(
            PipettingAction.DISPENSE, key=TubeKey.CLEAR, source=source, volume_ml=1
        )
        physics.step(1100)
        assert physics.pipetting.state == PipettingState.COMPLETE
    assert physics.liquid.tubes[TubeKey.CLEAR].volume_ml == pytest.approx(2)
    physics.release()
    physics.step(1000)
    np.testing.assert_allclose(
        physics.snapshot()["objects"][TubeKey.CLEAR][:3],
        physics.pipetting.target_position,
        atol=physics.pipetting.parameters.position_tolerance,
    )
    assert any(
        TubeKey.CLEAR in (contact["bodyA"], contact["bodyB"])
        for contact in physics.snapshot()["contacts"]
    )
