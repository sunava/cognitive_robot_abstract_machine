"""
The browser's scale transport runs against real contact simulation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import TracebackType

import numpy as np
import pytest
from typing_extensions import Any

from cramera.laboratory_physics import LaboratoryPhysics
from cramera.laboratory_physics_session import LaboratoryPhysicsSession
from cramera.laboratory_scale import ScaleField
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics import physics
from .test_laboratory_physics_robot import robot_physics


# %% browser protocol
class TransferField(StrEnum):
    """
    Messages connecting the browser controller to real simulation snapshots.
    """

    STATE = "state"
    CONFIG = "config"
    SELECTED = "selected"
    OBJECTS = "objects"
    KEY = "key"
    SCALE = "scale"
    LABORATORY = "laboratory"
    BOUNDS = "bounds"
    MINIMUM = "min"
    MAXIMUM = "max"
    COMMANDS = "commands"
    ERROR = "error"
    TYPE = "type"
    TARGET = "target"
    RELEASE = "release"
    POSITION = "position"
    ORIENTATION = "orientation"
    RUNNING = "running"


@dataclass
class BrowserTransport:
    """
    Exchange observed states with the unmodified JavaScript controller.
    """

    process: subprocess.Popen[str] = field(init=False)
    """
    Node process receiving one snapshot and returning commands per line.
    """

    def __enter__(self) -> BrowserTransport:
        """
        Start the browser controller without a browser or network service.
        """
        script = Path(__file__).parent / "dataset/laboratory_scale_controller.js"
        self.process = subprocess.Popen(
            ["node", str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Close the input stream and collect the completed child process.
        """
        self.process.stdin.close()
        self.process.wait(timeout=5)
        self.process.stdout.close()
        self.process.stderr.close()

    def observe(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        Let the controller decide from the authoritative simulation state.
        """
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        assert response, self.process.stderr.read()
        return json.loads(response)


# %% physical transfer acceptance
def transfer_to_scale(
    physics: LaboratoryPhysics,
    volume_ml: float = 5.0,
    key: LaboratoryBody = LaboratoryBody.CLEAR_TUBE,
) -> None:
    """
    Transport a vessel or stopper from its authored pose using browser commands.
    """
    if key in physics.liquid.tubes:
        physics.fill_liquid(key, volume_ml)
    physics.step(1500)
    mass = physics.model.body_mass[physics.objects[key].body_id] * 1000.0
    configuration = {
        TransferField.OBJECTS: [{TransferField.KEY: key}],
        TransferField.BOUNDS: {
            TransferField.MINIMUM: list(LaboratoryPhysicsSession.TARGET_MINIMUM),
            TransferField.MAXIMUM: list(LaboratoryPhysicsSession.TARGET_MAXIMUM),
        },
        TransferField.SCALE: physics.scene[TransferField.LABORATORY][
            TransferField.SCALE
        ],
    }
    commands = []
    with BrowserTransport() as browser:
        for _ in range(1000):
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
                commands.append(command)
                previous = physics.data.qpos.copy()
                if command[TransferField.TYPE] == TransferField.TARGET:
                    physics.set_target(
                        command[TransferField.KEY],
                        command[TransferField.POSITION],
                        command[TransferField.ORIENTATION],
                    )
                else:
                    assert command[TransferField.TYPE] == TransferField.RELEASE
                    physics.release()
                np.testing.assert_array_equal(physics.data.qpos, previous)
            if commands and commands[-1][TransferField.TYPE] == TransferField.RELEASE:
                break
            physics.step(33)
    assert [command[TransferField.TYPE] for command in commands] == [
        TransferField.TARGET,
        TransferField.TARGET,
        TransferField.TARGET,
        TransferField.TARGET,
        TransferField.RELEASE,
    ]
    physics.step(2500)
    scale = physics.snapshot()[TransferField.SCALE]
    assert physics.target is None
    assert scale[ScaleField.STABLE]
    assert scale[TransferField.OBJECTS] == [key]
    assert scale[ScaleField.GRAMS] == pytest.approx(mass, abs=0.015)
    if key in physics.liquid.tubes:
        assert physics.liquid.tubes[key].volume_ml == pytest.approx(volume_ml)
    assert not physics.data.xfrc_applied.any()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_moves_filled_glass_from_rack_to_measured_pan(
    physics: LaboratoryPhysics,
) -> None:
    """
    The actual client releases a supported full load without assigning poses.
    """
    transfer_to_scale(physics)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_scale_transport_shares_the_pr2_contact_world(
    robot_physics: LaboratoryPhysics,
) -> None:
    """
    The same transport succeeds beside the idle robot's collision geometry.
    """
    transfer_to_scale(robot_physics)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_weighs_an_empty_glass_for_tare(physics: LaboratoryPhysics) -> None:
    """
    The empty vessel reaches a settled pan load before the user tares it.
    """
    transfer_to_scale(physics, volume_ml=0.0)
    physics.tare_scale()
    assert physics.snapshot()[TransferField.SCALE][ScaleField.GRAMS] == 0.0


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_weighs_an_empty_glass_beside_the_pr2(
    robot_physics: LaboratoryPhysics,
) -> None:
    """
    The empty vessel can be weighed in the shared robot simulation.
    """
    transfer_to_scale(robot_physics, volume_ml=0.0)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_weighs_the_flat_bottomed_stopper(physics: LaboratoryPhysics) -> None:
    """
    The vial support also permits the smaller free stopper to reach the pan.
    """
    transfer_to_scale(physics, volume_ml=0.0, key=LaboratoryBody.STOPPER)
