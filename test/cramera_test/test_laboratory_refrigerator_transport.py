"""Production browser storage paths operate through real refrigerator contacts."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest

from cramera.laboratory_physics import LaboratoryPhysics
from cramera.laboratory_physics_session import LaboratoryPhysicsSession, PhysicsField
from cramera.laboratory_refrigerator_command import (
    RefrigeratorAction,
    RefrigeratorField,
)
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics import physics
from .test_laboratory_scale_transport import BrowserTransport, TransferField


# %% physical browser transport
class StorageOperation(StrEnum):
    """Storage path requests understood by the browser controller fixture."""

    STORE = "store"
    """Move an upright sample through the front aperture into a shelf holder."""
    RETRIEVE = "retrieve"
    """Remove a sample through the front aperture and return it to a free rack slot."""


@dataclass
class RefrigeratorTransport(BrowserTransport):
    """Connect production storage controls to measured physical sample positions."""

    def __enter__(self) -> RefrigeratorTransport:
        """Start the line-oriented production-controller fixture."""
        script = Path(__file__).parent / "dataset/laboratory_refrigerator_controller.js"
        self.process = subprocess.Popen(
            ["node", str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return self


def transport_sample(
    physics: LaboratoryPhysics, operation: StorageOperation, slot: str, key: str
) -> None:
    """Apply browser force commands until a supported transfer completes."""
    configuration = {
        TransferField.OBJECTS: [{TransferField.KEY: item} for item in physics.objects],
        TransferField.BOUNDS: {
            TransferField.MINIMUM: list(LaboratoryPhysicsSession.TARGET_MINIMUM),
            TransferField.MAXIMUM: list(LaboratoryPhysicsSession.TARGET_MAXIMUM),
        },
        RefrigeratorField.REFRIGERATOR: physics.scene[TransferField.LABORATORY][
            RefrigeratorField.REFRIGERATOR
        ],
        PhysicsField.RACK_SLOTS: physics.scene[TransferField.LABORATORY][
            PhysicsField.RACK_SLOTS
        ],
    }
    issued = []
    with RefrigeratorTransport() as browser:
        for index in range(700):
            state = physics.snapshot()
            state[TransferField.STATE] = TransferField.RUNNING
            message = {TransferField.STATE: state, TransferField.CONFIG: configuration}
            if index == 0:
                message.update(
                    {
                        RefrigeratorField.ACTION: operation,
                        TransferField.SELECTED: key,
                        "slot": slot,
                    }
                )
            answer = browser.observe(message)
            assert answer[TransferField.ERROR] is None, answer
            for command in answer[TransferField.COMMANDS]:
                issued.append(command.copy())
                before = physics.data.qpos.copy()
                kind = command.pop(TransferField.TYPE)
                if kind == TransferField.TARGET:
                    physics.set_target(
                        command[TransferField.KEY],
                        command[TransferField.POSITION],
                        command.get(TransferField.ORIENTATION),
                    )
                elif kind == TransferField.RELEASE:
                    physics.release()
                else:
                    assert kind == RefrigeratorField.REFRIGERATOR
                    physics.refrigerator_command(command[RefrigeratorField.ACTION])
                np.testing.assert_array_equal(physics.data.qpos, before)
            if not answer["moving"]:
                break
            physics.step(100)
    assert issued
    assert answer["moving"] is False, answer
    assert physics.target is None
    physics.step(700)


# %% front-entry storage and supported rack return
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("slot_index", [0, 5])
def test_browser_stores_and_retrieves_supported_sample(
    physics: LaboratoryPhysics, slot_index: int
) -> None:
    """Both shelf heights accept a physical sample and return it to a free rack."""
    physics.step(1000)
    refrigerator = physics.refrigerator
    slot = refrigerator.metadata[RefrigeratorField.SLOTS][slot_index]
    key = LaboratoryBody.CLEAR_TUBE
    physics.refrigerator_command(RefrigeratorAction.OPEN)
    physics.step(6000)
    transport_sample(physics, StorageOperation.STORE, slot[RefrigeratorField.ID], key)
    assert refrigerator.occupants()[slot[RefrigeratorField.ID]] == key
    physics.refrigerator_command(RefrigeratorAction.CLOSE)
    physics.step(6000)
    assert refrigerator.door_closed is True
    assert refrigerator.occupants()[slot[RefrigeratorField.ID]] == key
    physics.refrigerator_command(RefrigeratorAction.OPEN)
    physics.step(6000)
    transport_sample(
        physics, StorageOperation.RETRIEVE, slot[RefrigeratorField.ID], key
    )
    assert refrigerator.occupants()[slot[RefrigeratorField.ID]] is None
    rack_poses = [
        np.asarray(item["pose"][:3])
        for item in physics.scene[TransferField.LABORATORY][PhysicsField.RACK_SLOTS]
    ]
    observed = np.asarray(physics.snapshot()["objects"][key][:3])
    assert any(np.linalg.norm(observed - position) < 0.003 for position in rack_poses)
