"""The browser seats, secures, spins and retrieves samples in the PR2 world."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest

from cramera.laboratory_physics_program import LaboratoryProgramPhysics
from cramera.laboratory_physics_session import LaboratoryPhysicsSession
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_mixing_adapter import program_world
from .test_laboratory_scale_transport import BrowserTransport, TransferField


# %% browser process
class InstrumentTransportField(StrEnum):
    """Fields extending the ordinary force-target controller protocol."""

    CENTRIFUGE = "centrifuge"
    """Instrument state and command type."""
    SLOT = "slot"
    """Selected rotating holder."""
    OPERATION = "operation"
    """Load or unload operation requested from the browser."""
    LOAD = "load"
    """Move a selected sample into a holder."""
    PAIR = "pair"
    """Prepare and load an equal-mass pair without a robot program."""
    LIQUID = "liquid"
    """Fill a selected loose sample before loading."""
    VOLUME = "volumeMl"
    """Liquid fill amount in millilitres."""
    UNLOAD = "unload"
    """Release a holder and raise its sample."""
    MOVING = "moving"
    """Whether the browser still has an active sequence."""


@dataclass
class CentrifugeTransport(BrowserTransport):
    """Connect the instrument controller to the real contact engine."""

    def __enter__(self) -> CentrifugeTransport:
        """Start the instrument-specific controller while sharing the line protocol."""
        script = Path(__file__).parent / "dataset/laboratory_centrifuge_controller.js"
        self.process = subprocess.Popen(
            ["node", str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return self


def transport_sample(
    engine: LaboratoryProgramPhysics,
    key: LaboratoryBody,
    slot: str,
    operation: InstrumentTransportField,
) -> None:
    """Execute browser-generated force targets without assigning object poses."""
    configuration = {
        TransferField.OBJECTS: [
            {TransferField.KEY: object_key} for object_key in engine.physics.objects
        ],
        TransferField.BOUNDS: {
            TransferField.MINIMUM: list(LaboratoryPhysicsSession.TARGET_MINIMUM),
            TransferField.MAXIMUM: list(LaboratoryPhysicsSession.TARGET_MAXIMUM),
        },
        InstrumentTransportField.CENTRIFUGE: engine.physics.scene[
            TransferField.LABORATORY
        ][InstrumentTransportField.CENTRIFUGE],
    }
    issued = []
    with CentrifugeTransport() as browser:
        for _ in range(600):
            state = engine.snapshot()
            state[TransferField.STATE] = TransferField.RUNNING
            answer = browser.observe(
                {
                    TransferField.STATE: state,
                    TransferField.CONFIG: configuration,
                    TransferField.SELECTED: key,
                    InstrumentTransportField.SLOT: slot,
                    InstrumentTransportField.OPERATION: operation,
                }
            )
            assert answer[TransferField.ERROR] is None, answer
            for command in answer[TransferField.COMMANDS]:
                issued.append(command.copy())
                before = engine.physics.data.qpos.copy()
                kind = command.pop(TransferField.TYPE)
                if kind == TransferField.TARGET:
                    engine.set_target(
                        command[TransferField.KEY],
                        command[TransferField.POSITION],
                        command.get(TransferField.ORIENTATION),
                    )
                elif kind == TransferField.RELEASE:
                    engine.release()
                elif kind == InstrumentTransportField.LIQUID:
                    engine.fill_liquid(
                        command[TransferField.KEY],
                        command[InstrumentTransportField.VOLUME],
                    )
                else:
                    assert kind == InstrumentTransportField.CENTRIFUGE
                    engine.centrifuge_command(**command)
                np.testing.assert_array_equal(engine.physics.data.qpos, before)
            if not answer[InstrumentTransportField.MOVING]:
                break
            engine.step(100)
    assert issued, "The browser must issue an actual loading or unloading sequence."
    assert not answer[InstrumentTransportField.MOVING], engine.snapshot()


# %% complete timed operation
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_operates_loaded_centrifuge_beside_pr2(
    program_world: LaboratoryProgramPhysics,
) -> None:
    """Opposite loaded glasses complete a timed cycle and can then be retrieved."""
    engine = program_world
    for key in (LaboratoryBody.CLEAR_TUBE, LaboratoryBody.AMBER_TUBE):
        engine.fill_liquid(key, 5.0)
    engine.step(1000)
    for key, slot in (
        (LaboratoryBody.CLEAR_TUBE, "C1"),
        (LaboratoryBody.AMBER_TUBE, "C3"),
    ):
        transport_sample(engine, key, slot, InstrumentTransportField.LOAD)
        assert engine.physics.centrifuge.slots[slot].key == key
        assert engine.physics.target is None
    engine.centrifuge_command("close")
    engine.step(5000)
    engine.centrifuge_command("start", rpm=120.0, duration=2.0)
    acceleration_time = 120.0 / engine.physics.centrifuge.parameters.acceleration_rpm
    engine.step(round((acceleration_time + 0.5) / engine.physics.model.opt.timestep))
    assert engine.physics.centrifuge.rpm == pytest.approx(120.0, abs=2.0)
    engine.step(9000)
    assert (
        abs(engine.physics.centrifuge.rpm)
        < engine.physics.centrifuge.parameters.stopped_rpm
    )
    engine.centrifuge_command("open")
    engine.step(5000)
    seated = engine.physics.centrifuge.slot_position("C1").copy()
    transport_sample(
        engine, LaboratoryBody.CLEAR_TUBE, "C1", InstrumentTransportField.UNLOAD
    )
    assert engine.physics.centrifuge.slots["C1"].key is None
    position = engine.snapshot()[TransferField.OBJECTS][LaboratoryBody.CLEAR_TUBE][:3]
    assert position[2] > seated[2] + 0.18
    for key in (LaboratoryBody.CLEAR_TUBE, LaboratoryBody.AMBER_TUBE):
        assert engine.physics.liquid.tubes[key].volume_ml == pytest.approx(5.0)
