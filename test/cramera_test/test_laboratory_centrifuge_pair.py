"""
One browser operation prepares balanced centrifuge samples without robot execution.
"""

from __future__ import annotations

import shutil

import pytest

from cramera.laboratory_centrifuge_command import CentrifugeAction, CentrifugeSlot
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_mixing_adapter import program_world
from .test_laboratory_centrifuge_transport import (
    InstrumentTransportField,
    transport_sample,
)


# %% standalone pair preparation
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("slot", [CentrifugeSlot.C1, CentrifugeSlot.C2])
def test_one_click_loads_opposite_equal_samples_without_robot(
    program_world: LaboratoryProgramPhysics, slot: CentrifugeSlot
) -> None:
    """
    Both opposite placements complete from one operation and permit a cycle.
    """
    engine = program_world
    engine.step(1000)
    unused_volume = engine.physics.liquid.tubes[LaboratoryBody.TEAL_TUBE].volume_ml
    transport_sample(
        engine, LaboratoryBody.CLEAR_TUBE, slot, InstrumentTransportField.PAIR
    )
    centrifuge = engine.physics.centrifuge
    first = centrifuge.slots[slot]
    second = centrifuge.slots[first.opposite]
    assert first.key == LaboratoryBody.CLEAR_TUBE
    assert second.key == LaboratoryBody.AMBER_TUBE
    assert centrifuge.balanced
    assert centrifuge.slot_mass(first) == centrifuge.slot_mass(second)
    for key in (first.key, second.key):
        assert engine.physics.liquid.tubes[key].volume_ml == pytest.approx(5.0)
    assert (
        engine.physics.liquid.tubes[LaboratoryBody.TEAL_TUBE].volume_ml == unused_volume
    )
    assert engine.program.state == ProgramState.IDLE
    assert engine.physics.target is None
    engine.centrifuge_command(CentrifugeAction.CLOSE)
    engine.step(5000)
    engine.centrifuge_command(CentrifugeAction.START)
    assert centrifuge.target_rpm == centrifuge.parameters.default_rpm


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("sample_volume", [0.0, 7.0])
def test_pair_control_completes_existing_sample_without_changing_its_fill(
    program_world: LaboratoryProgramPhysics,
    sample_volume: float,
) -> None:
    """
    A single secured sample receives an equal counterpart without reset or refill.
    """
    engine = program_world
    engine.fill_liquid(LaboratoryBody.CLEAR_TUBE, sample_volume)
    engine.step(1000)
    transport_sample(
        engine,
        LaboratoryBody.CLEAR_TUBE,
        CentrifugeSlot.C1,
        InstrumentTransportField.LOAD,
    )
    initial_time = engine.physics.data.time
    transport_sample(
        engine,
        LaboratoryBody.CLEAR_TUBE,
        CentrifugeSlot.C1,
        InstrumentTransportField.PAIR,
    )
    centrifuge = engine.physics.centrifuge
    assert centrifuge.slots[CentrifugeSlot.C1].key == LaboratoryBody.CLEAR_TUBE
    assert centrifuge.slots[CentrifugeSlot.C3].key == LaboratoryBody.AMBER_TUBE
    assert centrifuge.balanced
    for key in (LaboratoryBody.CLEAR_TUBE, LaboratoryBody.AMBER_TUBE):
        assert engine.physics.liquid.tubes[key].volume_ml == pytest.approx(
            sample_volume
        )
    assert engine.physics.data.time > initial_time
    assert engine.program.state == ProgramState.IDLE
    assert engine.physics.target is None
