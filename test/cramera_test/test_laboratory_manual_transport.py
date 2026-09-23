"""
A revised document changes the same physical browser demonstration.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from cramera.laboratory_centrifuge import InvalidCentrifugeOperation
from cramera.laboratory_centrifuge_command import CentrifugeAction, CentrifugeSlot
from cramera.laboratory_manual import CentrifugeManual
from cramera.laboratory_physics_program import LaboratoryProgramPhysics
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_centrifuge_transport import (
    InstrumentTransportField,
    transport_sample,
)
from .test_laboratory_mixing_adapter import program_world
from .test_laboratory_physics_robot import robot_physics


# %% document-driven physical demonstration
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_revised_manual_requires_another_browser_loaded_pair_before_spinning(
    program_world: LaboratoryProgramPhysics,
) -> None:
    """
    Real force-based loading satisfies a revised rule without resetting samples.
    """
    engine = program_world
    centrifuge = engine.physics.centrifuge
    engine.step(1000)
    transport_sample(
        engine,
        LaboratoryBody.CLEAR_TUBE,
        CentrifugeSlot.C1,
        InstrumentTransportField.PAIR,
    )
    engine.centrifuge_command(CentrifugeAction.CLOSE)
    engine.step(5000)
    engine.centrifuge_command(CentrifugeAction.START)
    engine.centrifuge_command(CentrifugeAction.STOP)
    coordinates = engine.physics.data.qpos.copy()
    volumes = {
        key: sample.volume_ml for key, sample in engine.physics.liquid.tubes.items()
    }
    engine.read_centrifuge_manual(CentrifugeManual.builtin("v2").document)
    np.testing.assert_array_equal(engine.physics.data.qpos, coordinates)
    assert {
        key: sample.volume_ml for key, sample in engine.physics.liquid.tubes.items()
    } == volumes
    with pytest.raises(InvalidCentrifugeOperation):
        engine.centrifuge_command(CentrifugeAction.START)
    engine.centrifuge_command(CentrifugeAction.OPEN)
    engine.step(5000)
    transport_sample(
        engine,
        LaboratoryBody.CLEAR_TUBE,
        CentrifugeSlot.C1,
        InstrumentTransportField.PAIR,
    )
    engine.centrifuge_command(CentrifugeAction.CLOSE)
    engine.step(5000)
    engine.centrifuge_command(CentrifugeAction.START, duration=2.0)
    acceleration_time = (
        centrifuge.parameters.default_rpm / centrifuge.parameters.acceleration_rpm
    )
    engine.step(round((acceleration_time + 0.5) / engine.physics.model.opt.timestep))
    assert centrifuge.rpm == pytest.approx(centrifuge.parameters.default_rpm, abs=2.0)
