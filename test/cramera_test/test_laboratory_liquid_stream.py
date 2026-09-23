"""
Emission timing separates continuous liquid streams from detached drops.
"""

import numpy as np
import pytest

from cramera.laboratory_liquid import LaboratoryLiquid, LiquidField, TubeKey

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_liquid import liquid, liquid_poses, liquid_scene


# %% packet timing
def test_packets_report_emission_time_and_current_ballistic_velocity(
    liquid: LaboratoryLiquid, liquid_poses: dict
) -> None:
    """
    The renderer can distinguish consecutive emission from a pause in outflow.
    """
    liquid_poses[TubeKey.AMBER] = [0, 0, 1.5, 1, 0, 0, 0]
    liquid.reset(liquid_poses)
    duration = 0.01
    liquid.step(duration, liquid_poses)
    liquid.step(duration, liquid_poses)
    snapshot = liquid.snapshot()
    assert snapshot[LiquidField.TIME] == pytest.approx(2 * duration)
    droplets = snapshot[LiquidField.DROPLETS]
    assert droplets[0][LiquidField.EMITTED_AT] == 0
    assert droplets[1][LiquidField.EMITTED_AT] == pytest.approx(duration)
    np.testing.assert_allclose(
        droplets[0][LiquidField.VELOCITY], liquid.droplets[0].velocity
    )
    assert droplets[0][LiquidField.VELOCITY][2] < droplets[1][LiquidField.VELOCITY][2]


def test_reset_clears_the_stream_clock(liquid, liquid_poses) -> None:
    """
    A restarted scene has no timestamps inherited from a previous simulation.
    """
    liquid.step(0.1, liquid_poses)
    liquid.reset(liquid_poses)
    assert liquid.snapshot()[LiquidField.TIME] == 0
