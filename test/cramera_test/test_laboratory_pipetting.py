"""
Conservative dosing and measured receiver interlocks for the bench station.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cramera.laboratory_liquid import LaboratoryLiquid, TubeKey
from cramera.laboratory_pipetting import (
    InvalidPipettingCommand,
    LaboratoryPipetting,
    PipettingAction,
    PipettingCommand,
    PipettingField,
    PipettingSource,
    PipettingState,
)

from .test_laboratory_bundle import laboratory_directory


# %% measured receiver fixture
@dataclass
class AlignedReceiver:
    """
    Expose measured vessel poses independently of commanded targets.
    """

    station: LaboratoryPipetting
    """Pump connected to the real reduced liquid model."""
    poses: dict[str, list[float]]
    """
    Measured loose-body poses supplied to each dosing step.
    """

    def dispense(
        self, volume: float = 1.0, source: str = PipettingSource.AMBER
    ) -> None:
        """
        Request a dose into the aligned clear vessel.
        """
        self.station.command(
            PipettingAction.DISPENSE,
            self.poses,
            key=TubeKey.CLEAR,
            source=source,
            volume_ml=volume,
        )


@pytest.fixture
def receiver(laboratory_directory: Path) -> AlignedReceiver:
    """
    Connect finite reservoirs to an upright vessel beneath the nozzle.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    scene["laboratory"][PipettingField.PIPETTING] = {
        PipettingField.AVAILABLE: True,
        PipettingField.TARGET_POSITION: [-0.10, -0.11, 0.916],
        PipettingField.NOZZLE_POSITION: [-0.10, -0.11, 1.080],
    }
    liquid = LaboratoryLiquid(scene)
    station = LaboratoryPipetting(scene, liquid)
    poses = {item["key"]: item["spawn"].copy() for item in scene["objects"]}
    poses[TubeKey.CLEAR] = [*station.target_position, 0, 0, 0, 1]
    return AlignedReceiver(station, poses)


# %% conservation and simulation time
def test_dose_advances_only_with_simulation_time_and_conserves_volume(receiver):
    """
    Pump delivery removes exactly the amount accepted by the vessel.
    """
    station = receiver.station
    reservoir = station.reservoirs[PipettingSource.AMBER]
    before = reservoir.volume_ml + station.liquid.snapshot()["totalMl"]
    receiver.dispense(1.0)
    assert station.delivered_ml == 0
    station.advance(0.25, receiver.poses)
    assert station.delivered_ml == pytest.approx(station.parameters.flow_rate_ml * 0.25)
    station.advance(1.0, receiver.poses)
    assert station.state == PipettingState.COMPLETE
    assert station.delivered_ml == 1.0
    assert reservoir.volume_ml + station.liquid.snapshot()["totalMl"] == pytest.approx(
        before
    )
    assert station.liquid.snapshot()["totalMl"] == pytest.approx(
        station.liquid.initial_ml
    )


def test_independent_sources_mix_by_delivered_volume(receiver):
    """
    Successive channels yield their measured volume-weighted color.
    """
    station = receiver.station
    receiver.dispense(1.0, PipettingSource.AMBER)
    station.advance(1.0, receiver.poses)
    receiver.dispense(2.0, PipettingSource.TEAL)
    station.advance(1.0, receiver.poses)
    station.advance(1.0, receiver.poses)
    expected = (
        station.reservoirs[PipettingSource.AMBER].color
        + 2 * station.reservoirs[PipettingSource.TEAL].color
    ) / 3
    np.testing.assert_allclose(station.liquid.tubes[TubeKey.CLEAR].color, expected)
    assert station.liquid.tubes[TubeKey.CLEAR].volume_ml == pytest.approx(3)


def test_receiver_motion_blocks_remaining_delivery_without_losing_volume(receiver):
    """
    Moving an active vessel stops the pump before another dose enters it.
    """
    receiver.dispense(1.0)
    receiver.station.advance(0.2, receiver.poses)
    receiver.poses[TubeKey.CLEAR][0] += 0.03
    receiver.station.advance(0.8, receiver.poses)
    assert receiver.station.state == PipettingState.BLOCKED
    assert receiver.station.delivered_ml == pytest.approx(0.2)
    assert receiver.station.remaining_ml == pytest.approx(0.8)
    assert receiver.station.error


def test_stop_preserves_accounting_and_prevents_further_transfer(receiver):
    """
    An explicit stop retains the partial dose while halting flow.
    """
    receiver.dispense()
    receiver.station.advance(0.2, receiver.poses)
    receiver.station.command(PipettingAction.STOP, receiver.poses)
    receiver.station.advance(1.0, receiver.poses)
    assert receiver.station.state == PipettingState.IDLE
    assert receiver.station.delivered_ml == pytest.approx(0.2)


def test_reset_restores_reservoir_inventory_and_idle_state(receiver):
    """
    World reset restores both reagent inventories and dose accounting.
    """
    before = receiver.station.snapshot()
    receiver.dispense()
    receiver.station.advance(1.0, receiver.poses)
    receiver.station.reset()
    assert receiver.station.snapshot() == before


# %% measured interlocks
@pytest.mark.parametrize(
    "obstruction", ["misaligned", "tilted", "capped", "full", "empty", "secured"]
)
def test_invalid_receiver_or_inventory_cannot_start_a_dose(receiver, obstruction):
    """
    Rejected operations leave every volume and current dose unchanged.
    """
    station = receiver.station
    secured = frozenset()
    if obstruction == "misaligned":
        receiver.poses[TubeKey.CLEAR][0] += 0.03
    elif obstruction == "tilted":
        receiver.poses[TubeKey.CLEAR][3:] = (
            Rotation.from_euler("y", 0.3).as_quat().tolist()
        )
    elif obstruction == "capped":
        receiver.poses["stopper"] = [*station.target_position, 0, 0, 0, 1]
        receiver.poses["stopper"][2] += station.liquid.interior.rim - 0.01
    elif obstruction == "full":
        station.liquid.fill(TubeKey.CLEAR, station.liquid.interior.capacity_ml)
    elif obstruction == "empty":
        station.reservoirs[PipettingSource.AMBER].volume_ml = 0
    elif obstruction == "secured":
        secured = frozenset([TubeKey.CLEAR])
    before = station.snapshot()
    liquid_before = station.liquid.snapshot()
    with pytest.raises(InvalidPipettingCommand):
        station.command(
            PipettingAction.DISPENSE,
            receiver.poses,
            secured_keys=secured,
            key=TubeKey.CLEAR,
            source=PipettingSource.AMBER,
            volume_ml=1.0,
        )
    assert station.snapshot() == before
    assert station.liquid.snapshot() == liquid_before


def test_refill_requires_idle_pump_and_restores_only_selected_reservoir(receiver):
    """
    An explicit refill cannot add reagent while the pump is dispensing.
    """
    receiver.dispense()
    receiver.station.advance(0.2, receiver.poses)
    with pytest.raises(InvalidPipettingCommand):
        receiver.station.command(
            PipettingAction.REFILL, receiver.poses, source=PipettingSource.AMBER
        )
    receiver.station.command(PipettingAction.STOP, receiver.poses)
    receiver.station.command(
        PipettingAction.REFILL, receiver.poses, source=PipettingSource.AMBER
    )
    assert (
        receiver.station.reservoirs[PipettingSource.AMBER].volume_ml
        == receiver.station.parameters.reservoir_ml
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "action": "dispense",
            "key": "tube_clear",
            "source": "amber",
            "volumeMl": True,
        },
        {
            "action": "dispense",
            "key": "tube_clear",
            "source": "amber",
            "volumeMl": 0.01,
        },
        {"action": "dispense", "key": "tube_clear", "source": "amber", "volumeMl": 6},
        {
            "action": "dispense",
            "key": "tube_clear",
            "source": "amber",
            "volumeMl": float("nan"),
        },
        {"action": "dispense", "key": "tube_clear", "source": "unknown", "volumeMl": 1},
        {"action": "stop", "position": [0, 0, 0]},
        {"action": "refill"},
    ],
)
def test_malformed_commands_are_rejected(payload):
    """
    Public commands exclude nonfinite doses and arbitrary pose writes.
    """
    with pytest.raises(InvalidPipettingCommand):
        PipettingCommand.from_payload(payload)


def test_resume_completes_a_residual_smaller_than_minimum_new_dose(receiver):
    """
    Pausing near completion preserves the exact remainder without rounding it up.
    """
    receiver.dispense()
    receiver.station.advance(0.99, receiver.poses)
    receiver.station.command(PipettingAction.STOP, receiver.poses)
    receiver.station.command(PipettingAction.RESUME, receiver.poses)
    receiver.station.advance(0.1, receiver.poses)
    assert receiver.station.state == PipettingState.COMPLETE
    assert receiver.station.delivered_ml == pytest.approx(1.0)
    assert receiver.station.liquid.tubes[TubeKey.CLEAR].volume_ml == pytest.approx(1.0)


def test_reservoir_snapshot_declares_the_capacity_used_for_refill(receiver):
    """
    The renderer can scale reservoir fill against the actual pump capacity.
    """
    for reservoir in receiver.station.snapshot()[PipettingField.RESERVOIRS]:
        assert reservoir["capacityMl"] == receiver.station.parameters.reservoir_ml
