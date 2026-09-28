"""
The workshop demonstration validates its complete assembly outcome.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from coraplex.datastructures.dataclasses import Context
from cramera.live.bridge import Bridge
from cramera.workshop_assembly_demo import (
    AssemblyObservation,
    AssemblyStage,
    WorkshopAssemblyDemo,
    WorkshopAssemblyResult,
)


# %% complete result gate
def test_assembly_needs_both_hand_release_and_seated_screw():
    """
    A finished trajectory cannot be published with incomplete tool operation.
    """
    result = WorkshopAssemblyResult(
        action_statuses=["SUCCEEDED"] * len(AssemblyStage),
        box_position_error=0.0001,
        screw_depth=0.11,
        required_depth=0.11,
        driver_released=True,
        support_released=False,
        minimum_collision_distance=0.0,
        frames=1000,
    )
    assert result.success is False
    result.support_released = True
    assert result.success is True
    result.screw_depth = 0.01
    assert result.success is False


@pytest.mark.parametrize("minimum", [math.inf, -math.inf, math.nan, -0.0004])
def test_failure_diagnostics_create_parent_and_write_finite_json(
    world_with_two_bodies, tmp_path: Path, minimum: float
) -> None:
    """
    An early failure writes readable diagnostics even before an output exists.
    """
    world, _, _ = world_with_two_bodies
    output = tmp_path / "new-parent" / "assembly"
    observation = AssemblyObservation(
        context=Context(world=world, robot=None),
        actions=[],
        labels=[],
        bridge=Bridge(),
        minimum_distance=minimum,
    )
    WorkshopAssemblyDemo(output_directory=output).write_failure(observation, world)
    payload = json.loads(output.with_suffix(".failure.json").read_text())
    assert payload["minimumCollisionDistance"] == (
        minimum if math.isfinite(minimum) else None
    )
    json.dumps(payload, allow_nan=False)


def test_collision_or_failed_action_blocks_assembly_publication():
    """
    A reachable final goal does not hide a failed intermediate action.
    """
    result = WorkshopAssemblyResult(
        action_statuses=["SUCCEEDED"] * len(AssemblyStage),
        box_position_error=0.0,
        screw_depth=0.11,
        required_depth=0.11,
        driver_released=True,
        support_released=True,
        minimum_collision_distance=-0.02,
        frames=1000,
    )
    assert result.success is False
    result.minimum_collision_distance = 0.0
    result.action_statuses[0] = "FAILED"
    assert result.success is False
