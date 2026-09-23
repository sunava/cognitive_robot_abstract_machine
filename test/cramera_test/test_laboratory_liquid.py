"""
Conservative free surfaces and ballistic transfer between laboratory tubes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cramera.laboratory_liquid import (
    InvalidLiquidInput,
    LaboratoryLiquid,
    LiquidField,
    LiquidModel,
    TubeInterior,
    TubeKey,
)

from .test_laboratory_bundle import laboratory_directory


# %% authored laboratory fixtures
@pytest.fixture
def liquid_scene(laboratory_directory: Path) -> dict:
    """
    Read the authored tube layout without requiring the renderer or MuJoCo.
    """
    return json.loads((laboratory_directory / "scene.json").read_text())


@pytest.fixture
def liquid_poses(liquid_scene: dict) -> dict[str, list[float]]:
    """
    Copy independent rigid object poses from the scene fixture.
    """
    return {item["key"]: list(item["spawn"]) for item in liquid_scene["objects"]}


@pytest.fixture
def liquid(liquid_scene: dict) -> LaboratoryLiquid:
    """
    Create deterministic authored volumes in all three tube interiors.
    """
    return LaboratoryLiquid(liquid_scene)


class TestHydrostaticVolume:
    """
    Free surfaces retain volume and respond to the actual tube orientation.
    """

    def test_authored_upright_levels_are_preserved(
        self, liquid: LaboratoryLiquid, liquid_poses: dict
    ) -> None:
        before = liquid.snapshot()
        for _ in range(100):
            liquid.step(0.02, liquid_poses)
        after = liquid.snapshot()
        assert after[LiquidField.MODEL] == LiquidModel.REDUCED_FREE_SURFACE
        assert after[LiquidField.TOTAL] == pytest.approx(before[LiquidField.TOTAL])
        assert after[LiquidField.SPILLED] == 0
        assert after[LiquidField.IN_FLIGHT] == 0
        assert after[LiquidField.TUBES][TubeKey.CLEAR][LiquidField.VOLUME] == 0
        assert after[LiquidField.TUBES][TubeKey.AMBER][
            LiquidField.OFFSET
        ] == pytest.approx(0.074, abs=1e-6)
        assert after[LiquidField.TUBES][TubeKey.TEAL][
            LiquidField.OFFSET
        ] == pytest.approx(0.095, abs=1e-6)

    def test_round_bottom_capacity_matches_sphere_and_cylinder(self) -> None:
        interior = TubeInterior()
        expected = (
            np.pi * interior.radius**2 * (interior.rim - interior.center)
            + 2 * np.pi * interior.radius**3 / 3
        )
        assert interior.capacity_ml == pytest.approx(expected * 1e6, rel=1e-12)

    def test_gravity_normal_stays_vertical_in_world_when_tube_is_tilted(
        self, liquid: LaboratoryLiquid, liquid_poses: dict
    ) -> None:
        angle = np.deg2rad(35)
        liquid_poses[TubeKey.AMBER][3:] = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        liquid.reset(liquid_poses)
        normal = liquid.snapshot()[LiquidField.TUBES][TubeKey.AMBER][LiquidField.NORMAL]
        np.testing.assert_allclose(
            normal, [-np.sin(angle), 0, np.cos(angle)], atol=1e-12
        )

    def test_acceleration_excites_slosh_and_rest_allows_it_to_decay(
        self, liquid: LaboratoryLiquid, liquid_poses: dict
    ) -> None:
        liquid_poses[TubeKey.AMBER][0] += 0.004
        liquid.step(0.02, liquid_poses)
        normal = liquid.snapshot()[LiquidField.TUBES][TubeKey.AMBER][LiquidField.NORMAL]
        assert abs(normal[0]) > 1e-4
        for _ in range(200):
            liquid.step(0.02, liquid_poses)
        normal = liquid.snapshot()[LiquidField.TUBES][TubeKey.AMBER][LiquidField.NORMAL]
        np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-3)


class TestConservativePouring:
    """
    Spilling, flight and receiving account for every milliliter.
    """

    def test_flying_packets_expose_their_source_for_stream_rendering(
        self, liquid: LaboratoryLiquid, liquid_poses: dict
    ) -> None:
        liquid_poses[TubeKey.AMBER] = [0.0, 0.0, 1.5, 1, 0, 0, 0]
        liquid.reset(liquid_poses)
        liquid.step(0.02, liquid_poses)
        droplets = liquid.snapshot()[LiquidField.DROPLETS]
        assert droplets
        assert droplets[0]["source"] == TubeKey.AMBER

    def test_inverted_tube_drains_into_puddle_without_losing_volume(
        self, liquid: LaboratoryLiquid, liquid_poses: dict
    ) -> None:
        liquid_poses[TubeKey.AMBER] = [0.0, 0.0, 1.5, 1, 0, 0, 0]
        liquid.reset(liquid_poses)
        before = liquid.snapshot()
        initial_amber = before[LiquidField.TUBES][TubeKey.AMBER][LiquidField.VOLUME]
        for _ in range(150):
            liquid.step(0.02, liquid_poses)
            snapshot = liquid.snapshot()
            contained = sum(
                item[LiquidField.VOLUME]
                for item in snapshot[LiquidField.TUBES].values()
            )
            assert contained + snapshot[LiquidField.IN_FLIGHT] + snapshot[
                LiquidField.SPILLED
            ] == pytest.approx(before[LiquidField.INITIAL], abs=1e-9)
        assert snapshot[LiquidField.TUBES][TubeKey.AMBER][LiquidField.VOLUME] < 1e-4
        assert snapshot[LiquidField.SPILLED] == pytest.approx(initial_amber, abs=1e-4)
        assert snapshot[LiquidField.PUDDLES]
        assert snapshot[LiquidField.TOTAL] == pytest.approx(before[LiquidField.INITIAL])

    def test_lower_tube_receives_ballistic_outflow_through_open_rim(
        self, liquid: LaboratoryLiquid, liquid_poses: dict
    ) -> None:
        liquid_poses[TubeKey.AMBER] = [0.0, 0.0, 1.55, 1, 0, 0, 0]
        liquid_poses[TubeKey.CLEAR] = [0.0, 0.0, 1.05, 0, 0, 0, 1]
        liquid.reset(liquid_poses)
        before = liquid.snapshot()
        source = before[LiquidField.TUBES][TubeKey.AMBER]
        for _ in range(150):
            liquid.step(0.02, liquid_poses)
        after = liquid.snapshot()
        receiver = after[LiquidField.TUBES][TubeKey.CLEAR]
        assert receiver[LiquidField.VOLUME] == pytest.approx(
            source[LiquidField.VOLUME], abs=1e-4
        )
        np.testing.assert_allclose(
            receiver[LiquidField.COLOR], source[LiquidField.COLOR]
        )
        assert after[LiquidField.SPILLED] == 0
        assert after[LiquidField.IN_FLIGHT] == 0

    def test_full_receiver_cannot_create_volume(
        self, liquid: LaboratoryLiquid, liquid_poses: dict
    ) -> None:
        liquid_poses[TubeKey.AMBER] = [0.0, 0.0, 1.55, 1, 0, 0, 0]
        liquid_poses[TubeKey.CLEAR] = [0.0, 0.0, 1.05, 0, 0, 0, 1]
        liquid.reset(liquid_poses)
        capacity = liquid.snapshot()[LiquidField.TUBES][TubeKey.CLEAR][
            LiquidField.CAPACITY
        ]
        liquid.fill(TubeKey.CLEAR, capacity)
        initial = liquid.snapshot()[LiquidField.INITIAL]
        for _ in range(150):
            liquid.step(0.02, liquid_poses)
        snapshot = liquid.snapshot()
        assert (
            snapshot[LiquidField.TUBES][TubeKey.CLEAR][LiquidField.VOLUME] <= capacity
        )
        assert snapshot[LiquidField.TOTAL] == pytest.approx(initial, abs=1e-9)
        assert snapshot[LiquidField.SPILLED] > 0


class TestLiquidInputContract:
    """
    Invalid commands are transactional and reset is repeatable.
    """

    @pytest.mark.parametrize(
        "color",
        [
            [float("nan"), 0, 0],
            [float("inf"), 0, 0],
            [True, 0, 0],
            [-0.1, 0, 0],
            [1.1, 0, 0],
            [0, 0],
            ["0", 0, 0],
            "red",
        ],
    )
    def test_invalid_fill_color_does_not_mutate_volume_or_tint(
        self, liquid: LaboratoryLiquid, color: object
    ) -> None:
        before = liquid.snapshot()
        with pytest.raises(InvalidLiquidInput):
            liquid.fill(TubeKey.CLEAR, 10, color=color)
        assert liquid.snapshot() == before

    def test_fill_restores_volume_and_mixed_color(
        self, liquid: LaboratoryLiquid
    ) -> None:
        before = liquid.snapshot()
        color = [0.4, 0.5, 0.6]
        liquid.fill(TubeKey.CLEAR, 10, color=color)
        snapshot = liquid.snapshot()
        assert snapshot[LiquidField.TUBES][TubeKey.CLEAR][LiquidField.COLOR] == color
        assert snapshot[LiquidField.TUBES][TubeKey.CLEAR][LiquidField.VOLUME] == 10
        assert snapshot[LiquidField.INITIAL] == pytest.approx(
            before[LiquidField.INITIAL] + 10
        )

    @pytest.mark.parametrize("volume", [-1, float("nan"), float("inf"), True, "1", 100])
    def test_invalid_fill_does_not_mutate_state(
        self, liquid: LaboratoryLiquid, volume: object
    ) -> None:
        before = liquid.snapshot()
        with pytest.raises(InvalidLiquidInput):
            liquid.fill(TubeKey.CLEAR, volume)
        assert liquid.snapshot() == before

    def test_unknown_tube_does_not_mutate_state(self, liquid: LaboratoryLiquid) -> None:
        before = liquid.snapshot()
        with pytest.raises(InvalidLiquidInput):
            liquid.fill("stopper", 1)
        assert liquid.snapshot() == before

    def test_explicit_fill_updates_source_accounting(
        self, liquid: LaboratoryLiquid
    ) -> None:
        before = liquid.snapshot()
        liquid.fill(TubeKey.CLEAR, 10)
        after = liquid.snapshot()
        assert after[LiquidField.INITIAL] == pytest.approx(
            before[LiquidField.INITIAL] + 10
        )
        assert after[LiquidField.TOTAL] == pytest.approx(after[LiquidField.INITIAL])

    def test_reset_restores_authored_volume_and_removes_spill(
        self, liquid: LaboratoryLiquid, liquid_poses: dict
    ) -> None:
        before = liquid.snapshot()
        liquid.fill(TubeKey.CLEAR, 10)
        displaced = copy.deepcopy(liquid_poses)
        displaced[TubeKey.AMBER] = [0.0, 0.0, 1.5, 1, 0, 0, 0]
        for _ in range(100):
            liquid.step(0.02, displaced)
        liquid.reset(liquid_poses)
        assert liquid.snapshot() == before

    @pytest.mark.parametrize("duration", [-1, float("nan"), float("inf"), True])
    def test_invalid_duration_does_not_mutate_state(
        self, liquid: LaboratoryLiquid, liquid_poses: dict, duration: float
    ) -> None:
        before = liquid.snapshot()
        with pytest.raises(InvalidLiquidInput):
            liquid.step(duration, liquid_poses)
        assert liquid.snapshot() == before

    def test_invalid_pose_does_not_mutate_state(
        self, liquid: LaboratoryLiquid, liquid_poses: dict
    ) -> None:
        liquid_poses[TubeKey.TEAL][0] = float("nan")
        before = liquid.snapshot()
        with pytest.raises(InvalidLiquidInput):
            liquid.step(0.02, liquid_poses)
        assert liquid.snapshot() == before
