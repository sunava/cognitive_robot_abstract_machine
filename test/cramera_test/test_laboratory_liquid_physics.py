"""
Liquid follows authoritative contacts and bounded orientation commands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cramera import laboratory_physics_session
from cramera.laboratory_physics import InvalidPhysicsTarget, LaboratoryPhysics
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_physics_session import PhysicsRoute
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics import physics
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_physics_session import ControlledContactSimulation, physics_bundle
from .test_server import get_json, post, server


# %% actual contact dynamics
def test_tilt_uses_bounded_torque_without_pose_write(
    physics: LaboratoryPhysics,
) -> None:
    """
    A raised tube turns toward its desired orientation through physics steps.
    """
    key = LaboratoryBody.CLEAR_TUBE
    position = [0.42, -0.18, 1.19]
    physics.set_target(key, position)
    physics.step(2200)
    before = physics.data.qpos.copy()
    orientation = Rotation.from_euler("y", math.pi / 2).as_quat().tolist()
    physics.set_target(key, position, orientation=orientation)
    np.testing.assert_array_equal(physics.data.qpos, before)
    physics.step(1600)
    pose = physics.snapshot()["objects"][key]
    actual = Rotation.from_quat(pose[3:])
    assert (Rotation.from_quat(orientation) * actual.inv()).magnitude() < 0.15
    torque = physics.data.xfrc_applied[physics.objects[key].body_id, 3:]
    assert np.linalg.norm(torque) <= physics.parameters.maximum_torque


@pytest.mark.parametrize(
    "orientation", [[0, 0, 0, 0], [0, 0, 1], [0, 0, math.nan, 1], [True, 0, 0, 1]]
)
def test_invalid_tilt_preserves_previous_target(
    physics: LaboratoryPhysics, orientation: list[float]
) -> None:
    """
    Malformed orientations cannot partially replace an active force target.
    """
    physics.set_target(LaboratoryBody.CLEAR_TUBE, [0.2, 0, 1.1])
    target = physics.target
    with pytest.raises(InvalidPhysicsTarget):
        physics.set_target(
            LaboratoryBody.CLEAR_TUBE, [0.3, 0, 1.2], orientation=orientation
        )
    assert physics.target is target


def test_liquid_survives_steps_and_resets_with_world(
    physics: LaboratoryPhysics,
) -> None:
    """
    Physical frames own fill volumes, and reset restores the authored liquids.
    """
    key = LaboratoryBody.CLEAR_TUBE
    initial = physics.snapshot()["liquid"]
    physics.fill_liquid(key, 15.0)
    physics.step(500)
    state = physics.snapshot()["liquid"]
    assert state["tubes"][key]["volumeMl"] == pytest.approx(15.0)
    assert state["totalMl"] == pytest.approx(state["initialMl"])
    physics.reset()
    assert physics.snapshot()["liquid"] == initial


def test_physical_tilt_drains_liquid_and_accounts_for_every_drop(
    physics: LaboratoryPhysics,
) -> None:
    """
    MuJoCo-measured tilt causes overflow; a desired tilt alone does not.
    """
    key = LaboratoryBody.CLEAR_TUBE
    physics.fill_liquid(key, 15.0)
    position = [0.42, -0.18, 1.19]
    physics.set_target(key, position)
    physics.step(2200)
    before = physics.snapshot()["liquid"]
    orientation = Rotation.from_euler("y", 2.0).as_quat().tolist()
    physics.set_target(key, position, orientation=orientation)
    assert physics.snapshot()["liquid"] == before
    physics.step(2600)
    state = physics.snapshot()["liquid"]
    assert state["tubes"][key]["volumeMl"] < before["tubes"][key]["volumeMl"] - 1
    assert state["spilledMl"] + state["inFlightMl"] > 1
    assert state["totalMl"] == pytest.approx(state["initialMl"], abs=1e-7)


def test_pr2_transports_user_filled_tube(robot_physics: LaboratoryRobotPhysics) -> None:
    """
    Starting a fresh robot transfer retains the selected fill and carries it to A3.
    """
    world = LaboratoryProgramPhysics(robot_physics, robot_physics.robot_directory)
    key = world.program.reference.object_key
    world.fill_liquid(key, 15.0)
    world.run_program()
    assert world.snapshot()["liquid"]["tubes"][key]["volumeMl"] == 15.0
    with pytest.raises(InvalidPhysicsTarget):
        world.fill_liquid(key, 1.0)
    for _ in range(100):
        world.step(500)
        if world.program.state != ProgramState.RUNNING:
            break
    state = world.snapshot()
    assert state["robot"]["state"] == ProgramState.SUCCEEDED, state["robot"]
    assert state["liquid"]["tubes"][key]["volumeMl"] > 14.9
    assert state["liquid"]["totalMl"] == pytest.approx(
        state["liquid"]["initialMl"], abs=1e-7
    )


# %% HTTP isolation and validation
@dataclass
class ControlledLiquidSimulation(ControlledContactSimulation):
    """
    Observe liquid and orientation forwarding through the real local HTTP API.
    """

    volumes: dict[str, float] = field(default_factory=dict)
    """
    Fill requests accepted for authored tubes.
    """

    def set_target(
        self, key: str, position: list[float], orientation: list[float] | None = None
    ) -> None:
        """
        Retain the complete validated target without generating motion.
        """
        super().set_target(key, position)
        if orientation is not None:
            self.target["orientation"] = orientation

    def fill_liquid(self, key: str, volume_ml: float) -> None:
        """
        Retain bounded fill quantities for the authored tube.
        """
        if (
            key != LaboratoryBody.CLEAR_TUBE
            or not isinstance(volume_ml, (int, float))
            or isinstance(volume_ml, bool)
            or not math.isfinite(volume_ml)
            or not 0 <= volume_ml <= 28
        ):
            raise ValueError("Invalid fill")
        self.volumes[key] = volume_ml

    def snapshot(self) -> dict:
        """
        Expose changed volume in the same authoritative snapshot as body poses.
        """
        return {
            **super().snapshot(),
            "liquid": {
                "tubes": {
                    key: {"volumeMl": volume} for key, volume in self.volumes.items()
                }
            },
        }


@pytest.fixture
def liquid_api(physics_bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace only the engine while retaining session and HTTP validation.
    """
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", ControlledLiquidSimulation
    )


def test_liquid_api_returns_authoritative_fill(liquid_api: None, server: str) -> None:
    """
    A fill command reaches the owning simulation and subsequent state polls.
    """
    assert post(server + PhysicsRoute.START)[0] == 202
    code, answer = post(
        server + PhysicsRoute.LIQUID,
        {"key": LaboratoryBody.CLEAR_TUBE, "volumeMl": 12.5},
    )
    assert code == 200
    assert answer["liquid"]["tubes"][LaboratoryBody.CLEAR_TUBE]["volumeMl"] == 12.5
    assert get_json(server + PhysicsRoute.STATE)["liquid"] == answer["liquid"]


def test_orientation_api_forwards_quaternion(liquid_api: None, server: str) -> None:
    """
    Pose commands may include a finite normalized target rotation.
    """
    post(server + PhysicsRoute.START)
    orientation = Rotation.from_euler("y", 0.7).as_quat().tolist()
    code, answer = post(
        server + PhysicsRoute.TARGET,
        {
            "key": LaboratoryBody.CLEAR_TUBE,
            "position": [0.2, 0, 1.2],
            "orientation": orientation,
        },
    )
    assert code == 200
    np.testing.assert_allclose(answer["target"]["orientation"], orientation)


@pytest.mark.parametrize("volume", [math.nan, -1, 100, True])
def test_invalid_fill_is_rejected_without_mutation(
    liquid_api: None, server: str, volume: float
) -> None:
    """
    Malformed quantities cannot change the liquid state.
    """
    post(server + PhysicsRoute.START)
    code, answer = post(
        server + PhysicsRoute.LIQUID,
        {"key": LaboratoryBody.CLEAR_TUBE, "volumeMl": volume},
    )
    assert code == 400
    assert answer["ok"] is False
    assert get_json(server + PhysicsRoute.STATE)["liquid"]["tubes"] == {}
