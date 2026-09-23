"""
Scale zeroing remains serialized, authenticated, and isolated per simulation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus

import pytest

from cramera import laboratory_physics_session, laboratory_robot_session
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_physics_session import (
    LaboratoryPhysicsSession,
    PhysicsField,
    PhysicsFile,
    PhysicsRoute,
)
from cramera.laboratory_robot_session import LaboratoryRobotSession, RobotPhysicsRoute

from .test_laboratory_physics_session import physics_bundle
from .test_laboratory_robot_session import ControlledRobotSimulation, robot_bundle
from .test_laboratory_share import SharedLaboratory, shared_laboratory, sharing_prefix
from .test_server import get_json, post, server


# %% observable scale owner
@dataclass
class CountedScaleSimulation(ControlledRobotSimulation):
    """
    Record successful zeroing without importing an articulated model.
    """

    tare_count: int = 0
    """
    Number of accepted zeroing operations.
    """

    scale_available: bool = True
    """
    Whether this simulated scene contains a scale.
    """

    def tare_scale(self) -> None:
        """
        Reject an absent instrument before recording the zero operation.
        """
        if not self.scale_available:
            raise InvalidPhysicsTarget("This scene contains no laboratory scale")
        self.tare_count += 1

    def snapshot(self) -> dict:
        """
        Expose zeroing count in the scale portion of the engine snapshot.
        """
        return {**super().snapshot(), PhysicsField.SCALE: self.tare_count}


@pytest.fixture
def scale_api(physics_bundle, robot_bundle, monkeypatch):
    """
    Use observable zeroing in both independent server-owned simulations.
    """
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", CountedScaleSimulation
    )
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", CountedScaleSimulation
    )


@pytest.fixture(params=[PhysicsRoute, RobotPhysicsRoute])
def scale_routes(request):
    """
    Exercise zeroing through each public contact simulation contract.
    """
    return request.param


# %% local scale commands
def test_scale_zeroing_isolated_to_selected_session(scale_api, scale_routes, server):
    """
    Zeroing one simulation leaves the other scale untouched.
    """
    for routes in (PhysicsRoute, RobotPhysicsRoute):
        assert post(server + routes.START)[0] == HTTPStatus.ACCEPTED
    code, result = post(server + scale_routes.SCALE_TARE)
    assert code == HTTPStatus.OK
    assert result[PhysicsField.SCALE] == 1
    other = RobotPhysicsRoute if scale_routes is PhysicsRoute else PhysicsRoute
    assert get_json(server + other.STATE)[PhysicsField.SCALE] == 0


@pytest.mark.parametrize("body", [{"grams": 12}, [1]])
def test_scale_rejects_parameters_without_zeroing(
    scale_api, scale_routes, server, body
):
    """
    The browser can request tare but cannot supply a fabricated reading.
    """
    post(server + scale_routes.START)
    code, result = post(server + scale_routes.SCALE_TARE, body)
    assert code == HTTPStatus.BAD_REQUEST
    assert result[PhysicsField.OK] is False
    assert get_json(server + scale_routes.STATE)[PhysicsField.SCALE] == 0


def test_scale_zeroing_requires_a_live_session(scale_api, scale_routes, server):
    """
    Tare never starts an unavailable simulation implicitly.
    """
    code, result = post(server + scale_routes.SCALE_TARE)
    assert code == HTTPStatus.CONFLICT
    assert result[PhysicsField.OK] is False


def test_robot_scale_zeroing_requires_paused_program(scale_api, server):
    """
    A running robot program must pause before the scale reference changes.
    """
    post(server + RobotPhysicsRoute.START)
    post(server + RobotPhysicsRoute.RUN)
    code, result = post(server + RobotPhysicsRoute.SCALE_TARE)
    assert code == HTTPStatus.BAD_REQUEST
    assert result[PhysicsField.OK] is False
    assert get_json(server + RobotPhysicsRoute.STATE)[PhysicsField.SCALE] == 0
    post(server + RobotPhysicsRoute.PAUSE)
    code, result = post(server + RobotPhysicsRoute.SCALE_TARE)
    assert code == HTTPStatus.OK
    assert result[PhysicsField.SCALE] == 1


@pytest.mark.parametrize(
    "session_type", [LaboratoryPhysicsSession, LaboratoryRobotSession]
)
def test_scene_copy_retains_scale_metadata(scale_api, physics_bundle, session_type):
    """
    Both live scenes retain the authored instrument placement information.
    """
    scene_path = physics_bundle / PhysicsFile.SCENE
    authored = json.loads(scene_path.read_text())
    authored[PhysicsField.LABORATORY][PhysicsField.SCALE] = {
        PhysicsField.KEY: "analytical_balance"
    }
    scene_path.write_text(json.dumps(authored))
    session = session_type(physics_bundle.parents[1])
    try:
        assert session.start()[PhysicsField.OK] is True
        copied = json.loads((session.output_directory / PhysicsFile.SCENE).read_text())
        assert (
            copied[PhysicsField.PHYSICS][PhysicsField.SCALE]
            == authored[PhysicsField.LABORATORY][PhysicsField.SCALE]
        )
    finally:
        session.stop()


def test_missing_scale_does_not_record_zeroing(scale_api, physics_bundle):
    """
    An unavailable instrument rejects tare without changing its reference.
    """
    session = LaboratoryPhysicsSession(physics_bundle.parents[1])
    try:
        session.start()
        session.physics.scale_available = False
        with pytest.raises(InvalidPhysicsTarget):
            session.tare_scale()
        assert session.physics.tare_count == 0
    finally:
        session.stop()


def test_missing_scale_returns_request_error(
    scale_api, scale_routes, server, monkeypatch
):
    """
    An absent instrument produces a bounded API error instead of a fake reading.
    """
    factory = partial(CountedScaleSimulation, scale_available=False)
    monkeypatch.setattr(laboratory_physics_session, "create_physics", factory)
    monkeypatch.setattr(laboratory_robot_session, "create_robot_physics", factory)
    post(server + scale_routes.START)
    code, result = post(server + scale_routes.SCALE_TARE)
    assert code == HTTPStatus.BAD_REQUEST
    assert result[PhysicsField.OK] is False
    assert get_json(server + scale_routes.STATE)[PhysicsField.SCALE] == 0


# %% program adapter ownership
@dataclass
class ScaleProgramState:
    """
    Select whether the robot currently owns the simulated workspace.
    """

    state: ProgramState
    """
    Current program execution state.
    """


@dataclass
class ControlledScaleAdapter(LaboratoryProgramPhysics):
    """
    Exercise the adapter's ownership rule without loading robot geometry.
    """

    def __post_init__(self) -> None:
        """
        Start with an idle controller whose state is directly observable.
        """
        self.program = ScaleProgramState(ProgramState.IDLE)


@pytest.mark.parametrize("state", [ProgramState.IDLE, ProgramState.PAUSED])
def test_program_adapter_delegates_scale_zeroing(state, tmp_path):
    """
    A stopped controller allows zeroing the owned physical instrument.
    """
    physics = CountedScaleSimulation(tmp_path)
    adapter = ControlledScaleAdapter(physics, tmp_path)
    adapter.program.state = state
    adapter.tare_scale()
    assert physics.tare_count == 1


def test_program_adapter_blocks_scale_zeroing_while_running(tmp_path):
    """
    Direct adapter access cannot bypass an active program's ownership.
    """
    physics = CountedScaleSimulation(tmp_path)
    adapter = ControlledScaleAdapter(physics, tmp_path)
    adapter.program.state = ProgramState.RUNNING
    with pytest.raises(InvalidPhysicsTarget):
        adapter.tare_scale()
    assert physics.tare_count == 0


# %% protected remote control
@pytest.mark.parametrize("sharing_prefix", ["", "/laboratory"])
def test_shared_scale_zeroing_requires_login_and_keeps_backend_route(
    shared_laboratory: SharedLaboratory,
):
    """
    Tare uses the existing password and same-origin gateway contract.
    """
    route = shared_laboratory.options.public_path(RobotPhysicsRoute.SCALE_TARE)
    headers = {"Origin": shared_laboratory.origin, "Content-Type": "application/json"}
    denied = shared_laboratory.request(
        route, method="POST", body=b"{}", headers=headers
    )
    assert denied.status == HTTPStatus.UNAUTHORIZED
    assert shared_laboratory.backend.requests == []
    shared_laboratory.login()
    accepted = shared_laboratory.request(
        route, method="POST", body=b"{}", headers=headers
    )
    assert accepted.status == HTTPStatus.OK
    forwarded = shared_laboratory.backend.requests[-1]
    assert forwarded.path == RobotPhysicsRoute.SCALE_TARE
    assert forwarded.body == b"{}"
    assert forwarded.cookie is None
