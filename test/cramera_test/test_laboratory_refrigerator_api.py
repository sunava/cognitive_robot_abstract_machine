"""
Refrigerator commands retain authentication, ownership and session isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing_extensions import Any
from http import HTTPStatus
import json

import pytest

from cramera import laboratory_physics_session, laboratory_robot_session
from cramera.laboratory_refrigerator_command import (
    InvalidRefrigeratorCommand,
    RefrigeratorAction,
    RefrigeratorCommand,
    RefrigeratorField,
)
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


# %% observable command forwarding
@dataclass
class RecordedDoorCommands(ControlledRobotSimulation):
    """
    Record accepted door operations without a scientific engine dependency.
    """

    commands: list[RefrigeratorAction] = field(default_factory=list)
    """
    Door requests that reached the selected physical session.
    """

    def refrigerator_command(self, action: RefrigeratorAction) -> None:
        """
        Record one validated cabinet operation.
        """
        self.commands.append(action)

    def snapshot(self) -> dict:
        """
        Expose accepted requests beside the existing robot ownership state.
        """
        return {
            **super().snapshot(),
            RefrigeratorField.REFRIGERATOR: len(self.commands),
        }


@pytest.fixture
def refrigerator_api(
    physics_bundle: Path, robot_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Keep session locks and HTTP validation around observable door requests.
    """
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", RecordedDoorCommands
    )
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", RecordedDoorCommands
    )


@pytest.fixture(params=[PhysicsRoute, RobotPhysicsRoute])
def refrigerator_routes(
    request: pytest.FixtureRequest,
) -> type[PhysicsRoute] | type[RobotPhysicsRoute]:
    """
    Exercise both independent physics sessions.
    """
    return request.param


# %% command shape and API boundary
@pytest.mark.parametrize("action", list(RefrigeratorAction))
def test_valid_command_isolated_to_selected_session(
    refrigerator_api: None,
    refrigerator_routes: type[PhysicsRoute] | type[RobotPhysicsRoute],
    server: str,
    action: RefrigeratorAction,
) -> None:
    """
    Manual and robot sessions receive independent physical door operations.
    """
    for routes in (PhysicsRoute, RobotPhysicsRoute):
        assert post(server + routes.START)[0] == HTTPStatus.ACCEPTED
    code, result = post(
        server + refrigerator_routes.REFRIGERATOR, {RefrigeratorField.ACTION: action}
    )
    assert code == HTTPStatus.OK
    assert result[RefrigeratorField.REFRIGERATOR] == 1
    other = RobotPhysicsRoute if refrigerator_routes is PhysicsRoute else PhysicsRoute
    assert get_json(server + other.STATE)[RefrigeratorField.REFRIGERATOR] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"action": "freeze"},
        {"action": True},
        {"action": []},
        {"action": "open", "position": [0, 0, 0]},
        None,
    ],
)
def test_command_rejects_unknown_operations_and_pose_injection(
    payload: dict[str, Any] | None,
) -> None:
    """
    Malformed door requests never carry unvalidated pose overrides to the engine.
    """
    with pytest.raises(InvalidRefrigeratorCommand):
        RefrigeratorCommand.from_payload(payload)


def test_command_requires_running_session(
    refrigerator_api: None,
    refrigerator_routes: type[PhysicsRoute] | type[RobotPhysicsRoute],
    server: str,
) -> None:
    """
    A cabinet request cannot implicitly construct or start a physical session.
    """
    assert (
        post(server + refrigerator_routes.REFRIGERATOR, {"action": "open"})[0]
        == HTTPStatus.CONFLICT
    )


def test_robot_owns_door_operations_during_execution(
    refrigerator_api: None, server: str
) -> None:
    """
    Door controls remain blocked until the active robot program yields ownership.
    """
    post(server + RobotPhysicsRoute.START)
    post(server + RobotPhysicsRoute.RUN)
    assert (
        post(server + RobotPhysicsRoute.REFRIGERATOR, {"action": "open"})[0]
        == HTTPStatus.BAD_REQUEST
    )
    assert (
        get_json(server + RobotPhysicsRoute.STATE)[RefrigeratorField.REFRIGERATOR] == 0
    )
    post(server + RobotPhysicsRoute.PAUSE)
    assert (
        post(server + RobotPhysicsRoute.REFRIGERATOR, {"action": "open"})[0]
        == HTTPStatus.OK
    )


@pytest.mark.parametrize(
    "session_type", [LaboratoryPhysicsSession, LaboratoryRobotSession]
)
def test_generated_scene_retains_refrigerator_metadata(
    refrigerator_api: None,
    physics_bundle: Path,
    session_type: type[LaboratoryPhysicsSession],
) -> None:
    """
    Live viewer controls inherit the authored cabinet geometry and operating metadata.
    """
    scene_path = physics_bundle / PhysicsFile.SCENE
    authored = json.loads(scene_path.read_text())
    metadata = {RefrigeratorField.AVAILABLE: True}
    authored[PhysicsField.LABORATORY][RefrigeratorField.REFRIGERATOR] = metadata
    scene_path.write_text(json.dumps(authored))
    session = session_type(physics_bundle.parents[1])
    try:
        session.start()
        copied = json.loads((session.output_directory / PhysicsFile.SCENE).read_text())
        assert copied[PhysicsField.PHYSICS][RefrigeratorField.REFRIGERATOR] == metadata
    finally:
        session.stop()


def test_refrigerator_shelf_target_is_inside_session_bounds(
    refrigerator_api: None,
    refrigerator_routes: type[PhysicsRoute] | type[RobotPhysicsRoute],
    server: str,
) -> None:
    """
    Force targets can reach the low cabinet beside the existing bench.
    """
    post(server + refrigerator_routes.START)
    position = [-1.23, -0.1, 0.24]
    code, result = post(
        server + refrigerator_routes.TARGET, {"key": "tube_clear", "position": position}
    )
    assert code == HTTPStatus.OK
    assert result["target"]["position"] == position


@pytest.mark.parametrize("sharing_prefix", ["", "/laboratory"])
def test_shared_refrigerator_requires_password(
    shared_laboratory: SharedLaboratory,
) -> None:
    """
    The established remote gateway authenticates the new cabinet route.
    """
    route = shared_laboratory.options.public_path(RobotPhysicsRoute.REFRIGERATOR)
    body = json.dumps({RefrigeratorField.ACTION: RefrigeratorAction.OPEN}).encode()
    headers = {"Origin": shared_laboratory.origin, "Content-Type": "application/json"}
    assert (
        shared_laboratory.request(
            route, method="POST", body=body, headers=headers
        ).status
        == HTTPStatus.UNAUTHORIZED
    )
    assert shared_laboratory.backend.requests == []
    shared_laboratory.login()
    assert (
        shared_laboratory.request(
            route, method="POST", body=body, headers=headers
        ).status
        == HTTPStatus.OK
    )
    assert shared_laboratory.backend.requests[-1].body == body


@pytest.mark.parametrize(
    "session_type", [LaboratoryPhysicsSession, LaboratoryRobotSession]
)
def test_storage_retrieval_preserves_authored_rack_destinations(
    refrigerator_api: None,
    physics_bundle: Path,
    session_type: type[LaboratoryPhysicsSession],
) -> None:
    """
    Retrieval controls receive the same supported rack poses as the source scene.
    """
    scene_path = physics_bundle / PhysicsFile.SCENE
    authored = json.loads(scene_path.read_text())
    slots = [{RefrigeratorField.ID: "A1", "pose": [0.16, -0.065, 0.906, 0, 0, 0, 1]}]
    authored[PhysicsField.LABORATORY][PhysicsField.RACK_SLOTS] = slots
    scene_path.write_text(json.dumps(authored))
    session = session_type(physics_bundle.parents[1])
    try:
        session.start()
        copied = json.loads((session.output_directory / PhysicsFile.SCENE).read_text())
        assert copied[PhysicsField.PHYSICS][PhysicsField.RACK_SLOTS] == slots
    finally:
        session.stop()
