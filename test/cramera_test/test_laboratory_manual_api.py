"""
Manual loading shares session ownership, validation and the password gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from http import HTTPStatus

import pytest

from cramera import laboratory_physics_session, laboratory_robot_session
from cramera.laboratory_centrifuge_command import CentrifugeField, ManualRequestField
from cramera.laboratory_manual import CentrifugeManual, ManualKnowledge
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_session import PhysicsField, PhysicsRoute
from cramera.laboratory_robot_session import RobotPhysicsRoute

from .test_laboratory_physics_session import physics_bundle
from .test_laboratory_robot_session import ControlledRobotSimulation, robot_bundle
from .test_laboratory_scale_api import ControlledScaleAdapter
from .test_laboratory_share import SharedLaboratory, shared_laboratory, sharing_prefix
from .test_server import get_json, post, server


# %% serialized knowledge updates
@dataclass
class RecordedManualReadings(ControlledRobotSimulation):
    """
    Expose knowledge updates independently of simulation progress.
    """

    knowledge: ManualKnowledge = field(default_factory=ManualKnowledge)
    """
    The active document and its accepted revision history.
    """

    def read_centrifuge_manual(self, document: str) -> None:
        """
        Validate a document without changing any simulated object.
        """
        self.knowledge.read(document)

    def snapshot(self) -> dict:
        """
        Include knowledge beside observable object and robot state.
        """
        return {
            **super().snapshot(),
            CentrifugeField.CENTRIFUGE: {
                CentrifugeField.KNOWLEDGE: self.knowledge.snapshot()
            },
        }


@pytest.fixture
def manual_api(physics_bundle, robot_bundle, monkeypatch):
    """
    Provide independently owned knowledge for the two simulation routes.
    """
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", RecordedManualReadings
    )
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", RecordedManualReadings
    )


@pytest.fixture(params=[PhysicsRoute, RobotPhysicsRoute])
def manual_routes(request):
    """
    Exercise document updates through both public session variants.
    """
    return request.param


def knowledge(snapshot: dict) -> dict:
    """
    Select document knowledge from the complete session snapshot.
    """
    return snapshot[CentrifugeField.CENTRIFUGE][CentrifugeField.KNOWLEDGE]


@pytest.mark.parametrize(
    "field_name", [ManualRequestField.EDITION, ManualRequestField.DOCUMENT]
)
def test_manual_request_updates_only_selected_session(
    manual_api, manual_routes, server, field_name
):
    """
    A built-in or uploaded document affects only the addressed laboratory.
    """
    for route in (PhysicsRoute, RobotPhysicsRoute):
        assert post(server + route.START)[0] == HTTPStatus.ACCEPTED
    other = RobotPhysicsRoute if manual_routes is PhysicsRoute else PhysicsRoute
    untouched = knowledge(get_json(server + other.STATE))
    before = get_json(server + manual_routes.STATE)
    document = CentrifugeManual.builtin("v2")
    value = "v2" if field_name == ManualRequestField.EDITION else document.document
    code, result = post(server + manual_routes.MANUAL, {field_name: value})
    assert code == HTTPStatus.OK
    assert result[PhysicsField.OBJECTS] == before[PhysicsField.OBJECTS]
    assert knowledge(result)["manual"] == document.snapshot()
    assert knowledge(get_json(server + other.STATE)) == untouched


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"edition": "v3"},
        {"edition": "v2", "document": ""},
        {"edition": "v2", "position": [0, 0, 0]},
        {"document": ""},
        {"document": 3},
    ],
)
def test_invalid_manual_request_preserves_current_knowledge(
    manual_api, manual_routes, server, body
):
    """
    Malformed and unsupported documents cannot mutate accepted knowledge.
    """
    post(server + manual_routes.START)
    before = knowledge(get_json(server + manual_routes.STATE))
    code, result = post(server + manual_routes.MANUAL, body)
    assert code == HTTPStatus.BAD_REQUEST
    assert result[PhysicsField.OK] is False
    assert knowledge(get_json(server + manual_routes.STATE)) == before


def test_manual_requires_live_session(manual_api, manual_routes, server):
    """
    Reading a document cannot implicitly start or reset a simulation.
    """
    assert (
        post(server + manual_routes.MANUAL, {ManualRequestField.EDITION: "v2"})[0]
        == HTTPStatus.CONFLICT
    )


def test_running_robot_prevents_manual_update(manual_api, server):
    """
    Active robot execution keeps its acquired rules until explicitly paused.
    """
    post(server + RobotPhysicsRoute.START)
    before = knowledge(get_json(server + RobotPhysicsRoute.STATE))
    post(server + RobotPhysicsRoute.RUN)
    assert (
        post(server + RobotPhysicsRoute.MANUAL, {ManualRequestField.EDITION: "v2"})[0]
        == HTTPStatus.BAD_REQUEST
    )
    assert knowledge(get_json(server + RobotPhysicsRoute.STATE)) == before
    post(server + RobotPhysicsRoute.PAUSE)
    assert (
        post(server + RobotPhysicsRoute.MANUAL, {ManualRequestField.EDITION: "v2"})[0]
        == HTTPStatus.OK
    )


@pytest.mark.parametrize(
    "state", [ProgramState.IDLE, ProgramState.PAUSED, ProgramState.RUNNING]
)
def test_direct_adapter_obeys_manual_ownership(state, tmp_path):
    """
    Direct program calls enforce the same ownership as HTTP document updates.
    """
    physics = RecordedManualReadings(tmp_path)
    adapter = ControlledScaleAdapter(physics, tmp_path)
    adapter.program.state = state
    document = CentrifugeManual.builtin("v2")
    if state == ProgramState.RUNNING:
        with pytest.raises(InvalidPhysicsTarget):
            adapter.read_centrifuge_manual(document.document)
        assert physics.knowledge.active == CentrifugeManual.builtin("v1")
    else:
        adapter.read_centrifuge_manual(document.document)
        assert physics.knowledge.active == document


# %% protected public document route
@pytest.mark.parametrize("sharing_prefix", ["", "/laboratory"])
def test_shared_manual_requires_login_and_same_origin(
    shared_laboratory: SharedLaboratory,
):
    """
    Document updates retain the existing password and origin protection.
    """
    route = shared_laboratory.options.public_path(RobotPhysicsRoute.MANUAL)
    body = json.dumps({ManualRequestField.EDITION: "v2"}).encode()
    headers = {"Origin": shared_laboratory.origin, "Content-Type": "application/json"}
    assert (
        shared_laboratory.request(
            route, method="POST", body=body, headers=headers
        ).status
        == HTTPStatus.UNAUTHORIZED
    )
    assert shared_laboratory.backend.requests == []
    shared_laboratory.login()
    foreign_headers = {**headers, "Origin": "https://other.example"}
    assert (
        shared_laboratory.request(
            route, method="POST", body=body, headers=foreign_headers
        ).status
        == HTTPStatus.FORBIDDEN
    )
    assert shared_laboratory.backend.requests == []
    assert (
        shared_laboratory.request(
            route, method="POST", body=body, headers=headers
        ).status
        == HTTPStatus.OK
    )
    forwarded = shared_laboratory.backend.requests[-1]
    assert forwarded.path == RobotPhysicsRoute.MANUAL
    assert forwarded.body == body
    assert forwarded.cookie is None


def test_shared_manual_accepts_maximum_document_with_json_overhead(
    shared_laboratory: SharedLaboratory,
):
    """
    The document's supported byte limit remains usable through the gateway.
    """
    source = CentrifugeManual.builtin("v1").document
    document = source + "x" * (
        CentrifugeManual.MAXIMUM_DOCUMENT_BYTES - len(source.encode())
    )
    assert CentrifugeManual.from_markdown(document).document == document
    body = json.dumps({ManualRequestField.DOCUMENT: document}).encode()
    shared_laboratory.login()
    route = shared_laboratory.options.public_path(RobotPhysicsRoute.MANUAL)
    headers = {"Origin": shared_laboratory.origin, "Content-Type": "application/json"}
    assert (
        shared_laboratory.request(
            route, method="POST", body=body, headers=headers
        ).status
        == HTTPStatus.OK
    )
    assert shared_laboratory.backend.requests[-1].body == body


def test_shared_manual_rejects_excessive_request_before_proxying(
    shared_laboratory: SharedLaboratory,
):
    """
    Allowing JSON overhead still bounds the authenticated document request.
    """
    shared_laboratory.login()
    route = shared_laboratory.options.public_path(RobotPhysicsRoute.MANUAL)
    headers = {
        "Origin": shared_laboratory.origin,
        "Content-Type": "application/json",
        "Content-Length": str(shared_laboratory.options.MAX_MANUAL_BODY_BYTES + 1),
    }
    assert (
        shared_laboratory.request(
            route, method="POST", body=b"{}", headers=headers
        ).status
        == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    )
    assert shared_laboratory.backend.requests == []
