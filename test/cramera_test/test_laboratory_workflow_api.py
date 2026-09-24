"""
Explicit workflow preparation, resumable ownership and authenticated control.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from http import HTTPStatus
import json
from pathlib import Path

import pytest
from typing_extensions import Any

from cramera import laboratory_robot_session
from cramera.laboratory_physics_program import ProgramField, ProgramState
from cramera.laboratory_physics_session import PhysicsField
from cramera.laboratory_refrigerator_command import (
    RefrigeratorAction,
    RefrigeratorField,
)
from cramera.laboratory_robot_session import RobotField, RobotPhysicsRoute
from cramera.laboratory_workflow_command import (
    WorkflowCommand,
    InvalidWorkflowCommand,
    WorkflowField,
    WorkflowSource,
)
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_physics_session import physics_bundle
from .test_laboratory_robot_session import ControlledRobotSimulation, robot_bundle
from .test_laboratory_share import SharedLaboratory, shared_laboratory, sharing_prefix
from .test_server import get_json, post, server


# %% observable preparation and resume
class ObservationField(StrEnum):
    """Recorded request evidence added by the test simulation."""

    PREPARATIONS = "preparations"
    """Preparation decisions forwarded through the HTTP boundary."""


@dataclass
class RecordedWorkflowRequests(ControlledRobotSimulation):
    """
    Record the distinction between demo preparation and ordinary resume.
    """

    preparations: list[bool] = field(default_factory=list)
    """
    Explicit preparation flags forwarded to the program adapter.
    """

    def run_workflow(self, prepare: bool) -> None:
        """
        Record one workflow request and acquire robot ownership.
        """
        self.preparations.append(prepare)
        self.run_program()

    def snapshot(self) -> dict[str, Any]:
        """
        Expose workflow ownership even while its program is paused.
        """
        snapshot = super().snapshot()
        if self.preparations:
            snapshot[RobotField.ROBOT].update(
                {
                    ProgramField.SOURCE: WorkflowSource.PROGRAM,
                    WorkflowField.WORKFLOW: {
                        WorkflowField.RESERVED: self.program_state
                        in (ProgramState.RUNNING, ProgramState.PAUSED)
                    },
                }
            )
        snapshot[ObservationField.PREPARATIONS] = self.preparations.copy()
        return snapshot


@pytest.fixture
def workflow_api(
    physics_bundle: Path, robot_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Keep the actual HTTP and session boundaries around recorded intentions.
    """
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", RecordedWorkflowRequests
    )


# %% validated requests
@pytest.mark.parametrize("prepare", [True, False])
def test_explicit_prepare_and_resume_flags(prepare: bool) -> None:
    """
    The request preserves the caller's explicit preparation decision.
    """
    assert (
        WorkflowCommand.from_payload({WorkflowField.PREPARE: prepare}).prepare
        is prepare
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        None,
        [],
        {WorkflowField.PREPARE: 1},
        {WorkflowField.PREPARE: "true"},
        {WorkflowField.PREPARE: True, PhysicsField.POSITION: [0, 0, 0]},
    ],
)
def test_invalid_workflow_requests(payload: Any) -> None:
    """
    Ambiguous or injected preparation requests are rejected.
    """
    with pytest.raises(InvalidWorkflowCommand):
        WorkflowCommand.from_payload(payload)


def test_workflow_requires_active_robot_session(
    workflow_api: None, server: str
) -> None:
    """
    Preparation cannot start an unrequested simulation worker.
    """
    assert (
        post(server + RobotPhysicsRoute.WORKFLOW, {WorkflowField.PREPARE: True})[0]
        == HTTPStatus.CONFLICT
    )


def test_workflow_forwards_prepare_and_resume(workflow_api: None, server: str) -> None:
    """
    Resuming a paused workflow forwards false rather than preparing samples again.
    """
    post(server + RobotPhysicsRoute.START)
    code, result = post(
        server + RobotPhysicsRoute.WORKFLOW, {WorkflowField.PREPARE: True}
    )
    assert code == HTTPStatus.OK
    assert result[ObservationField.PREPARATIONS] == [True]
    post(server + RobotPhysicsRoute.PAUSE)
    code, result = post(
        server + RobotPhysicsRoute.WORKFLOW, {WorkflowField.PREPARE: False}
    )
    assert code == HTTPStatus.OK
    assert result[ObservationField.PREPARATIONS] == [True, False]


@pytest.mark.parametrize(
    "route,payload",
    [
        (
            RobotPhysicsRoute.TARGET,
            {
                PhysicsField.KEY: LaboratoryBody.CLEAR_TUBE,
                PhysicsField.POSITION: [0.2, 0, 1.1],
            },
        ),
        (
            RobotPhysicsRoute.LIQUID,
            {PhysicsField.KEY: LaboratoryBody.CLEAR_TUBE, PhysicsField.VOLUME_ML: 1},
        ),
        (RobotPhysicsRoute.SCALE_TARE, {}),
        (
            RobotPhysicsRoute.REFRIGERATOR,
            {RefrigeratorField.ACTION: RefrigeratorAction.OPEN},
        ),
    ],
)
def test_paused_workflow_retains_sample_ownership(
    workflow_api: None, server: str, route: RobotPhysicsRoute, payload: dict[str, Any]
) -> None:
    """
    Pause holds the workflow state without yielding samples to manual edits.
    """
    post(server + RobotPhysicsRoute.START)
    post(server + RobotPhysicsRoute.WORKFLOW, {WorkflowField.PREPARE: True})
    post(server + RobotPhysicsRoute.PAUSE)
    assert post(server + route, payload)[0] == HTTPStatus.BAD_REQUEST
    assert (
        get_json(server + RobotPhysicsRoute.STATE)[RobotField.ROBOT][ProgramField.STATE]
        == ProgramState.PAUSED
    )


@pytest.mark.parametrize("sharing_prefix", ["", "/laboratory"])
def test_workflow_requires_existing_password(
    shared_laboratory: SharedLaboratory,
) -> None:
    """
    The new program uses the established authentication and origin boundary.
    """
    route = shared_laboratory.options.public_path(RobotPhysicsRoute.WORKFLOW)
    payload = json.dumps({WorkflowField.PREPARE: True}).encode()
    headers = {"Origin": shared_laboratory.origin, "Content-Type": "application/json"}
    assert (
        shared_laboratory.request(
            route, method="POST", body=payload, headers=headers
        ).status
        == HTTPStatus.UNAUTHORIZED
    )
    assert shared_laboratory.backend.requests == []
    shared_laboratory.login()
    assert (
        shared_laboratory.request(
            route, method="POST", body=payload, headers=headers
        ).status
        == HTTPStatus.OK
    )
    assert shared_laboratory.backend.requests[-1].body == payload
