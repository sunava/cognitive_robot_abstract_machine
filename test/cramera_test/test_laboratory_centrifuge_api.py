"""
Centrifuge control shares session locks and the authenticated laboratory gateway.
"""

from dataclasses import dataclass, field
from http import HTTPStatus
import json

import pytest

from cramera import laboratory_physics_session, laboratory_robot_session
from cramera.laboratory_centrifuge_command import (
    CentrifugeAction,
    CentrifugeCommand,
    CentrifugeField,
)
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import ProgramState
from cramera.laboratory_physics_session import (
    LaboratoryPhysicsSession,
    PhysicsField,
    PhysicsFile,
    PhysicsRoute,
)
from cramera.laboratory_robot_session import LaboratoryRobotSession, RobotPhysicsRoute

from .test_laboratory_physics_session import physics_bundle
from .test_laboratory_robot_session import ControlledRobotSimulation, robot_bundle
from .test_laboratory_scale_api import ControlledScaleAdapter
from .test_laboratory_share import SharedLaboratory, shared_laboratory, sharing_prefix
from .test_server import get_json, post, server


# %% observable instrument operations
@dataclass
class RecordedInstrumentCommands(ControlledRobotSimulation):
    """
    Record operations accepted by the serialized instrument API.
    """

    commands: list = field(default_factory=list)
    """
    Physical operation arguments received from the session.
    """

    def centrifuge_command(self, action, slot=None, key=None, rpm=None, duration=None):
        """
        Record an instrument operation without loading articulated geometry.
        """
        self.commands.append(CentrifugeCommand(action, slot, key, rpm, duration))

    def snapshot(self):
        """
        Expose accepted operation count beside the robot state.
        """
        return {**super().snapshot(), CentrifugeField.CENTRIFUGE: len(self.commands)}


@pytest.fixture
def centrifuge_api(physics_bundle, robot_bundle, monkeypatch):
    """
    Use observable independent instruments in both session variants.
    """
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", RecordedInstrumentCommands
    )
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", RecordedInstrumentCommands
    )


@pytest.fixture(params=[PhysicsRoute, RobotPhysicsRoute])
def centrifuge_routes(request):
    """
    Exercise the manual and robot instrument routes.
    """
    return request.param


# %% session command validation
@pytest.mark.parametrize(
    "body",
    [
        {"action": "open"},
        {"action": "close"},
        {"action": "stop"},
        {"action": "secure", "slot": "C1", "key": "tube_clear"},
        {"action": "release", "slot": "C1"},
        {"action": "start", "rpm": 120, "duration": 20},
    ],
)
def test_instrument_command_isolated_to_selected_session(
    centrifuge_api, centrifuge_routes, server, body
):
    """
    A valid operation cannot mutate another user's local simulation session.
    """
    for routes in (PhysicsRoute, RobotPhysicsRoute):
        assert post(server + routes.START)[0] == HTTPStatus.ACCEPTED
    code, result = post(server + centrifuge_routes.CENTRIFUGE, body)
    assert code == HTTPStatus.OK
    assert result[CentrifugeField.CENTRIFUGE] == 1
    other = RobotPhysicsRoute if centrifuge_routes is PhysicsRoute else PhysicsRoute
    assert get_json(server + other.STATE)[CentrifugeField.CENTRIFUGE] == 0


@pytest.mark.parametrize(
    "body",
    [
        {"action": "secure", "slot": "C1", "key": "unknown"},
        {"action": "start", "rpm": True},
        {"action": "open", "position": [0, 0, 0]},
    ],
)
def test_invalid_instrument_request_does_not_mutate(
    centrifuge_api, centrifuge_routes, server, body
):
    """
    Invalid values and pose injections never reach the physical instrument.
    """
    post(server + centrifuge_routes.START)
    code, result = post(server + centrifuge_routes.CENTRIFUGE, body)
    assert code == HTTPStatus.BAD_REQUEST
    assert result[PhysicsField.OK] is False
    assert get_json(server + centrifuge_routes.STATE)[CentrifugeField.CENTRIFUGE] == 0


def test_instrument_requires_live_session(centrifuge_api, centrifuge_routes, server):
    """
    An instrument operation cannot implicitly start physics.
    """
    assert (
        post(server + centrifuge_routes.CENTRIFUGE, {"action": "open"})[0]
        == HTTPStatus.CONFLICT
    )


def test_robot_ownership_blocks_manipulation_but_allows_rotor_stop(
    centrifuge_api, server
):
    """
    Stopping remains available while the robot owns object manipulation.
    """
    post(server + RobotPhysicsRoute.START)
    post(server + RobotPhysicsRoute.RUN)
    assert (
        post(server + RobotPhysicsRoute.CENTRIFUGE, {"action": "open"})[0]
        == HTTPStatus.BAD_REQUEST
    )
    assert (
        post(server + RobotPhysicsRoute.CENTRIFUGE, {"action": "stop"})[0]
        == HTTPStatus.OK
    )
    assert get_json(server + RobotPhysicsRoute.STATE)[CentrifugeField.CENTRIFUGE] == 1


@pytest.mark.parametrize(
    "session_type", [LaboratoryPhysicsSession, LaboratoryRobotSession]
)
def test_live_scene_retains_instrument_metadata(
    centrifuge_api, physics_bundle, session_type
):
    """
    Scene controls receive the authored instrument description.
    """
    scene_path = physics_bundle / PhysicsFile.SCENE
    authored = json.loads(scene_path.read_text())
    authored[PhysicsField.LABORATORY][CentrifugeField.CENTRIFUGE] = {
        CentrifugeField.AVAILABLE: True
    }
    scene_path.write_text(json.dumps(authored))
    session = session_type(physics_bundle.parents[1])
    try:
        session.start()
        copied = json.loads((session.output_directory / PhysicsFile.SCENE).read_text())
        assert (
            copied[PhysicsField.PHYSICS][CentrifugeField.CENTRIFUGE]
            == authored[PhysicsField.LABORATORY][CentrifugeField.CENTRIFUGE]
        )
    finally:
        session.stop()


# %% direct adapter and protected gateway
@pytest.mark.parametrize(
    "state", [ProgramState.IDLE, ProgramState.PAUSED, ProgramState.RUNNING]
)
def test_program_wrapper_applies_ownership_to_instrument(state, tmp_path):
    """
    Direct program wrapper calls share the session ownership contract.
    """
    physics = RecordedInstrumentCommands(tmp_path)
    adapter = ControlledScaleAdapter(physics, tmp_path)
    adapter.program.state = state
    if state == ProgramState.RUNNING:
        with pytest.raises(InvalidPhysicsTarget):
            adapter.centrifuge_command(CentrifugeAction.OPEN)
    else:
        adapter.centrifuge_command(CentrifugeAction.OPEN)
        assert physics.commands == [CentrifugeCommand(CentrifugeAction.OPEN)]
    physics.commands.clear()
    adapter.centrifuge_command(CentrifugeAction.STOP)
    assert physics.commands == [CentrifugeCommand(CentrifugeAction.STOP)]


@pytest.mark.parametrize("sharing_prefix", ["", "/laboratory"])
def test_shared_instrument_requires_login(shared_laboratory: SharedLaboratory):
    """
    Remote centrifuge operations pass through the existing password gate.
    """
    route = shared_laboratory.options.public_path(RobotPhysicsRoute.CENTRIFUGE)
    body = json.dumps({"action": "start", "rpm": 120, "duration": 20}).encode()
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
    forwarded = shared_laboratory.backend.requests[-1]
    assert forwarded.path == RobotPhysicsRoute.CENTRIFUGE
    assert forwarded.body == body
    assert forwarded.cookie is None


# %% initial articulated presentation
@dataclass
class ObservedInstrumentJoints(ControlledRobotSimulation):
    """
    Expose an instrument pose that differs from the authored starting frame.
    """

    def snapshot(self):
        """
        Include the observed open lid alongside the robot joint feedback.
        """
        return {
            **super().snapshot(),
            CentrifugeField.CENTRIFUGE: {
                CentrifugeField.JOINTS: {"laboratory_centrifuge_lid_joint": 1.5}
            },
        }


@pytest.mark.parametrize(
    "session_type", [LaboratoryPhysicsSession, LaboratoryRobotSession]
)
def test_initial_trajectory_retains_measured_instrument_joints(
    physics_bundle, robot_bundle, monkeypatch, session_type
):
    """
    The initial render uses the same instrument articulation as live physics.
    """
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", ObservedInstrumentJoints
    )
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", ObservedInstrumentJoints
    )
    session = session_type(physics_bundle.parents[1])
    try:
        session.start()
        trajectory = json.loads(
            (session.output_directory / PhysicsFile.TRAJECTORY).read_text()
        )
        expected = session.physics.snapshot()[CentrifugeField.CENTRIFUGE][
            CentrifugeField.JOINTS
        ]
        for joint, angle in expected.items():
            assert trajectory[PhysicsField.FRAMES][0][joint] == angle
    finally:
        session.stop()
