"""
The contact laboratory owns one isolated simulation behind a bounded local API.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cramera import laboratory_physics_session
from cramera import server as server_module
from cramera.laboratory_physics_session import (
    InvalidPhysicsRequest,
    LaboratoryPhysicsSession,
    PhysicsField,
    PhysicsRoute,
    PhysicsState,
)

from .test_server import get_json, post, server


# %% deterministic contact simulation
@dataclass
class ControlledContactSimulation:
    """
    Expose target and step changes without importing a physics engine.
    """

    bundle_directory: Path
    """The authored bundle used to construct the simulated laboratory."""
    target: dict | None = None
    """
    The requested object and desired position.
    """

    steps: int = 0
    """
    The fixed simulation steps performed by the session worker.
    """

    stepped: threading.Event = field(default_factory=threading.Event)
    """
    Signal that the background worker has progressed.
    """

    timestep: float = 0.001
    """
    The simulated seconds advanced per step.
    """

    def step(self, count: int = 1) -> None:
        """
        Record simulation progress for worker lifecycle assertions.
        """
        self.steps += count
        self.stepped.set()

    def reset(self) -> None:
        """
        Restore the initial free object state.
        """
        self.target = None

    def set_target(self, key: str, position: list[float]) -> None:
        """
        Retain the validated target passed by the session.
        """
        self.target = {"key": key, "position": position}

    def release(self) -> None:
        """
        Remove the manipulation force target.
        """
        self.target = None

    def snapshot(self) -> dict:
        """
        Return an independently serializable object and contact snapshot.
        """
        return {
            "objects": {"tube_clear": [0.16, -0.065, 0.906, 0, 0, 0, 1]},
            "target": self.target,
            "contacts": [],
            "time": self.steps * self.timestep,
        }


@pytest.fixture()
def physics_bundle(fixture_scene) -> Path:
    """
    Build an authored laboratory without requiring the local Blender assets.
    """
    directory = fixture_scene / "scenes" / LaboratoryPhysicsSession.SOURCE_SCENE
    directory.mkdir()
    original = {
        "name": LaboratoryPhysicsSession.SOURCE_SCENE,
        "camera": {"position": [2, -2.8, 1.85], "target": [0, 0, 0.92]},
        "rendering": {"ambientOcclusion": False, "exposure": 0.65},
        "models": [
            {
                "name": "laboratory",
                "urdf": "environment.urdf",
                "robot": False,
                "preserveMaterials": True,
            }
        ],
        "objects": [
            {
                "key": "tube_clear",
                "mesh": "assets/tube.glb",
                "spawn": [0.16, -0.065, 0.906, 0, 0, 0, 1],
                "preserveMaterials": True,
            }
        ],
        "laboratory": {"tubes": [{"key": "tube_clear", "slot": "A1"}]},
        "validation": {"mode": "manual_kinematic"},
    }
    (directory / "scene.json").write_text(json.dumps(original))
    (directory / "trajectory.json").write_text(json.dumps({"frames": [{"drawer": 0}]}))
    (directory / "assets").mkdir()
    (directory / "assets" / "tube.glb").write_bytes(b"fixture mesh")
    (directory / "environment.urdf").write_bytes(b"fixture urdf")
    (directory / "laboratory.blend").write_bytes(b"large authoring file")
    (directory / "render_overview.png").write_bytes(b"render")
    return directory


@pytest.fixture()
def contact_session(physics_bundle, monkeypatch):
    """
    Run the real session with a controlled engine and stop every worker.
    """
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", ControlledContactSimulation
    )
    session = LaboratoryPhysicsSession(physics_bundle.parents[1])
    yield session
    session.stop()


# %% scene isolation and worker ownership
def test_scene_copy_retains_authored_presentation_without_manual_controls(
    contact_session, physics_bundle
):
    """
    Physics gets its own lightweight scene while preserving the authored look.
    """
    original = (physics_bundle / "scene.json").read_bytes()
    assert contact_session.start()[PhysicsField.STATE] == PhysicsState.RUNNING
    destination = contact_session.output_directory
    scene = json.loads((destination / "scene.json").read_text())
    source = json.loads(original)
    for key in ("camera", "rendering", "models", "objects"):
        assert scene[key] == source[key]
    assert "laboratory" not in scene
    assert scene["physics"]["objects"] == [{"key": "tube_clear", "label": "tube_clear"}]
    assert (destination / "assets" / "tube.glb").read_bytes() == b"fixture mesh"
    assert not (destination / "laboratory.blend").exists()
    assert not (destination / "render_overview.png").exists()
    assert (physics_bundle / "scene.json").read_bytes() == original


def test_repeated_start_keeps_the_same_running_engine(contact_session):
    """
    Double clicks preserve the existing simulation and worker.
    """
    contact_session.start()
    engine = contact_session.physics
    worker = contact_session.worker
    assert engine.stepped.wait(1)
    assert contact_session.start()[PhysicsField.STATE] == PhysicsState.RUNNING
    assert contact_session.physics is engine
    assert contact_session.worker is worker


def test_physics_controls_keep_authored_camera_presets(contact_session, physics_bundle):
    """
    Contact manipulation retains the laboratory's inspection cameras.
    """
    scene_path = physics_bundle / "scene.json"
    scene = json.loads(scene_path.read_text())
    cameras = {"overview": scene["camera"]}
    scene["laboratory"]["cameras"] = cameras
    scene_path.write_text(json.dumps(scene))
    contact_session.start()
    copied = json.loads((contact_session.output_directory / "scene.json").read_text())
    assert copied["physics"]["cameras"] == cameras


def test_stop_joins_the_worker_and_reports_idle(contact_session):
    """
    An ended session cannot leave a physics worker running.
    """
    contact_session.start()
    worker = contact_session.worker
    result = contact_session.stop()
    assert result[PhysicsField.STATE] == PhysicsState.IDLE
    assert not worker.is_alive()


def test_target_release_and_reset_control_the_owned_engine(contact_session):
    """
    Browser targets reach the engine and release/reset remove applied forces.
    """
    contact_session.start()
    position = [0.2, -0.05, 1.1]
    assert contact_session.set_target("tube_clear", position)["target"] == {
        "key": "tube_clear",
        "position": position,
    }
    assert contact_session.release()["target"] is None
    contact_session.set_target("tube_clear", position)
    assert contact_session.reset()["target"] is None


@pytest.mark.parametrize(
    "key,position",
    [
        ("unknown", [0.2, 0, 1]),
        ("tube_clear", [float("nan"), 0, 1]),
        ("tube_clear", [0, 0, float("inf")]),
        ("tube_clear", [0, 0, 10]),
        ("tube_clear", [True, 0, 1]),
        ("tube_clear", [0, 1]),
    ],
)
def test_invalid_targets_do_not_change_the_engine(contact_session, key, position):
    """
    Unknown objects and invalid or out-of-range positions never reach physics.
    """
    contact_session.start()
    with pytest.raises(InvalidPhysicsRequest):
        contact_session.set_target(key, position)
    assert contact_session.physics.target is None


# %% local HTTP contract
@pytest.fixture()
def controlled_api(physics_bundle, monkeypatch):
    """
    Replace only the optional physics engine used by the real HTTP server.
    """
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", ControlledContactSimulation
    )


def test_api_starts_the_isolated_scene_and_accepts_targets(controlled_api, server):
    """
    The server provides the contact scene and exposes updated object targets.
    """
    code, result = post(server + PhysicsRoute.START)
    assert code == 202
    assert result[PhysicsField.VIEWER_URL] == LaboratoryPhysicsSession.VIEWER_URL
    code, result = post(
        server + PhysicsRoute.TARGET, {"key": "tube_clear", "position": [0.2, 0, 1]}
    )
    assert code == 200
    assert result["target"] == {"key": "tube_clear", "position": [0.2, 0, 1]}
    assert (
        get_json(server + PhysicsRoute.STATE)[PhysicsField.STATE]
        == PhysicsState.RUNNING
    )
    assert post(server + PhysicsRoute.STOP)[1][PhysicsField.STATE] == PhysicsState.IDLE


def test_api_rejects_extra_execution_parameters(controlled_api, server):
    """
    A browser cannot supply an engine, bundle path, or executable command.
    """
    code, result = post(server + PhysicsRoute.START, {"bundle_directory": "/tmp"})
    assert code == 400
    assert result[PhysicsField.OK] is False
    assert (
        get_json(server + PhysicsRoute.STATE)[PhysicsField.STATE] == PhysicsState.IDLE
    )


def test_api_rejects_cross_origin_mutation(controlled_api, server):
    """
    Other websites cannot start or manipulate local contact sessions.
    """
    request = urllib.request.Request(
        server + PhysicsRoute.START,
        method="POST",
        data=b"{}",
        headers={"Origin": "https://example.org", "Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as failure:
        urllib.request.urlopen(request)
    assert failure.value.code == 403
    assert (
        get_json(server + PhysicsRoute.STATE)[PhysicsField.STATE] == PhysicsState.IDLE
    )


def test_api_requires_a_running_session_before_targeting(controlled_api, server):
    """
    Target requests do not implicitly start a simulation.
    """
    code, result = post(
        server + PhysicsRoute.TARGET, {"key": "tube_clear", "position": [0.2, 0, 1]}
    )
    assert code == 409
    assert result[PhysicsField.OK] is False


def test_server_close_joins_owned_physics_worker(controlled_api):
    """
    Closing the listener also terminates its simulation worker.
    """
    listener = server_module.make_server(0)
    try:
        listener.laboratory_physics.start()
        worker = listener.laboratory_physics.worker
        assert worker.is_alive()
    finally:
        listener.server_close()
    assert not worker.is_alive()


def test_occupied_port_preserves_the_bind_error(controlled_api):
    """
    A listener that fails to bind can clean up its partially initialized state.
    """
    listener = server_module.make_server(0)
    try:
        with pytest.raises(OSError):
            server_module.make_server(listener.server_address[1])
    finally:
        listener.server_close()


# %% high-frequency state polling
@dataclass
class RequestLoggingHandler(server_module.Handler):
    """
    Exercise request logging without opening a socket or parsing HTTP.
    """

    command: str
    """
    The request method whose access log is under test.
    """

    path: str
    """
    The requested API route, including any query string.
    """

    @property
    def requestline(self) -> str:
        """
        Return the request line logged by the standard HTTP handler.
        """
        return f"{self.command} {self.path} HTTP/1.1"


@pytest.mark.parametrize(
    "method,route,code,logged",
    [
        ("GET", PhysicsRoute.STATE, 200, False),
        ("GET", PhysicsRoute.STATE + "?sample=1", 200, False),
        ("GET", PhysicsRoute.STATE, 500, True),
        ("GET", PhysicsRoute.STATE, 404, True),
        ("POST", PhysicsRoute.STATE, 200, True),
        ("GET", PhysicsRoute.START, 200, True),
    ],
)
def test_only_successful_physics_state_polling_is_silent(
    caplog, method, route, code, logged
):
    """
    Routine polling stays quiet while failures and unrelated requests remain visible.
    """
    caplog.set_level(logging.INFO, logger=server_module.logger.name)
    handler = RequestLoggingHandler(method, route)
    handler.log_request(code)
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == server_module.logger.name
    ]
    assert messages == ([f'  "{handler.requestline}" {code} -'] if logged else [])
