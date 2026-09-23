"""Authenticated sharing exposes only the fixed laboratory and its controls."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

import pytest

from cramera import paths
from cramera.laboratory_robot_session import RobotPhysicsRoute
from cramera.laboratory_share import ShareOptions, ShareRoute, make_server


# %% isolated HTTP peers
@dataclass
class ForwardedRequest:
    """A request received by the isolated simulation mimic."""

    method: str
    """Requested operation."""
    path: str
    """Requested route."""
    origin: str | None
    """Origin supplied by the sharing gateway."""
    cookie: str | None
    """Authentication cookies reaching the backend, if any."""
    body: bytes
    """Raw request body."""


class ControlReceiver(BaseHTTPRequestHandler):
    """Record control requests without starting a simulation."""

    def do_GET(self) -> None:
        """Return an inert successful simulation reply."""
        size = int(self.headers.get("Content-Length", "0"))
        self.server.requests.append(
            ForwardedRequest(
                self.command,
                self.path,
                self.headers.get("Origin"),
                self.headers.get("Cookie"),
                self.rfile.read(size),
            )
        )
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        """Accept the same bounded control contract for a mutation."""
        self.do_GET()

    def log_message(self, format: str, *args: object) -> None:
        """Keep deterministic HTTP tests quiet."""


@dataclass(init=False)
class ControlReceiverServer(ThreadingHTTPServer):
    """Own the list of requests received by one isolated backend."""

    requests: list[ForwardedRequest]
    """Requests observed during the test."""

    def __init__(self) -> None:
        """Listen on an ephemeral loopback port."""
        self.requests = []
        super().__init__(("127.0.0.1", 0), ControlReceiver)


@dataclass
class SharingResponse:
    """Fully consumed response from a sharing request."""

    status: int
    """HTTP response status."""
    body: bytes
    """Returned payload."""
    headers: dict[str, str]
    """Returned response headers."""


@dataclass
class SharedLaboratory:
    """An isolated gateway with an inert simulation backend."""

    server: ThreadingHTTPServer
    """Gateway listener under test."""
    backend: ControlReceiverServer
    """Receiver of allowed proxy requests."""
    options: ShareOptions
    """Configuration shared by assertions and the gateway."""
    origin: str
    """Public HTTPS origin accepted by the gateway."""
    password: str
    """Test-only login secret."""
    cookie: str = ""
    """Authenticated browser cookie after login."""

    def request(
        self,
        path: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> SharingResponse:
        """Send an HTTP request carrying the tunnel's public host."""
        outgoing = {"Host": self.origin.removeprefix("https://")}
        if self.cookie:
            outgoing["Cookie"] = self.cookie
        outgoing.update(headers or {})
        connection = HTTPConnection(*self.server.server_address, timeout=5)
        connection.request(method, path, body=body, headers=outgoing)
        response = connection.getresponse()
        result = SharingResponse(
            response.status, response.read(), dict(response.getheaders())
        )
        connection.close()
        return result

    def login(self) -> SharingResponse:
        """Authenticate and retain the browser's session cookie."""
        result = self.request(
            self.options.path_prefix + ShareRoute.LOGIN,
            method="POST",
            body=urlencode({"password": self.password}).encode(),
            headers={
                "Origin": self.origin,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        self.cookie = result.headers["Set-Cookie"].split(";", 1)[0]
        return result


@pytest.fixture
def sharing_prefix() -> str:
    """Default to a host-root share unless a test selects a mounted laboratory."""
    return ""


@pytest.fixture
def shared_laboratory(fixture_scene: Path, sharing_prefix: str) -> SharedLaboratory:
    """Serve the existing miniature scene behind an authenticated gateway."""
    password = "test-only-strong-password"
    origin = "https://laboratory.example.test"
    password_file = fixture_scene / "password"
    password_file.write_text(password)
    origin_file = fixture_scene / "origin"
    origin_file.write_text(origin)
    backend = ControlReceiverServer()
    options = ShareOptions(
        port=0,
        backend_port=backend.server_address[1],
        password_file=password_file,
        public_origin_file=origin_file,
        scene_directory=fixture_scene / "scenes" / "fixture",
        path_prefix=sharing_prefix,
    )
    server = make_server(options)
    workers = [
        threading.Thread(target=listener.serve_forever, daemon=True)
        for listener in (backend, server)
    ]
    for worker in workers:
        worker.start()
    yield SharedLaboratory(server, backend, options, origin, password)
    for listener in (server, backend):
        listener.shutdown()
        listener.server_close()
    for worker in workers:
        worker.join()


# %% authentication and public origin
@pytest.mark.parametrize("sharing_prefix", ["/laboratory"])
def test_mounted_login_stays_within_laboratory(
    shared_laboratory: SharedLaboratory,
) -> None:
    """Form submission and authenticated redirects retain the mounted path."""
    prefix = shared_laboratory.options.path_prefix
    form = shared_laboratory.request(prefix + ShareRoute.ROOT)
    assert form.status == 200
    assert f'action="{prefix}{ShareRoute.LOGIN}"'.encode() in form.body
    response = shared_laboratory.login()
    assert response.headers["Location"] == prefix + ShareRoute.VIEWER
    assert "Path=/;" in response.headers["Set-Cookie"]
    assert "Domain=" not in response.headers["Set-Cookie"]
    assert shared_laboratory.request(prefix + ShareRoute.VIEWER).status == 200


@pytest.mark.parametrize("sharing_prefix", ["/laboratory"])
def test_mounted_controls_keep_internal_backend_route(
    shared_laboratory: SharedLaboratory,
) -> None:
    """A mounted URL reaches the existing backend without its public path prefix."""
    shared_laboratory.login()
    response = shared_laboratory.request(
        shared_laboratory.options.path_prefix + RobotPhysicsRoute.STATE
    )
    assert response.status == 200
    assert shared_laboratory.backend.requests[0].path == RobotPhysicsRoute.STATE
    assert shared_laboratory.request(RobotPhysicsRoute.STATE).status == 404


@pytest.mark.parametrize("sharing_prefix", ["/laboratory"])
def test_mounted_viewer_controls_target_same_laboratory(
    shared_laboratory: SharedLaboratory,
) -> None:
    """The browser's simulation client uses the prefixed service endpoints."""
    shared_laboratory.login()
    response = shared_laboratory.request(
        shared_laboratory.options.path_prefix + "/core/laboratory-physics.js"
    )
    assert response.status == 200
    for route in RobotPhysicsRoute:
        assert (
            f"'{shared_laboratory.options.path_prefix}{route}'".encode()
            in response.body
        )
        assert f"'{route}'".encode() not in response.body


@pytest.mark.parametrize("sharing_prefix", ["/laboratory"])
@pytest.mark.parametrize(
    "outside",
    [
        "/",
        "/index.html",
        "/laboratory-other/index.html",
        "/laboratory/../index.html",
        "/laboratory/%2e%2e/index.html",
    ],
)
def test_mounted_share_rejects_outside_routes(
    shared_laboratory: SharedLaboratory, outside: str
) -> None:
    """The mount cannot expose or redirect neighboring application routes."""
    assert shared_laboratory.request(outside).status == 404


def test_missing_public_origin_fails_closed(
    shared_laboratory: SharedLaboratory,
) -> None:
    """An unfinished tunnel configuration cannot expose the viewer."""
    shared_laboratory.options.public_origin_file.unlink()
    assert shared_laboratory.request(ShareRoute.ROOT).status == 503


def test_login_form_preserves_same_origin_for_browser_submission(
    shared_laboratory: SharedLaboratory,
) -> None:
    """The form's referrer policy must not replace its POST Origin with null."""
    response = shared_laboratory.request(ShareRoute.ROOT)
    assert response.headers["Referrer-Policy"] == "same-origin"


def test_viewer_connection_policy_allows_embedded_texture_blobs(
    shared_laboratory: SharedLaboratory,
) -> None:
    """The GLB loader can fetch embedded texture blobs without external origins."""
    shared_laboratory.login()
    response = shared_laboratory.request(ShareRoute.VIEWER)
    directives = {
        parts[0]: parts[1:]
        for directive in response.headers["Content-Security-Policy"].split(";")
        if (parts := directive.strip().split())
    }
    assert directives["connect-src"] == ["'self'", "blob:"]


@pytest.mark.parametrize(
    "path", ["/core/bus.js", "/scenes/index.json", RobotPhysicsRoute.STATE]
)
def test_every_asset_and_state_requires_login(
    shared_laboratory: SharedLaboratory, path: str
) -> None:
    """Data is withheld before the browser authenticates."""
    assert shared_laboratory.request(path).status == 401
    assert shared_laboratory.backend.requests == []


def test_login_sets_protected_cookie_and_opens_laboratory(
    shared_laboratory: SharedLaboratory,
) -> None:
    """A correct password establishes a browser session for the canonical page."""
    response = shared_laboratory.login()
    assert response.status == 303
    assert response.headers["Location"] == ShareRoute.VIEWER
    for directive in ("HttpOnly", "SameSite=Strict", "Secure"):
        assert directive in response.headers["Set-Cookie"]
    assert shared_laboratory.request(ShareRoute.VIEWER).status == 200


def test_wrong_password_does_not_create_session(
    shared_laboratory: SharedLaboratory,
) -> None:
    """An incorrect password leaves every simulation route inaccessible."""
    response = shared_laboratory.request(
        ShareRoute.LOGIN,
        method="POST",
        body=urlencode({"password": "incorrect"}).encode(),
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    assert response.status == 401
    assert "Set-Cookie" not in response.headers


def test_unknown_host_is_rejected(shared_laboratory: SharedLaboratory) -> None:
    """The tunnel cannot select an arbitrary HTTP host."""
    response = shared_laboratory.request(
        ShareRoute.ROOT, headers={"Host": "untrusted.example.test"}
    )
    assert response.status == 403


# %% exact asset and operation boundaries
def test_viewer_html_image_dependency_is_available(
    shared_laboratory: SharedLaboratory,
) -> None:
    """The viewer's declared brand image remains available after authentication."""
    shared_laboratory.login()
    relative = "img/aicor-logo.png"
    response = shared_laboratory.request("/" + relative)
    assert response.status == 200
    assert response.body == (paths.WEB_ROOT / relative).read_bytes()


@pytest.mark.parametrize(
    "path",
    [
        "/api/eql",
        "/api/plan/scaffold",
        "/api/recording/save",
        "/api/laboratory/start",
        "/api/laboratory/physics/start",
        "/scenes/other/scene.json",
        "/scenes/precision_lab_pr2_physics/semantics.json",
        "/core/../config.js",
        "/core/%2e%2e/config.js",
        "/core/%252e%252e/config.js",
        "/core/%5c../config.js",
        "https://elsewhere.example.test/core/bus.js",
        "/core/",
    ],
)
def test_unlisted_routes_and_traversal_are_rejected(
    shared_laboratory: SharedLaboratory, path: str
) -> None:
    """Authentication grants only the fixed laboratory dependency graph."""
    shared_laboratory.login()
    assert shared_laboratory.request(path).status == 404
    assert shared_laboratory.backend.requests == []


def test_filtered_scene_index_and_dependency_are_available(
    shared_laboratory: SharedLaboratory,
) -> None:
    """The scene picker reveals only the shared laboratory."""
    shared_laboratory.login()
    response = shared_laboratory.request(ShareRoute.SCENE_INDEX)
    payload = json.loads(response.body)
    assert payload["default"] == ShareOptions.SCENE_NAME
    assert [scene["name"] for scene in payload["scenes"]] == [ShareOptions.SCENE_NAME]
    response = shared_laboratory.request(
        f"/scenes/{ShareOptions.SCENE_NAME}/robot.urdf"
    )
    assert (
        response.body
        == (shared_laboratory.options.scene_directory / "robot.urdf").read_bytes()
    )


def test_changed_symlink_cannot_escape_scene(
    shared_laboratory: SharedLaboratory, tmp_path: Path
) -> None:
    """A file replaced after startup remains confined to the shared bundle."""
    shared_laboratory.login()
    outside = tmp_path / "private"
    outside.write_bytes(b"private-test-content")
    target = shared_laboratory.options.scene_directory / "robot.urdf"
    target.unlink()
    target.symlink_to(outside)
    response = shared_laboratory.request(
        f"/scenes/{ShareOptions.SCENE_NAME}/robot.urdf"
    )
    assert response.status == 404
    assert outside.read_bytes() not in response.body


def test_control_proxy_rewrites_origin_without_forwarding_cookie(
    shared_laboratory: SharedLaboratory,
) -> None:
    """An authenticated same-origin command reaches only the local simulation."""
    shared_laboratory.login()
    response = shared_laboratory.request(
        RobotPhysicsRoute.MIX,
        method="POST",
        body=b"{}",
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/json",
        },
    )
    assert response.status == 200
    assert shared_laboratory.backend.requests == [
        ForwardedRequest(
            "POST",
            RobotPhysicsRoute.MIX,
            f"http://127.0.0.1:{shared_laboratory.options.backend_port}",
            None,
            b"{}",
        )
    ]


@pytest.mark.parametrize("origin", [None, "https://untrusted.example.test"])
def test_control_rejects_missing_or_foreign_origin(
    shared_laboratory: SharedLaboratory, origin: str | None
) -> None:
    """A session cookie cannot authorize a cross-origin mutation."""
    shared_laboratory.login()
    headers = {"Content-Type": "application/json"}
    if origin is not None:
        headers["Origin"] = origin
    response = shared_laboratory.request(
        RobotPhysicsRoute.RESET, method="POST", body=b"{}", headers=headers
    )
    assert response.status == 403
    assert shared_laboratory.backend.requests == []


def test_control_rejects_oversized_body(shared_laboratory: SharedLaboratory) -> None:
    """An excessive declared request size is rejected before proxying."""
    shared_laboratory.login()
    response = shared_laboratory.request(
        RobotPhysicsRoute.MIX,
        method="POST",
        body=b"{}",
        headers={
            "Origin": shared_laboratory.origin,
            "Content-Type": "application/json",
            "Content-Length": str(ShareOptions.MAX_BODY_BYTES + 1),
        },
    )
    assert response.status == 413
    assert shared_laboratory.backend.requests == []
