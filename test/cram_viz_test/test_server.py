"""
End-to-end tests of the HTTP server: static frontend, scenes and JSON API.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

# %% helpers


@pytest.fixture()
def server(fixture_scene: Path) -> str:
    """
    The real server on an ephemeral port, bound to the fixture scene.
    """
    from cram_viz import server as server_module

    httpd = server_module.make_server(0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://localhost:%d" % httpd.server_address[1]
    httpd.shutdown()


def get(url: str, timeout: int = 10) -> tuple[int, bytes]:
    """
    The status code and raw body of a GET request.
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read()


def get_json(url: str) -> dict:
    """
    The parsed JSON body of a successful GET request.
    """
    status, body = get(url)
    assert status == 200
    return json.loads(body)


# %% static frontend and scene bundles


class TestStatic:
    def test_index_is_served(self, server: str) -> None:
        """
        The frontend shell is served at /.
        """
        status, body = get(server + "/")
        assert status == 200
        assert b"CRAM Visualization" in body
        assert b'data-slot="left"' in body

    def test_static_responses_disable_caching(self, server: str) -> None:
        """
        Handler.end_headers() sends Cache-Control: no-cache on every response.
        """
        with urllib.request.urlopen(server + "/") as response:
            assert response.headers.get("Cache-Control") == "no-cache"

    def test_panel_scripts_are_served(self, server: str) -> None:
        """
        Every packaged panel script is reachable under its own path.
        """
        for path in (
            "/core/bus.js",
            "/core/registry.js",
            "/config.js",
            "/panels/robot_scene/panel.js",
            "/panels/eql/panel.js",
            "/panels/graph/panel.js",
            "/panels/graph/graph.js",
        ):
            status, _ = get(server + path)
            assert status == 200, path

    def test_scene_bundle_is_served_from_data_dir(self, server: str) -> None:
        """
        The fixture scene's bundle and index are served under /scenes/.
        """
        scene = get_json(server + "/scenes/fixture/scene.json")
        assert scene["name"] == "fixture"
        index = get_json(server + "/scenes/index.json")
        assert index["default"] == "fixture"

    def test_scene_path_traversal_is_blocked(self, server: str) -> None:
        """
        A ../..

        path escaping the scenes directory is rejected, not served.
        """
        request = urllib.request.Request(server + "/scenes/../../etc/passwd")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = response.status
        except urllib.error.HTTPError as err:
            status = err.code
        assert status in (403, 404)


# %% JSON API


class TestApi:
    def test_kb_overview(self, server: str) -> None:
        """
        GET /api/kb returns the knowledge-graph overview, including the milk node.
        """
        pytest.importorskip("krrood")
        payload = get_json(server + "/api/kb")
        assert payload["ok"]
        assert any(n["id"] == "milk" for n in payload["nodes"])

    def test_kb_views(self, server: str) -> None:
        """
        GET /api/kb/view?name= returns each of the knowledge/kinematics/plan/chart tabs.
        """
        pytest.importorskip("krrood")
        for name, expect_live in (
            ("kinematics", None),
            ("plan", "plan"),
            ("chart", "chart"),
        ):
            payload = get_json(server + "/api/kb/view?name=" + name)
            assert payload["ok"], name
            assert payload.get("live") == expect_live

    def test_kb_expand_drillable_node(self, server: str) -> None:
        """
        GET /api/kb/expand?node= returns the inside view of a drillable package node.
        """
        pytest.importorskip("krrood")
        payload = get_json(server + "/api/kb/expand?node=coraplex")
        assert payload["ok"]
        assert payload["crumb"] == "coraplex"

    def test_kb_expand_not_drillable_node(self, server: str) -> None:
        """
        A node with no inside view reports the documented "not drillable" error.
        """
        pytest.importorskip("krrood")
        payload = get_json(server + "/api/kb/expand?node=milk")
        assert payload == {"ok": False, "error": "not drillable"}

    def test_eql_query_roundtrip(self, server: str) -> None:
        """
        A well-formed EQL query executes and returns its matching entities.
        """
        pytest.importorskip("krrood")
        request = urllib.request.Request(
            server + "/api/eql",
            data=json.dumps(
                {"code": "the(entity(object).where(object.name == 'milk'))"}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        assert payload["ok"] and payload["count"] == 1

    def test_broken_query_returns_json_error(self, server: str) -> None:
        """
        A syntactically invalid EQL query is reported as a JSON error, not a 500.
        """
        pytest.importorskip("krrood")
        request = urllib.request.Request(
            server + "/api/eql",
            data=json.dumps({"code": "definitely not python ((("}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        assert payload["ok"] is False and "error" in payload

    def test_malformed_json_body_returns_json_error(self, server: str) -> None:
        """
        A POST body that isn't JSON at all is reported as a JSON error, not a crash.
        """
        pytest.importorskip("krrood")
        request = urllib.request.Request(
            server + "/api/eql",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
        assert payload["ok"] is False and "error" in payload

    def test_empty_body_is_reported_as_empty_query(self, server: str) -> None:
        """
        A POST with no body (Content-Length: 0) is treated as an empty query, not a
        crash.
        """
        pytest.importorskip("krrood")
        request = urllib.request.Request(server + "/api/eql", data=b"")
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
        assert payload == {"ok": False, "error": "empty query"}

    def test_unknown_post_endpoint_is_json_404(self, server: str) -> None:
        """
        A POST to an undefined endpoint is a JSON 404, not the static-file 404.
        """
        request = urllib.request.Request(server + "/api/nope", data=b"{}")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as err:
            status, body = err.code, err.read()
        assert status == 404
        assert json.loads(body)["ok"] is False


# %% the documented krrood-absent fallback


@pytest.fixture()
def server_without_eql(fixture_scene: Path) -> str:
    """
    A server instance with krrood forced unavailable, for testing the no-EQL fallback
    regardless of whether krrood actually happens to be installed in this environment.
    """
    from cram_viz import server as server_module

    httpd = server_module.make_server(0, kb_module=None)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://localhost:%d" % httpd.server_address[1]
    httpd.shutdown()


class TestApiWithoutEql:
    def test_get_routes_report_no_eql(self, server_without_eql: str) -> None:
        """
        Every read-only API route reports the documented no-EQL error, not a crash.
        """
        for path in ("/api/kb", "/api/kb/view", "/api/kb/expand?node=milk"):
            payload = get_json(server_without_eql + path)
            assert payload == {
                "ok": False,
                "error": "krrood/EQL not available in this environment",
            }

    def test_post_route_reports_no_eql(self, server_without_eql: str) -> None:
        """
        POST /api/eql reports the documented no-EQL error, not a crash.
        """
        request = urllib.request.Request(
            server_without_eql + "/api/eql",
            data=json.dumps({"code": "1"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
        assert payload == {
            "ok": False,
            "error": "krrood/EQL not available in this environment",
        }


# %% the console-script entrypoint


class TestMain:
    def test_main_parses_port_argument_and_serves(
        self, monkeypatch: pytest.MonkeyPatch, fixture_scene: Path
    ) -> None:
        """
        Main(argv) parses argv[0] as the port and starts serving on it.
        """
        from cram_viz import server as server_module

        calls = {}

        class FakeServer:
            def __enter__(self) -> "FakeServer":
                return self

            def __exit__(self, *exc_info: object) -> bool:
                return False

            def serve_forever(self) -> None:
                pass

        def fake_make_server(port: int = 0) -> FakeServer:
            calls["port"] = port
            return FakeServer()

        monkeypatch.setattr(server_module, "make_server", fake_make_server)
        server_module.main(["9999"])
        assert calls["port"] == 9999
