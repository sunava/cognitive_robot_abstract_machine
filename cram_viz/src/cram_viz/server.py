"""
The cram_viz HTTP server: static frontend + JSON API.

Serves three things from one port (default 8711):

  * the packaged web frontend (``cram_viz/web`` — panels, vendored libs)
  * scene bundles from :func:`cram_viz.paths.scenes_dir` under ``/scenes/``
  * the JSON API the panels talk to:

      GET  /api/kb              the knowledge-graph overview payload
      GET  /api/kb/view?name=   one graph tab (knowledge/kinematics/plan/chart)
      GET  /api/kb/expand?node= drill-down subgraph for one node
      POST /api/eql             run an EQL query string

The API needs krrood (EQL). Without it the server still serves the viewer and
answers API calls with ``{"ok": false, "error": ...}`` so the frontend can say
why the knowledge panel is empty.

    cram-viz            # console script
    python -m cram_viz.server [port]
"""

from __future__ import annotations

import http.server
import json
import logging
import mimetypes
import os
import socketserver
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

from cram_viz import paths

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8711

try:
    import krrood  # noqa: F401  (the EQL engine)

    from cram_viz import kb as _kb_module
except ImportError:  # pragma: no cover - depends on the environment
    _kb_module = None
    logger.warning("krrood not importable — serving the viewer without the EQL API")
except Exception:  # pragma: no cover - a broken KB should not kill the viewer
    _kb_module = None
    logger.exception(
        "unexpected error loading cram_viz.kb — serving the viewer without the EQL API"
    )


def _no_eql_error() -> dict[str, Any]:
    """
    The standard error payload for any API route when krrood isn't importable.
    """
    return {"ok": False, "error": "krrood/EQL not available in this environment"}


def _exception_payload(exception: Exception) -> dict[str, Any]:
    """
    The standard error payload for an API route that raised.
    """
    return {"ok": False, "error": "%s: %s" % (type(exception).__name__, exception)}


class Server(socketserver.ThreadingTCPServer):
    """
    A :class:`~socketserver.ThreadingTCPServer` carrying the EQL dependencies
    every :class:`Handler` needs, instead of module-level globals.
    """

    allow_reuse_address = True

    def __init__(
        self, address: tuple[str, int], handler_class: type[Handler], kb_module: Any
    ) -> None:
        """
        :param kb_module: the resolved ``cram_viz.kb`` module, or ``None`` if
            krrood isn't importable in this environment.
        """
        super().__init__(address, handler_class)
        self.kb_module = kb_module
        #: krrood's SymbolGraph singleton is not threadsafe; queries are serialized
        self.eql_lock = threading.Lock()


class Handler(http.server.SimpleHTTPRequestHandler):
    """
    Static files from the packaged web root, plus the JSON API routes.
    """

    if TYPE_CHECKING:
        server: Server

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Bind the handler to the packaged web root as its static-file directory.
        """
        super().__init__(*args, directory=str(paths.WEB_ROOT), **kwargs)

    def end_headers(self) -> None:
        """
        Disable caching so a rebuilt scene/frontend is never served stale.
        """
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """
        Route the per-request access log through logging.
        """
        logger.info("  " + format, *args)

    # %% helpers -----------------------------------------------------------------
    def _json(self, payload: Any, code: int = 200) -> None:
        """
        Send a payload as JSON with the given status code.
        """
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query(self) -> dict[str, list[str]]:
        """
        The parsed query-string parameters of the current request.
        """
        return parse_qs(urlparse(self.path).query)

    def _guarded(self, function: Callable[[], Any]) -> None:
        """
        Run an API handler; report exceptions as a JSON error payload.
        """
        if self.server.kb_module is None:
            return self._json(_no_eql_error())
        try:
            return self._json(function())
        except Exception as exception:
            return self._json(_exception_payload(exception))

    # %% scene bundles (generated data, lives outside the package) ----------------
    def _serve_scene_file(self, url_path: str) -> None:
        """
        Serve one file of a scene bundle, with path-traversal protection.
        """
        relative_path = unquote(url_path[len("/scenes/") :])
        base = paths.scenes_dir().resolve()
        target = (base / relative_path).resolve()
        if not str(target).startswith(str(base) + os.sep) and target != base:
            self.send_response(403)
            self.end_headers()
            return
        if not target.is_file():
            self.send_response(404)
            self.end_headers()
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # %% routes --------------------------------------------------------------------
    def do_GET(self) -> None:
        """
        Route static files, scene bundles and the read-only API.
        """
        route = self.path.split("?")[0]
        if route.startswith("/scenes/"):
            return self._serve_scene_file(route)
        if route == "/api/kb":
            return self._guarded(lambda: self.server.kb_module.graph_payload())
        if route == "/api/kb/view":
            name = (self._query().get("name") or ["knowledge"])[0]
            return self._guarded(lambda: self.server.kb_module.view_payload(name))
        if route == "/api/kb/expand":
            node = (self._query().get("node") or [""])[0]

            def expand() -> dict[str, Any]:
                """
                The node's subgraph, or a "not drillable" error if it has none.
                """
                payload = self.server.kb_module.expand_node(node)
                return payload if payload else {"ok": False, "error": "not drillable"}

            return self._guarded(expand)
        return super().do_GET()

    def do_POST(self) -> None:
        """
        Execute an EQL query (the only write-ish endpoint).
        """
        if self.path.split("?")[0] != "/api/eql":
            return self._json({"ok": False, "error": "unknown endpoint"}, 404)

        def run_eql_query() -> dict[str, Any]:
            """
            Parse the request body and execute its EQL code under the lock.
            """
            length = int(self.headers.get("Content-Length") or 0)
            request_body = json.loads(self.rfile.read(length) or b"{}")
            code = (request_body.get("code") or "").strip()
            if not code:
                return {"ok": False, "error": "empty query"}
            with self.server.eql_lock:
                return self.server.kb_module.run_query(code)

        return self._guarded(run_eql_query)


def make_server(port: int = 0, kb_module: Any = _kb_module) -> Server:
    """A ready-to-serve Server (port 0 = ephemeral, for tests)."""
    return Server(("127.0.0.1", port), Handler, kb_module)


def main(argv: list[str] | None = None) -> None:
    """
    ``cram-viz`` — serve the viewer, the scenes and the JSON API.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argv = sys.argv[1:] if argv is None else argv
    port = int(argv[0]) if argv else DEFAULT_PORT
    if _kb_module is not None:  # build the KB once, before the first query
        _kb_module.get_kb()
    with make_server(port) as httpd:
        eql = (
            "EQL ready (krrood)" if _kb_module is not None else "EQL unavailable — static only"
        )
        scenes = paths.scenes_dir()
        logger.info("cram_viz running at http://localhost:%d/ (%s)", port, eql)
        logger.info(
            "scene bundles: %s%s",
            scenes,
            "" if Path(scenes).is_dir() else "  (missing — run cram-viz-onboard)",
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
