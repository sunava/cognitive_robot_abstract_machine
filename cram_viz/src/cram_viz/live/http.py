"""
HTTP endpoints of the live bridge (default port 8765).

::

    GET /info    {running, robot, objects, plan, chart, seq}
    GET /state   {seq, frames: {prefixed_joint: pos}, base: pose7,
                  objects: {mesh_key: pose7}}
    GET /objects geometry catalog (mesh served via /mesh?key=)
    GET /plan    {signature, nodes: [{id, parent, kind, label, status, derived}]}
    GET /chart   {signature, title, nodes: [{id, parent, name, class, life, observation}],
                  edges: [{from, to, kind}]}
    POST /move   queue an object move (applied on the simulation thread)

Handlers only ever read finished snapshot dicts — never the world (see
:mod:`cram_viz.live.hooks`).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from typing_extensions import Any

from cram_viz.live.bridge import Bridge, MoveRequest, MoveRequestError

logger = logging.getLogger(__name__)

DEFAULT_PORT = int(os.environ.get("LIVE_VIZ_PORT", "8765"))

#: read-only snapshot routes, dispatched by exact path match
_GET_ROUTES = ("/state", "/plan", "/chart", "/objects", "/mesh", "/info")


# %% server ------------------------------------------------------------------------
class BridgeHTTPServer(ThreadingHTTPServer):
    """
    HTTP server that hands each request handler the bridge it serves.
    """

    def __init__(
        self, server_address: tuple, handler_class: type, bridge: Bridge
    ) -> None:
        super().__init__(server_address, handler_class)
        self.bridge = bridge
        """
        The bridge instance this server's handlers read and write.
        """


class BridgeRequestHandler(BaseHTTPRequestHandler):
    """
    Serves the bridge's snapshots and accepts viewer moves.
    """

    server: BridgeHTTPServer
    """
    The owning server, carrying the :class:`Bridge` this handler reads and writes.
    """

    @property
    def bridge(self) -> Bridge:
        """
        The bridge instance backing this connection, from the owning server.
        """
        return self.server.bridge

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        """
        Send a JSON payload with the CORS headers the viewer needs.
        """
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # %% snapshot routes -------------------------------------------------------
    def do_GET(self) -> None:
        """
        Route the read-only snapshot endpoints.
        """
        path = urllib.parse.urlparse(self.path).path
        if path not in _GET_ROUTES:
            self.send_response(404)
            self.end_headers()
            return
        if path == "/state":
            return self._send_json(self.bridge.get_state())
        if path == "/plan":
            return self._send_json(self.bridge.get_plan())
        if path == "/chart":
            return self._send_json(self.bridge.get_chart())
        if path == "/objects":
            return self._send_json({"objects": self.bridge.get_objects()})
        if path == "/mesh":
            return self._send_mesh()
        return self._send_json(self.bridge.get_info())

    def _send_mesh(self) -> None:
        """
        Serve one object's mesh file (plain file IO, no world access).
        """
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        key = (query.get("key") or [""])[0]
        path = self.bridge.get_mesh_path(key)
        if not path or not Path(path).is_file():
            self.send_response(404)
            self.end_headers()
            return
        data = Path(path).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # %% move handling ----------------------------------------------------------
    def do_POST(self) -> None:
        """
        Queue an object move requested by the viewer.
        """
        if urllib.parse.urlparse(self.path).path != "/move":
            self.send_response(404)
            self.end_headers()
            return
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            move = MoveRequest.from_payload(payload)
        except MoveRequestError as error:
            return self._send_json({"ok": False, "error": str(error)}, status=400)
        self.bridge.queue_move(move)
        return self._send_json({"ok": True})

    def _read_json_body(self) -> dict[str, Any] | None:
        """
        Read and parse the request body as JSON, sending a 400 on failure.

        :return: The parsed payload, or ``None`` if a 400 response was already sent.
        """
        content_length = self.headers.get("Content-Length")
        try:
            length = int(content_length) if content_length is not None else 0
        except ValueError:
            self._send_json(
                {"ok": False, "error": "invalid Content-Length"}, status=400
            )
            return None
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            self._send_json({"ok": False, "error": str(error)}, status=400)
            return None

    # %% cors and logging --------------------------------------------------------
    def do_OPTIONS(self) -> None:
        """
        CORS preflight for the viewer's cross-origin POSTs.
        """
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()

    def log_message(self, message_format: str, *args: Any) -> None:
        """
        Route the per-request access log to debug (15 Hz polling is noisy).
        """
        logger.debug(message_format, *args)


# %% startup ------------------------------------------------------------------------
def serve(bridge: Bridge, port: int = DEFAULT_PORT) -> BridgeHTTPServer:
    """
    Start the bridge's HTTP server on a daemon thread.

    :param bridge: Bridge instance the server's handlers read and write.
    :param port: Port to listen on (all interfaces).
    :return: The running server.
    """
    server = BridgeHTTPServer(("0.0.0.0", port), BridgeRequestHandler, bridge)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
