#!/usr/bin/env python3
"""
Zero-dependency web server for the placement-pattern GUI.

Runs on a separate machine and is opened from the SIMATIC HMI panel's browser
via http://<ip>:<port>/. The web/geometry layer is pure standard library; only
the data source (catalog.py) needs the third-party ``psycopg`` PostgreSQL
driver (see requirements.txt).

    python3 app.py --host 0.0.0.0 --port 8000

API:
    GET  /api/status                                  -> {demo: bool}
    GET  /api/shapes                                  -> shape catalog
    GET  /api/patterns                                -> stored patterns
    POST /api/patterns                                -> save a pattern
    GET  /api/placements?pattern=ID&offset_x=&offset_y= -> computed placements
    GET  /api/preview?shape=ID&box_width=&box_height=&rows=&columns=&gap=
                                                      -> placements for an
                                                         unsaved pattern
Static files are served from ./static (index.html etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from catalog import Catalog, UnknownShapeError, create_catalog
from placement import Pattern, Size, compute_placements

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
}

_PREVIEW_PATTERN_ID = "preview"


class PatternRequestError(Exception):
    """
    Raised when a pattern request carries invalid or incomplete data; maps to
    HTTP 400.
    """


class PlacementServer(ThreadingHTTPServer):
    """
    HTTP server that owns the catalog the request handlers read from.
    """

    def __init__(self, address: tuple[str, int], catalog: Catalog):
        super().__init__(address, Handler)
        self.catalog = catalog
        """
        The shape/pattern data source shared by all requests.
        """


class Handler(BaseHTTPRequestHandler):
    server_version = "PlacementGUI/0.2"

    @property
    def catalog(self) -> Catalog:
        return self.server.catalog

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_db_error(self, exc):
        # The data source (PostgreSQL, see catalog.py) is unreachable or
        # failed. Surface a clean 503 to the UI instead of dropping the
        # connection.
        print(f"data source error: {exc}", file=sys.stderr)
        return self._send_json(
            {"error": "data source unavailable", "detail": str(exc)}, 503
        )

    def _send_static(self, path):
        # Default document.
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.normpath(os.path.join(STATIC_DIR, rel))
        # Prevent path traversal outside STATIC_DIR.
        if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
            self.send_error(404, "Not found")
            return
        ext = os.path.splitext(full)[1].lower()
        ctype = _CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/status":
            return self._send_json({"demo": self.catalog.is_demo})
        if path == "/api/shapes":
            return self._get_shapes()
        if path == "/api/patterns":
            return self._get_patterns()
        if path == "/api/placements":
            return self._get_placements(query)
        if path == "/api/preview":
            return self._get_preview(query)
        return self._send_static(path)

    def do_POST(self):
        if urlparse(self.path).path == "/api/patterns":
            return self._post_pattern()
        self.send_error(404, "Not found")

    def _get_shapes(self):
        try:
            shapes = self.catalog.list_shapes()
        except Exception as exc:
            return self._send_db_error(exc)
        return self._send_json([
            {"id": shape.id, "name": shape.name, **asdict(shape.size)}
            for shape in shapes
        ])

    def _get_patterns(self):
        try:
            patterns = self.catalog.list_patterns()
        except Exception as exc:
            return self._send_db_error(exc)
        return self._send_json([_pattern_to_json(p) for p in patterns])

    def _get_placements(self, query):
        pattern_id = _first(query, "pattern")
        try:
            pattern = self.catalog.get_pattern(pattern_id)
        except Exception as exc:
            return self._send_db_error(exc)
        if pattern is None:
            return self._send_json(
                {"error": f"unknown pattern '{pattern_id}'"}, 404)
        offset_x = _to_float(_first(query, "offset_x"))
        offset_y = _to_float(_first(query, "offset_y"))
        return self._send_json(compute_placements(pattern, offset_x, offset_y))

    def _get_preview(self, query):
        """
        Placements for a pattern that is being edited and not saved yet, so the
        editor tab and the placement view share one geometry code path.
        """
        shape_id = _first(query, "shape")
        try:
            shape = self.catalog.get_shape(shape_id)
        except Exception as exc:
            return self._send_db_error(exc)
        if shape is None:
            return self._send_json({"error": f"unknown shape '{shape_id}'"},
                                   404)
        pattern = Pattern(
            id=_PREVIEW_PATTERN_ID,
            name=_first(query, "name") or "Preview",
            box=Size(_to_float(_first(query, "box_width"), 600.0),
                     _to_float(_first(query, "box_height"), 400.0)),
            shape=shape,
            rows=_to_int(_first(query, "rows")),
            columns=_to_int(_first(query, "columns")),
            gap=_to_float(_first(query, "gap"), 10.0),
        )
        return self._send_json(compute_placements(pattern))

    def _post_pattern(self):
        length = _to_int(self.headers.get("Content-Length"), 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._send_json({"error": "invalid JSON body"}, 400)
        try:
            shape_id = str(payload.get("shape_id", ""))
            shape = self.catalog.get_shape(shape_id)
            if shape is None:
                raise UnknownShapeError(shape_id)
            pattern = _pattern_from_json(payload, shape)
            self.catalog.save_pattern(pattern)
        except (PatternRequestError, UnknownShapeError) as exc:
            return self._send_json({"error": str(exc)}, 400)
        except Exception as exc:
            return self._send_db_error(exc)
        return self._send_json(_pattern_to_json(pattern), 201)

    def log_message(self, fmt, *args):  # quieter console
        pass


def _pattern_to_json(pattern: Pattern) -> dict:
    return {
        "id": pattern.id,
        "name": pattern.name,
        "box": asdict(pattern.box),
        "shape_id": pattern.shape.id,
        "shape_name": pattern.shape.name,
        "rows": pattern.rows,
        "columns": pattern.columns,
        "gap": pattern.gap,
    }


def _pattern_from_json(payload: dict, shape) -> Pattern:
    """
    Build a pattern from a save request; the id is derived from the name when
    absent so saving under the same name updates the same pattern.
    """
    name = str(payload.get("name", "")).strip()
    if not name:
        raise PatternRequestError("pattern name must not be empty")
    box = payload.get("box") or {}
    box_width = _to_float(box.get("width"), 0.0)
    box_height = _to_float(box.get("height"), 0.0)
    if box_width <= 0 or box_height <= 0:
        raise PatternRequestError("box width and height must be positive")
    pattern_id = str(payload.get("id") or _slug(name))
    if pattern_id == _PREVIEW_PATTERN_ID:
        raise PatternRequestError(
            f"'{_PREVIEW_PATTERN_ID}' is a reserved pattern id")
    return Pattern(
        id=pattern_id,
        name=name,
        box=Size(box_width, box_height),
        shape=shape,
        rows=max(_to_int(payload.get("rows")), 0),
        columns=max(_to_int(payload.get("columns")), 0),
        gap=max(_to_float(payload.get("gap"), 10.0), 0.0),
    )


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "pattern"


def _first(query: dict, key: str) -> str:
    return (query.get(key) or [""])[0]


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser(description="Placement-pattern GUI server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--demo", action="store_true",
        help="serve built-in sample shapes/patterns instead of querying "
             "PostgreSQL (no database / driver required)",
    )
    args = ap.parse_args()
    if args.demo:
        os.environ["PLACEMENT_DEMO"] = "1"

    catalog = create_catalog()
    httpd = PlacementServer((args.host, args.port), catalog)
    mode = "DEMO data (no database)" if catalog.is_demo else "PostgreSQL"
    print(f"Placement GUI on http://{args.host}:{args.port}/  [{mode}]  "
          f"(Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
