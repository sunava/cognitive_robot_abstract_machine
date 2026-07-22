#!/usr/bin/env python3
"""
Zero-dependency web server for the placement-pattern GUI (Anforderungsprofil
1).

Runs on a separate machine and is opened from the SIMATIC HMI panel's browser
via http://<ip>:<port>/. The web/geometry layer is pure standard library; only
the data source (recipes.py) needs the third-party ``psycopg`` PostgreSQL
driver (see requirements.txt).

    python3 app.py --host 0.0.0.0 --port 8000

API:
    GET /api/status                                    -> {demo: bool}
    GET /api/recipes                                   -> stored recipes
    GET /api/pattern?recipe=A&offset_x=0&offset_y=0    -> computed pattern
Static files are served from ./static (index.html etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from placement import compute_pattern
from recipes import RecipeSource, create_recipe_source

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
}


class PlacementServer(ThreadingHTTPServer):
    """
    HTTP server that owns the recipe source the request handlers read from.
    """

    def __init__(self, address: tuple[str, int], recipe_source: RecipeSource):
        super().__init__(address, Handler)
        self.recipe_source = recipe_source
        """
        The recipe data source shared by all requests.
        """


class Handler(BaseHTTPRequestHandler):
    server_version = "PlacementGUI/0.3"

    @property
    def recipe_source(self) -> RecipeSource:
        return self.server.recipe_source

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_db_error(self, exc):
        # The data source (PostgreSQL, see recipes.py) is unreachable or
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
            return self._send_json({"demo": self.recipe_source.is_demo})
        if path == "/api/recipes":
            return self._get_recipes()
        if path == "/api/pattern":
            return self._get_pattern(query)
        return self._send_static(path)

    def _get_recipes(self):
        try:
            recipes = self.recipe_source.list_recipes()
        except Exception as exc:
            return self._send_db_error(exc)
        return self._send_json([
            {"id": recipe.id, "name": recipe.name, "box": asdict(recipe.box),
             "part": asdict(recipe.part), "gap": recipe.gap}
            for recipe in recipes
        ])

    def _get_pattern(self, query):
        recipe_id = _first(query, "recipe")
        try:
            recipe = self.recipe_source.get_recipe(recipe_id)
        except Exception as exc:
            return self._send_db_error(exc)
        if recipe is None:
            return self._send_json(
                {"error": f"unknown recipe '{recipe_id}'"}, 404)
        offset_x = _to_float(_first(query, "offset_x"))
        offset_y = _to_float(_first(query, "offset_y"))
        return self._send_json(compute_pattern(recipe, offset_x, offset_y))

    def log_message(self, fmt, *args):  # quieter console
        pass


def _first(query: dict, key: str) -> str:
    return (query.get(key) or [""])[0]


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser(description="Placement-pattern GUI server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--demo", action="store_true",
        help="serve built-in sample recipes instead of querying PostgreSQL "
             "(no database / driver required)",
    )
    args = ap.parse_args()
    if args.demo:
        os.environ["PLACEMENT_DEMO"] = "1"

    recipe_source = create_recipe_source()
    httpd = PlacementServer((args.host, args.port), recipe_source)
    mode = "DEMO data (no database)" if recipe_source.is_demo else "PostgreSQL"
    print(f"Placement GUI on http://{args.host}:{args.port}/  [{mode}]  "
          f"(Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
