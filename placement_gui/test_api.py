"""
End-to-end tests for the JSON API in ``app.py``, run against a live server with
the demo catalog.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from app import PlacementServer
from catalog import DemoCatalog


@pytest.fixture
def api_base_url() -> Iterator[str]:
    server = PlacementServer(("127.0.0.1", 0), DemoCatalog())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get_json(url: str) -> dict | list:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def post_json(url: str, payload: dict):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request) as response:
        return response.status, json.load(response)


class TestShapeAndPatternListing:
    def test_shapes_endpoint_serves_the_catalog(self, api_base_url):
        shapes = get_json(f"{api_base_url}/api/shapes")
        assert shapes
        assert {"id", "name", "width", "height"} <= set(shapes[0])

    def test_patterns_endpoint_serves_stored_patterns(self, api_base_url):
        patterns = get_json(f"{api_base_url}/api/patterns")
        assert patterns
        assert {"id", "name", "box", "shape_id", "rows", "columns",
                "gap"} <= set(patterns[0])


class TestPlacementComputation:
    def test_placements_for_a_stored_pattern(self, api_base_url):
        pattern = get_json(f"{api_base_url}/api/patterns")[0]
        result = get_json(
            f"{api_base_url}/api/placements?pattern={pattern['id']}")
        assert result["count"] == len(result["placements"]) > 0

    def test_preview_computes_placements_for_unsaved_patterns(
            self, api_base_url):
        shape = get_json(f"{api_base_url}/api/shapes")[0]
        result = get_json(
            f"{api_base_url}/api/preview?shape={shape['id']}"
            f"&box_width=600&box_height=400&rows=2&columns=0&gap=12")
        assert result["rows"] == 2
        assert result["count"] == len(result["placements"])

    def test_unknown_pattern_yields_404(self, api_base_url):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            get_json(f"{api_base_url}/api/placements?pattern=nope")
        assert excinfo.value.code == 404


class TestPatternSaving:
    def test_saved_pattern_appears_in_the_listing(self, api_base_url):
        shape = get_json(f"{api_base_url}/api/shapes")[0]
        status, saved = post_json(f"{api_base_url}/api/patterns", {
            "name": "Operator pattern",
            "shape_id": shape["id"],
            "box": {"width": 500, "height": 350},
            "rows": 3, "columns": 0, "gap": 8,
        })
        assert status == 201
        assert saved["id"] == "operator-pattern"
        pattern_ids = [p["id"]
                       for p in get_json(f"{api_base_url}/api/patterns")]
        assert "operator-pattern" in pattern_ids
        result = get_json(
            f"{api_base_url}/api/placements?pattern=operator-pattern")
        assert result["rows"] == 3

    def test_saving_without_a_name_is_rejected(self, api_base_url):
        shape = get_json(f"{api_base_url}/api/shapes")[0]
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            post_json(f"{api_base_url}/api/patterns", {
                "name": "", "shape_id": shape["id"],
                "box": {"width": 500, "height": 350},
            })
        assert excinfo.value.code == 400

    def test_saving_with_an_unknown_shape_is_rejected(self, api_base_url):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            post_json(f"{api_base_url}/api/patterns", {
                "name": "Ghost", "shape_id": "does-not-exist",
                "box": {"width": 500, "height": 350},
            })
        assert excinfo.value.code == 400


class TestOrientationAndFlips:
    def test_preview_honours_orientation_and_flipped_indices(
            self, api_base_url):
        result = get_json(
            f"{api_base_url}/api/preview?shape=bracket"
            f"&box_width=600&box_height=400&rows=0&columns=0&gap=15"
            f"&orientation=rotated&flipped=0,2")
        assert result["orientation"] == "rotated"
        first = result["placements"][0]
        assert (first["width"], first["height"]) == (80.0, 120.0)
        assert first["flipped"] is True
        assert result["placements"][1]["flipped"] is False

    def test_preview_with_unknown_orientation_is_rejected(self, api_base_url):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            get_json(f"{api_base_url}/api/preview?shape=bracket"
                     f"&orientation=diagonal")
        assert excinfo.value.code == 400

    def test_saved_orientation_and_flips_survive_the_roundtrip(
            self, api_base_url):
        status, saved = post_json(f"{api_base_url}/api/patterns", {
            "name": "Rotated brackets",
            "shape_id": "bracket",
            "box": {"width": 600, "height": 400},
            "gap": 15,
            "orientation": "rotated",
            "flipped_placements": [1, 3],
        })
        assert status == 201
        assert saved["orientation"] == "rotated"
        assert saved["flipped_placements"] == [1, 3]
        result = get_json(
            f"{api_base_url}/api/placements?pattern={saved['id']}")
        assert result["orientation"] == "rotated"
        assert result["placements"][1]["flipped"] is True
