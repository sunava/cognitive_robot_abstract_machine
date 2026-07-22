"""
End-to-end tests for the JSON API in ``app.py``, run against a live server with
the demo recipe source.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from app import PlacementServer
from recipes import DemoRecipes


@pytest.fixture
def api_base_url() -> Iterator[str]:
    server = PlacementServer(("127.0.0.1", 0), DemoRecipes())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get_json(url: str) -> dict | list:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


class TestRecipeListing:
    def test_recipes_endpoint_serves_the_stored_recipes(self, api_base_url):
        recipes = get_json(f"{api_base_url}/api/recipes")
        assert len(recipes) == 2
        assert {"id", "name", "box", "part", "gap"} <= set(recipes[0])

    def test_status_reports_the_demo_backend(self, api_base_url):
        assert get_json(f"{api_base_url}/api/status") == {"demo": True}


class TestPatternComputation:
    def test_pattern_for_a_stored_recipe(self, api_base_url):
        recipe = get_json(f"{api_base_url}/api/recipes")[0]
        result = get_json(
            f"{api_base_url}/api/pattern?recipe={recipe['id']}")
        assert result["count"] == len(result["placements"]) > 0

    def test_offset_is_applied_and_clamped(self, api_base_url):
        recipe = get_json(f"{api_base_url}/api/recipes")[0]
        result = get_json(
            f"{api_base_url}/api/pattern?recipe={recipe['id']}"
            f"&offset_x=10000&offset_y=10000")
        limits = result["offset_range"]
        assert result["offset"] == {"x": limits["max_x"], "y": limits["max_y"]}

    def test_unknown_recipe_yields_404(self, api_base_url):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            get_json(f"{api_base_url}/api/pattern?recipe=nope")
        assert excinfo.value.code == 404
