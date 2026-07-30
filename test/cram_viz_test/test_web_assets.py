"""
Consistency checks of the packaged frontend, plus the node-based JS tests.

The asset checks keep the panel architecture honest: every script index.html includes
must exist, every panel id in config.js must be defined by an included panel script, and
panels must not reach into each other's DOM.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from cram_viz.paths import WEB_ROOT

JS_DIR = Path(__file__).parent / "js"


def read(rel: str) -> str:
    """
    The text content of a file under the packaged web root.
    """
    return (WEB_ROOT / rel).read_text(encoding="utf-8")


# %% asset consistency checks


class TestAssetConsistency:
    def test_every_included_script_exists(self) -> None:
        """
        Every <script src=...> in index.html points at a file that exists.
        """
        for src in re.findall(r'<script src="([^"]+)"', read("index.html")):
            assert (WEB_ROOT / src).is_file(), src

    def test_stylesheet_and_slots_exist(self) -> None:
        """
        Every stylesheet link resolves, and both layout slots are present.
        """
        html = read("index.html")
        for href in re.findall(r'<link rel="stylesheet" href="([^"]+)"', html):
            assert (WEB_ROOT / href).is_file(), href
        assert 'data-slot="left"' in html and 'data-slot="right"' in html

    def test_configured_panels_are_defined(self) -> None:
        """
        Every panel id referenced in config.js's layout is Panels.define()d somewhere.
        """
        config = read("config.js")
        configured = set(re.findall(r"'([\w-]+)'", config.split("layout", 1)[1]))
        defined = set()
        for panel_js in WEB_ROOT.glob("panels/*/panel.js"):
            defined |= set(
                re.findall(r"Panels\.define\('([\w-]+)'", panel_js.read_text())
            )
        assert configured <= defined, configured - defined

    def test_panels_do_not_reach_into_each_other(self) -> None:
        """
        A panel may only query its own root; document.getElementById would recouple
        panels.
        """
        # a panel may only query its own root; document.getElementById on
        # another panel's elements would silently couple them again
        for panel_js in WEB_ROOT.glob("panels/*/panel.js"):
            assert "document.getElementById" not in panel_js.read_text(), panel_js.name

    def test_no_stale_static_urls(self) -> None:
        """
        No panel references the old repo's /static/ URL prefix.
        """
        # the old repo served under /static/; the packaged app is rooted at /
        for panel_js in WEB_ROOT.glob("panels/*/panel.js"):
            assert "'static/" not in panel_js.read_text(), panel_js.name


# %% node-based JS unit tests


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
class TestJsUnits:
    def run_node(self, name: str) -> None:
        """
        Run a node:test file under test/cram_viz_test/js and fail loudly on a non-zero
        exit.
        """
        result = subprocess.run(
            ["node", "--test", str(JS_DIR / name)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_bus_and_registry(self) -> None:
        """
        core/bus.js and core/registry.js behave per their node:test suite.
        """
        self.run_node("test_bus_registry.js")

    def test_graph_status_rendering(self) -> None:
        """
        panels/graph/graph.js behaves per its node:test suite.
        """
        self.run_node("test_graph_status.js")

    def test_eql_panel(self) -> None:
        """
        panels/eql/panel.js behaves per its node:test suite.
        """
        self.run_node("test_eql_panel.js")

    def test_graph_panel(self) -> None:
        """
        panels/graph/panel.js behaves per its node:test suite.
        """
        self.run_node("test_graph_panel.js")

    def test_robot_scene_panel(self) -> None:
        """
        panels/robot_scene's extracted modules behave per their node:test suite.
        """
        self.run_node("test_robot_scene_panel.js")
