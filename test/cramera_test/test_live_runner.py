"""
Unit tests for the live runner: the library entry point and the demo wrapper CLI.
"""

from __future__ import annotations

import os
import runpy
import signal
import sys
from dataclasses import dataclass, field

from semantic_digital_twin.world import World
from typing_extensions import Any, Dict, List

from cramera.live import runner, visualization
from cramera.live.bridge import BRIDGE


@dataclass
class StartedVisualization:
    """
    A visualization recorder standing in for ``LiveVisualization``.
    """

    world: World
    port: int
    started: bool = False

    def start(self):
        self.started = True
        return self


class TestStart:
    def test_start_serves_the_given_world(self, monkeypatch):
        monkeypatch.setattr(visualization, "LiveVisualization", StartedVisualization)
        world = World()

        result = runner.start(world, port=1234)

        assert isinstance(result, StartedVisualization)
        assert result.world is world
        assert result.port == 1234
        assert result.started is True


class TestMain:
    def test_main_preselects_the_cramera_backend_and_runs_the_demo(self, monkeypatch):
        run_calls: List[Dict[str, Any]] = []
        # main() preselects the backend by writing the real environment, and monkeypatch
        # records an undo only for a variable that was there: setting it first, then
        # deleting it, leaves the test with none and still restores that afterwards
        monkeypatch.setenv(
            runner.VISUALIZATION_BACKEND_VARIABLE, "restored-on-teardown"
        )
        monkeypatch.delenv(runner.VISUALIZATION_BACKEND_VARIABLE)
        monkeypatch.setattr(sys, "argv", ["cramera-live", "/demos/demo.py"])
        monkeypatch.setattr(
            runpy,
            "run_path",
            lambda path, run_name: run_calls.append(
                {"path": path, "run_name": run_name}
            ),
        )
        monkeypatch.setattr(signal, "pause", lambda: None)

        runner.main()

        assert os.environ[runner.VISUALIZATION_BACKEND_VARIABLE] == "cramera"
        assert run_calls == [{"path": "/demos/demo.py", "run_name": "__main__"}]
        assert "/demos" in sys.path

    def test_main_keeps_an_explicit_backend_choice(self, monkeypatch):
        """
        ``CORAPLEX_VISUALIZATION=rerun cramera-live demo.py`` is a deliberate override
        and must survive.
        """
        monkeypatch.setenv(runner.VISUALIZATION_BACKEND_VARIABLE, "rerun")
        monkeypatch.setattr(sys, "argv", ["cramera-live", "/demos/demo.py"])
        monkeypatch.setattr(runpy, "run_path", lambda path, run_name: None)
        monkeypatch.setattr(signal, "pause", lambda: None)

        runner.main()

        assert os.environ[runner.VISUALIZATION_BACKEND_VARIABLE] == "rerun"


# %% what the runner does once the demo is over
class TestWaitingForInspection:
    """
    ``cramera-live`` keeps the process alive so the finished demo's world stays
    browsable.

    A demo that stopped its own viewer leaves nothing to browse, and waiting for it
    would leave a process behind that serves nobody.
    """

    def run_demo(self, monkeypatch) -> List[bool]:
        """
        Run ``main()`` on a demo that does nothing.

        :return: One entry per wait for Ctrl-C the runner performed.
        """
        waits: List[bool] = []
        monkeypatch.setattr(sys, "argv", ["cramera-live", "/demos/demo.py"])
        monkeypatch.setattr(runpy, "run_path", lambda path, run_name: None)
        monkeypatch.setattr(signal, "pause", lambda: waits.append(True))
        runner.main()
        return waits

    def test_a_still_served_world_is_kept_up(self, monkeypatch):
        monkeypatch.setattr(BRIDGE, "live_server", object())
        assert self.run_demo(monkeypatch) == [True]

    def test_a_demo_that_stopped_its_own_viewer_ends_the_run(self, monkeypatch):
        monkeypatch.setattr(BRIDGE, "live_server", None)
        assert self.run_demo(monkeypatch) == []
