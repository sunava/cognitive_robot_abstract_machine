"""
Tests for the Plan-Builder demo process: what it writes, what it answers about how it
went, and its refusal to start into a bridge port another demo still holds.
"""

import socket

import pytest

from cramera import paths
from cramera.plan_scaffold import (
    BridgePortTaken,
    PlanScaffold,
    ScaffoldField,
    bridge_port_in_use,
)

DEMO_CODE = "print('a generated demo')\n"


def stub_launcher(directory, script):
    """
    An executable standing in for ``cramera-live``, which runs ``script`` instead of a
    demo.

    :param directory: Directory the stub is written to.
    :param script: Shell body the stub runs, with the demo path as ``$1``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stub = directory / paths.ConsoleScript.LIVE_DEMO.value
    stub.write_text("#!/bin/sh\n" + script)
    stub.chmod(0o755)
    return stub


# %% the bridge port


class TestBridgePortInUse:
    def test_a_bound_port_is_in_use(self):
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            assert bridge_port_in_use(taken.getsockname()[1]) is True

    def test_a_port_nobody_listens_on_is_free(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free = probe.getsockname()[1]
        assert bridge_port_in_use(free) is False


# %% launching


class TestLaunch:
    def test_the_generated_demo_is_written_where_it_is_run_from(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            paths,
            "console_script",
            lambda script: stub_launcher(tmp_path / "bin", "exit 0"),
        )
        monkeypatch.setattr(
            "cramera.plan_scaffold.bridge_port_in_use", lambda port: False
        )
        scaffold = PlanScaffold.launch(DEMO_CODE, tmp_path / "generated")
        scaffold.process.wait(timeout=10)

        assert scaffold.demo_path.read_text() == DEMO_CODE
        assert scaffold.demo_path.parent == tmp_path / "generated"

    def test_a_demo_cannot_start_into_a_port_another_one_holds(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            paths,
            "console_script",
            lambda script: stub_launcher(tmp_path / "bin", "exit 0"),
        )
        monkeypatch.setattr(
            "cramera.plan_scaffold.bridge_port_in_use", lambda port: True
        )

        with pytest.raises(BridgePortTaken):
            PlanScaffold.launch(DEMO_CODE, tmp_path / "generated")
        assert not (tmp_path / "generated").exists()


# %% how the demo went


class TestState:
    def scaffold_that(self, monkeypatch, tmp_path, script):
        """
        A launched scaffold whose stub launcher runs ``script``.

        :param script: Shell body the stub runs.
        """
        monkeypatch.setattr(
            paths, "console_script", lambda s: stub_launcher(tmp_path / "bin", script)
        )
        monkeypatch.setattr(
            "cramera.plan_scaffold.bridge_port_in_use", lambda port: False
        )
        return PlanScaffold.launch(DEMO_CODE, tmp_path / "generated")

    def test_a_demo_that_died_reports_its_code_and_its_last_words(
        self, monkeypatch, tmp_path
    ):
        scaffold = self.scaffold_that(
            monkeypatch, tmp_path, "echo 'Address already in use' >&2\nexit 3\n"
        )
        scaffold.process.wait(timeout=10)
        state = scaffold.state()

        assert state[ScaffoldField.RUNNING] is False
        assert state[ScaffoldField.EXIT_CODE] == 3
        assert "Address already in use" in state[ScaffoldField.OUTPUT]

    def test_a_running_demo_reports_no_exit_code(self, monkeypatch, tmp_path):
        scaffold = self.scaffold_that(monkeypatch, tmp_path, "sleep 30\n")
        state = scaffold.state()
        scaffold.stop()

        assert state[ScaffoldField.RUNNING] is True
        assert state[ScaffoldField.EXIT_CODE] is None

    def test_stopping_ends_the_demo(self, monkeypatch, tmp_path):
        scaffold = self.scaffold_that(monkeypatch, tmp_path, "sleep 30\n")
        scaffold.stop()

        assert scaffold.is_running() is False

    def test_only_the_end_of_a_long_run_is_reported(self, monkeypatch, tmp_path):
        scaffold = self.scaffold_that(
            monkeypatch, tmp_path, "for i in $(seq 1 200); do echo line $i; done\n"
        )
        scaffold.process.wait(timeout=10)
        reported = scaffold.recent_output().splitlines()

        assert len(reported) == PlanScaffold.REPORTED_LINES
        assert reported[-1] == "line 200"
