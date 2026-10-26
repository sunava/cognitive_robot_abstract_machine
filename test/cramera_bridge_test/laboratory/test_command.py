"""
The optional transfer's plain-data command boundary, without importing CRAM.
"""

import sys

import pytest

from cramera_bridge.laboratory import command
from cramera_protocol.laboratory.runs import TransferOption


def test_dependency_discovery_names_missing_modules_without_importing(monkeypatch):
    present = {"semantic_digital_twin", "coraplex"}
    monkeypatch.setattr(
        command.importlib.util,
        "find_spec",
        lambda name: object() if name in present else None,
    )
    before = set(sys.modules)
    assert command.missing_libraries() == ["giskardpy", "krrood", "segmind"]
    assert set(sys.modules) == before


def test_a_missing_workspace_stops_before_loading_native_transfer(monkeypatch, capsys):
    monkeypatch.setattr(command, "missing_libraries", lambda: ["coraplex"])
    called = []
    monkeypatch.setattr(
        command, "run_transfer", lambda *arguments: called.append(arguments)
    )
    with pytest.raises(SystemExit) as stopped:
        command.main([TransferOption.BUNDLE, "/laboratory"])
    assert stopped.value.code == 2
    assert "CRAM workspace: coraplex" in capsys.readouterr().err
    assert called == []


def test_the_command_passes_the_bundle_path_and_task_options(monkeypatch, tmp_path):
    monkeypatch.setattr(command, "missing_libraries", lambda: [])
    called = []
    monkeypatch.setattr(
        command, "run_transfer", lambda *arguments: called.append(arguments)
    )
    command.main([TransferOption.BUNDLE, str(tmp_path), "--port", "0"])
    assert called == [(str(tmp_path), ["--port", "0"])]


def test_the_command_requires_an_explicit_bundle(monkeypatch):
    monkeypatch.setattr(command, "missing_libraries", lambda: [])
    with pytest.raises(SystemExit) as stopped:
        command.main([])
    assert stopped.value.code == 2
