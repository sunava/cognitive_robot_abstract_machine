"""
Fixed laboratory executions are observable and cannot accept browser code.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import sys
import urllib.error
import urllib.request

import pytest

from cramera import laboratory_runs
from cramera.laboratory_runs import LaboratoryRun, RunState, RunField, LaboratoryRoute

from .test_server import server, get_json, post


# %% controlled process lifecycle
@dataclass
class RunningTransfer:
    """
    A child whose exit is controlled without starting a robot simulation.
    """

    returncode: int | None = None
    """
    The exit code observed by process polling.
    """

    def poll(self) -> int | None:
        """
        Report completion only when the test makes the child exit.
        """
        return self.returncode


@dataclass
class TransferLaunches:
    """
    Capture the fixed command and retain its controlled child.
    """

    child: RunningTransfer = field(default_factory=RunningTransfer)
    """
    The sole child produced by this launch recorder.
    """

    commands: list[list[str]] = field(default_factory=list)
    """
    Argument arrays passed to the operating system.
    """

    def __call__(self, command: list[str], **options) -> RunningTransfer:
        """
        Record process arguments without executing a subprocess.
        """
        self.commands.append(command)
        return self.child


@pytest.fixture()
def launches(monkeypatch) -> TransferLaunches:
    """
    Replace the process boundary while retaining the real service and API.
    """
    launched = TransferLaunches()
    monkeypatch.setattr(laboratory_runs.subprocess, "Popen", launched)
    return launched


def completed_output(run: LaboratoryRun, success: bool = True) -> None:
    """
    Write the demo's terminal report and replay artifact markers.
    """
    run.output_directory.mkdir(parents=True, exist_ok=True)
    run.metrics_path.write_text(json.dumps({"success": success}))
    for filename in ("scene.json", "trajectory.json"):
        (run.output_directory / filename).write_text("{}")


def test_running_transfer_reuses_the_existing_child(tmp_path, launches):
    """
    Double clicks never spawn a second process or reset the running task.
    """
    run = LaboratoryRun(tmp_path)
    assert run.status()[RunField.STATE] == RunState.IDLE
    first = run.start()
    assert first[RunField.STATE] == RunState.RUNNING
    assert run.start()[RunField.STATE] == RunState.RUNNING
    assert len(launches.commands) == 1
    assert launches.commands[0][:3] == [sys.executable, "-m", run.MODULE]


def test_success_needs_both_exit_and_the_demo_report(tmp_path, launches):
    """
    A successful Python exit alone is insufficient evidence of a transfer.
    """
    run = LaboratoryRun(tmp_path)
    run.start()
    launches.child.returncode = 0
    assert run.status()[RunField.STATE] == RunState.FAILED
    assert run.status()[RunField.OK] is False


def test_verified_transfer_exposes_its_recording(tmp_path, launches):
    """
    The successful recorded run is available after the child has exited.
    """
    run = LaboratoryRun(tmp_path)
    run.start()
    completed_output(run)
    launches.child.returncode = 0
    status = run.status()
    assert status[RunField.STATE] == RunState.SUCCEEDED
    assert status[RunField.RECORDING_URL] == run.RECORDING_URL
    assert status[RunField.EXIT_CODE] == 0


def test_previous_success_cannot_validate_a_later_failed_run(tmp_path, launches):
    """
    Starting a run removes the old terminal report before launching.
    """
    run = LaboratoryRun(tmp_path)
    completed_output(run)
    run.start()
    launches.child.returncode = 0
    assert run.status()[RunField.STATE] == RunState.FAILED
    assert not run.metrics_path.exists()


def test_nonzero_exit_stays_failed_even_with_a_success_report(tmp_path, launches):
    """
    The child process remains authoritative about execution failure.
    """
    run = LaboratoryRun(tmp_path)
    run.start()
    completed_output(run)
    launches.child.returncode = 2
    assert run.status()[RunField.STATE] == RunState.FAILED
    assert run.status()[RunField.EXIT_CODE] == 2


# %% local HTTP execution contract
def test_launch_endpoint_starts_and_reports_the_fixed_demo(server, launches):
    """
    The fixed endpoint returns its child's observable lifecycle.
    """
    code, started = post(server + LaboratoryRoute.START)
    assert code == 202
    assert started[RunField.STATE] == RunState.RUNNING
    assert get_json(server + LaboratoryRoute.STATUS)[RunField.STATE] == RunState.RUNNING
    assert len(launches.commands) == 1


def test_launch_endpoint_rejects_client_code(server, launches):
    """
    Client-controlled executable or argument fields cannot reach the process.
    """
    code, answer = post(server + LaboratoryRoute.START, {"code": "arbitrary"})
    assert code == 400
    assert answer[RunField.OK] is False
    assert launches.commands == []


def test_launch_endpoint_rejects_cross_origin_browser_requests(server, launches):
    """
    An unrelated website cannot trigger a localhost robot demonstration.
    """
    request = urllib.request.Request(
        server + LaboratoryRoute.START,
        method="POST",
        data=b"{}",
        headers={"Origin": "https://example.org", "Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as failure:
        urllib.request.urlopen(request)
    assert failure.value.code == 403
    assert launches.commands == []


def test_launch_endpoint_accepts_its_own_browser_origin(server, launches):
    """
    The local viewer can start the predetermined operation with JSON.
    """
    request = urllib.request.Request(
        server + LaboratoryRoute.START,
        method="POST",
        data=b"{}",
        headers={"Origin": server, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        assert response.status == 202
    assert len(launches.commands) == 1
