"""
Verify local lifecycle controls without opening an external tunnel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from collections.abc import Iterator

import pytest


# %% isolated launcher
@pytest.fixture
def launcher(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    """
    Provide the launcher with an isolated private runtime directory.
    """
    script = Path(__file__).resolve().parents[2] / "scripts/share_cramera_lab.sh"
    state = tmp_path / "share"
    environment = dict(os.environ, CRAMERA_SHARE_STATE=str(state))
    return script, environment, state


@dataclass
class GatewayRecovery:
    """
    An isolated gateway launch with an existing owned tunnel process.
    """

    script: Path
    """
    Copied launcher whose project directory contains only test fixtures.
    """

    environment: dict[str, str]
    """
    Private paths and mock readiness checks for child processes.
    """

    state: Path
    """
    Temporary share runtime directory.
    """

    tunnel: subprocess.Popen
    """
    Live process bearing the command-line identity of the existing tunnel.
    """

    origin: str
    """
    Original remote origin that gateway recovery must retain.
    """

    def start(self) -> subprocess.CompletedProcess:
        """
        Run the real shell launcher against isolated process fixtures.
        """
        return subprocess.run(
            ["bash", str(self.script), "start"],
            env=self.environment,
            capture_output=True,
            timeout=10,
        )


@pytest.fixture
def gateway_recovery(tmp_path: Path) -> Iterator[GatewayRecovery]:
    """
    Exercise process ownership and recovery without connecting to a tunnel.
    """
    source = Path(__file__).resolve().parents[2]
    dataset = Path(__file__).parent / "dataset" / "laboratory_share_launcher"
    repository = tmp_path / "repository"
    script = repository / "scripts" / "share_cramera_lab.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(source / "scripts" / script.name, script)
    interpreter = repository / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    package = repository / "cramera"
    package.mkdir()
    (package / "__init__.py").touch()
    shutil.copyfile(dataset / "process.py", package / "laboratory_share.py")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for filename in ("curl", "cloudflared"):
        target = binaries / filename
        shutil.copyfile(dataset / filename, target)
        target.chmod(0o700)
    state = tmp_path / "share"
    state.mkdir()
    (state / "password").write_text("launcher-test-password-only")
    origin = "https://existing-laboratory.trycloudflare.com"
    (state / "origin").write_text(origin + "\n")
    environment = dict(
        os.environ,
        CRAMERA_SHARE_STATE=str(state),
        CRAMERA_CLOUDFLARED=str(binaries / "cloudflared"),
        CRAMERA_LAUNCHER_TEST_GATEWAY_READY=str(tmp_path / "gateway-ready"),
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
    )
    tunnel = subprocess.Popen(
        [
            str(binaries / "cloudflared"),
            str(dataset / "process.py"),
            "--url",
            "http://127.0.0.1:8717",
        ],
        executable=sys.executable,
        env=environment,
    )
    (state / "tunnel.pid").write_text(str(tunnel.pid) + "\n")
    recovery = GatewayRecovery(script, environment, state, tunnel, origin)
    try:
        yield recovery
    finally:
        subprocess.run(
            ["bash", str(script), "stop"],
            env=environment,
            capture_output=True,
            timeout=10,
        )
        if tunnel.poll() is None:
            tunnel.terminate()
        tunnel.wait(timeout=5)


# %% lifecycle boundaries
def test_inactive_status_does_not_create_runtime_directory(launcher) -> None:
    """
    An informational status call leaves an unused installation unchanged.
    """
    script, environment, state = launcher
    result = subprocess.run(
        ["bash", str(script), "status"], env=environment, capture_output=True
    )
    assert result.returncode == 1
    assert state.exists() is False


def test_invalid_command_returns_usage_error(launcher) -> None:
    """
    Unknown lifecycle actions fail before creating runtime state.
    """
    script, environment, state = launcher
    result = subprocess.run(
        ["bash", str(script), "publish-everything"],
        env=environment,
        capture_output=True,
    )
    assert result.returncode == 2
    assert state.exists() is False


def test_stop_does_not_terminate_unrelated_process(launcher) -> None:
    """
    A stale or replaced PID file cannot authorize stopping another process.
    """
    script, environment, state = launcher
    state.mkdir()
    process = subprocess.Popen(["sleep", "30"])
    (state / "gateway.pid").write_text(str(process.pid))
    try:
        result = subprocess.run(
            ["bash", str(script), "stop"], env=environment, capture_output=True
        )
        assert result.returncode == 1
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_stop_of_inactive_share_is_successful(launcher) -> None:
    """
    Stopping an unused share is safe and idempotent.
    """
    script, environment, state = launcher
    result = subprocess.run(
        ["bash", str(script), "stop"], env=environment, capture_output=True
    )
    assert result.returncode == 0
    assert state.exists() is False


def test_gateway_recovery_preserves_running_tunnel(
    gateway_recovery: GatewayRecovery,
) -> None:
    """
    Replacing a missing gateway keeps the reachable public address alive.
    """
    result = gateway_recovery.start()
    assert gateway_recovery.tunnel.poll() is None
    assert result.returncode == 0, result.stderr.decode()
    assert (gateway_recovery.state / "tunnel.pid").read_text().strip() == str(
        gateway_recovery.tunnel.pid
    )
    assert (gateway_recovery.state / "origin").read_text().strip() == (
        gateway_recovery.origin
    )
    gateway = int((gateway_recovery.state / "gateway.pid").read_text())
    assert Path(f"/proc/{gateway}").exists()


def test_failed_gateway_recovery_preserves_running_tunnel(
    gateway_recovery: GatewayRecovery,
) -> None:
    """
    A gateway startup error cannot tear down the separately healthy tunnel.
    """
    gateway_recovery.environment["CRAMERA_LAUNCHER_TEST_GATEWAY_FAIL"] = "1"
    result = gateway_recovery.start()
    assert gateway_recovery.tunnel.poll() is None
    assert result.returncode == 1
    assert (gateway_recovery.state / "tunnel.pid").read_text().strip() == str(
        gateway_recovery.tunnel.pid
    )
    assert (gateway_recovery.state / "origin").read_text().strip() == (
        gateway_recovery.origin
    )
