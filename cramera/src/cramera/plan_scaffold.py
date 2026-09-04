"""
The demo process the Plan Builder runs a scene and a plan in.

The builder hands over generated Python; this turns it into a running demo and answers
afterwards how that went. A demo can fail on its own -- a plan that cannot be performed,
a bridge port another demo still holds -- and a failure nobody can see is worse than a
slow one, so the process keeps its output and reports its own death.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, ClassVar, Dict, Optional

from cramera import paths
from cramera.live.http import DEFAULT_PORT as BRIDGE_PORT
from cramera.live.runner import CRAMERA_BACKEND, VISUALIZATION_BACKEND_VARIABLE
from cramera.logging_setup import get_logger

logger = get_logger(__name__)

PORT_PROBE_TIMEOUT_SECONDS = 0.25
"""
How long to wait for the bridge port to accept a connection before calling it free.
"""


class ScaffoldField(StrEnum):
    """
    Key the demo process reports one part of its state under.
    """

    RUNNING = "running"
    EXIT_CODE = "exitCode"
    OUTPUT = "output"


class BridgePortTaken(Exception):
    """
    Raised when a demo cannot be started because something already listens on the bridge
    port.
    """


def bridge_port_in_use(port: int) -> bool:
    """
    Whether something already listens on a demo's bridge port.

    :param port: The port a demo's bridge would bind.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(PORT_PROBE_TIMEOUT_SECONDS)
        return probe.connect_ex(("127.0.0.1", port)) == 0


@dataclass
class PlanScaffold:
    """
    One Plan-Builder demo process, running or finished.
    """

    DEMO_FILE: ClassVar[str] = "_builder_scaffold.py"
    """
    Name the generated demo is written under.
    """

    LOG_FILE: ClassVar[str] = "_builder_scaffold.log"
    """
    Name the demo's own output is written under, beside the demo itself.
    """

    REPORTED_LINES: ClassVar[int] = 20
    """
    How many of the demo's last lines :meth:`recent_output` reports.
    """

    STOP_GRACE_SECONDS: ClassVar[float] = 2.0
    """
    How long a stopped demo may take to exit before it is killed.
    """

    demo_path: Path
    """
    The generated demo file this process runs.
    """

    log_path: Path
    """
    File the demo's own output is written to.
    """

    process: subprocess.Popen
    """
    The demo process itself.
    """

    @classmethod
    def launch(cls, code: str, directory: Path) -> PlanScaffold:
        """
        Write generated demo code and run it, with its output kept beside it.

        :param code: The generated demo, as the builder produced it.
        :param directory: Directory the demo and its output are written to.
        :raises BridgePortTaken: If something already listens on the bridge port, which
            a starting demo would die on.
        :raises cramera.paths.ConsoleScriptMissing: If ``cramera-live`` is not
            installed.
        """
        if bridge_port_in_use(BRIDGE_PORT):
            raise BridgePortTaken(
                "a demo is already running on bridge port %d — stop it before starting "
                "another" % BRIDGE_PORT
            )
        launcher = paths.console_script(paths.ConsoleScript.LIVE_DEMO)
        directory.mkdir(parents=True, exist_ok=True)
        demo_path = directory / cls.DEMO_FILE
        demo_path.write_text(code)
        log_path = directory / cls.LOG_FILE
        log = log_path.open("w")
        try:
            process = subprocess.Popen(
                [str(launcher), str(demo_path)],
                cwd=str(paths.repository_root()),
                env=dict(
                    os.environ, **{VISUALIZATION_BACKEND_VARIABLE: CRAMERA_BACKEND}
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own group, so stopping takes its children too
            )
        finally:
            log.close()
        logger.info("running the built plan: %s", demo_path)
        return cls(demo_path=demo_path, log_path=log_path, process=process)

    def is_running(self) -> bool:
        """
        Whether the demo is still running.
        """
        return self.process.poll() is None

    def stop(self) -> None:
        """
        End the demo and everything it started, waiting briefly before killing it.

        Does nothing to a demo that has already finished.
        """
        if not self.is_running():
            return
        group = os.getpgid(self.process.pid)
        os.killpg(group, signal.SIGTERM)
        deadline = time.monotonic() + self.STOP_GRACE_SECONDS
        while time.monotonic() < deadline and self.is_running():
            time.sleep(0.05)
        if self.is_running():
            os.killpg(group, signal.SIGKILL)
            self.process.wait()

    def recent_output(self) -> str:
        """
        The demo's last lines, which is where a demo that died says why.
        """
        if not self.log_path.is_file():
            return ""
        lines = self.log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-self.REPORTED_LINES :])

    def state(self) -> Dict[str, Any]:
        """
        What the builder polls to tell a demo that is running from one that died.
        """
        exit_code: Optional[int] = self.process.poll()
        return {
            ScaffoldField.RUNNING.value: exit_code is None,
            ScaffoldField.EXIT_CODE.value: exit_code,
            ScaffoldField.OUTPUT.value: self.recent_output(),
        }
