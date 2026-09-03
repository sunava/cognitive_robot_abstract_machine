"""
Tests for ``cramera_command.sh``, the console-script lookup the start scripts use.
"""

import shutil
import subprocess
from pathlib import Path

from cramera import paths

DRIVER = Path(__file__).parent / "dataset" / "resolve_cramera_command.sh"
"""
The shell script that sources the helper and prints what it resolved.
"""

BASH = shutil.which("bash")
"""
The shell the driver runs under, resolved here because the tests hand it a ``PATH``
holding nothing but the directory under test.
"""


def executable(path: Path) -> Path:
    """
    An executable file standing in for an installed console script.

    :param path: Where the stub is written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def resolve(
    repository: Path, search_path: Path, script: str
) -> subprocess.CompletedProcess:
    """
    Run the helper against one repository checkout and one search path.

    :param repository: The checkout holding ``cramera_command.sh``.
    :param search_path: The single directory to offer as ``PATH``.
    :param script: The console script name to look up.
    """
    shutil.copy(paths.repository_root() / "cramera_command.sh", repository)
    return subprocess.run(
        [BASH, str(DRIVER), script],
        env={"REPO": str(repository), "PATH": str(search_path)},
        capture_output=True,
        text=True,
        check=False,
    )


class TestConsoleScriptLookup:
    def test_the_active_environment_is_preferred(self, tmp_path):
        active = executable(
            tmp_path / "environment" / "bin" / paths.ConsoleScript.LIVE_DEMO.value
        )
        executable(
            tmp_path
            / "repository"
            / ".venv"
            / "bin"
            / paths.ConsoleScript.LIVE_DEMO.value
        )

        answer = resolve(
            tmp_path / "repository", active.parent, paths.ConsoleScript.LIVE_DEMO.value
        )

        assert answer.returncode == 0
        assert answer.stdout.strip() == str(active)

    def test_the_repository_environment_is_the_fallback(self, tmp_path):
        installed = executable(
            tmp_path
            / "repository"
            / ".venv"
            / "bin"
            / paths.ConsoleScript.LIVE_DEMO.value
        )
        (tmp_path / "bare").mkdir()

        answer = resolve(
            tmp_path / "repository",
            tmp_path / "bare",
            paths.ConsoleScript.LIVE_DEMO.value,
        )

        assert answer.returncode == 0
        assert answer.stdout.strip() == str(installed)

    def test_a_script_installed_nowhere_fails_with_an_explanation(self, tmp_path):
        (tmp_path / "repository").mkdir()
        (tmp_path / "bare").mkdir()

        answer = resolve(
            tmp_path / "repository",
            tmp_path / "bare",
            paths.ConsoleScript.LIVE_DEMO.value,
        )

        assert answer.returncode == 1
        assert paths.ConsoleScript.LIVE_DEMO.value in answer.stderr
