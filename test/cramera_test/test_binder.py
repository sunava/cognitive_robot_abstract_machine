"""
The Binder deployment: cramera as the page the launch URL opens.

``binder/jupyter_server_config.py`` is what makes that happen, and it is only ever read
by Jupyter — nothing imports it — so its contents are checked here.
"""

import re
import subprocess
import tomllib
from pathlib import Path

import pytest
from traitlets.config import Config
from typing_extensions import Any, Dict, List

from cramera import server
from cramera.server import NO_BROWSER_FLAG

REPOSITORY_ROOT = Path(__file__).parents[2]

BINDER_DIRECTORY = REPOSITORY_ROOT / "binder"

GITLINK_MODE = "160000"
"""
The mode git records a committed submodule under in the index.
"""

SERVER_CONFIG_FILE = BINDER_DIRECTORY / "jupyter_server_config.py"
"""
The Jupyter config the image appends to ``/etc/jupyter/jupyter_server_config.py``.
"""

LAUNCH_ROUTE_PATTERN = re.compile(r"\?urlpath=([\w-]+)/")
"""
How the documented launch URL names the route the viewer is served on.
"""

PORT_PLACEHOLDER = "{port}"
"""
What jupyter-server-proxy replaces with the port it assigned the viewer.
"""

VIEWER_ENTRY_POINT = "%s:%s" % (server.__name__, server.main.__name__)
"""
The function a console script has to call to be the one that serves the viewer.
"""


def proxied_servers() -> Dict[str, Any]:
    """
    The ``ServerProxy.servers`` mapping the Binder config declares, loaded the way
    Jupyter loads it.
    """
    config = Config()
    namespace = {"get_config": lambda: config}
    source = SERVER_CONFIG_FILE.read_text()
    exec(compile(source, str(SERVER_CONFIG_FILE), "exec"), namespace)
    return config.ServerProxy.servers


def viewer_command_name() -> str:
    """
    The command cramera's packaging installs for :func:`cramera.server.main`.
    """
    packaging = tomllib.loads(
        (REPOSITORY_ROOT / "cramera" / "pyproject.toml").read_text()
    )
    scripts = packaging["project"]["scripts"]
    [name] = [
        script for script, target in scripts.items() if target == VIEWER_ENTRY_POINT
    ]
    return name


def read_git(*arguments: str) -> List[str]:
    """
    The non-empty output lines of one git command run in the repository.

    :param arguments: The git command line, without the program name.
    """
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.splitlines()


def committed_submodule_paths() -> List[str]:
    """
    Every submodule the checkout records, whether or not it is declared.
    """
    entries = [line for line in read_git("ls-files", "--stage") if GITLINK_MODE in line]
    return [line.split("\t", 1)[1] for line in entries]


def declared_submodule_paths() -> List[str]:
    """
    The submodule paths ``.gitmodules`` gives a url, as git itself reads them.
    """
    settings = read_git(
        "config",
        "--file",
        str(REPOSITORY_ROOT / ".gitmodules"),
        "--get-regexp",
        r"^submodule\..*\.path$",
    )
    return [setting.split(" ", 1)[1] for setting in settings]


@pytest.mark.skipif(
    not (REPOSITORY_ROOT / ".git").exists(), reason="not a git checkout"
)
class TestEverySubmoduleCanBeCloned:
    """
    Binder clones the repository with ``git submodule update --init --recursive``, which
    aborts the whole build on a recorded submodule that ``.gitmodules`` has no url for.
    """

    def test_every_committed_submodule_is_declared(self):
        undeclared = sorted(
            set(committed_submodule_paths()) - set(declared_submodule_paths())
        )
        assert not undeclared, undeclared


class TestTheLaunchUrlReachesTheViewer:
    """
    The route the proxy mounts the viewer on and the route the launch URL asks for are
    the same string in two files; a rename in one alone lands the launch on a 404.
    """

    def test_the_launch_url_names_the_route_the_proxy_mounts(self):
        [route] = LAUNCH_ROUTE_PATTERN.findall(
            (BINDER_DIRECTORY / "README.md").read_text()
        )
        assert route in proxied_servers()


class TestTheProxyStartsTheViewer:
    """
    What the proxy runs when the route is first requested.
    """

    def command(self) -> List[str]:
        """
        The command line the proxy starts the viewer with.
        """
        [server_specification] = proxied_servers().values()
        return server_specification["command"]

    def test_it_runs_the_command_that_serves_the_viewer(self):
        assert self.command()[0] == viewer_command_name()

    def test_it_hands_the_viewer_the_port_it_assigned(self):
        assert PORT_PLACEHOLDER in self.command()

    def test_it_keeps_the_viewer_from_opening_a_browser(self):
        """
        A browser opened inside the container reaches nobody.
        """
        assert NO_BROWSER_FLAG in self.command()
