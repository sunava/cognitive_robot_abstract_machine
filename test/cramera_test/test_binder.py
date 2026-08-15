"""
The Binder deployment: cramera as the page the launch URL opens.

``binder/jupyter_server_config.py`` is what makes that happen, and it is only ever read
by Jupyter — nothing imports it — so its contents are checked here.
"""

import re
import tomllib
from pathlib import Path

from traitlets.config import Config
from typing_extensions import Any, Dict, List

from cramera import server
from cramera.server import NO_BROWSER_FLAG

REPOSITORY_ROOT = Path(__file__).parents[2]

BINDER_DIRECTORY = REPOSITORY_ROOT / "binder"

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
