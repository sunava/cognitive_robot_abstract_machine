"""
Starting the live bridge: as a library call or as the demo wrapper CLI.
"""

from __future__ import annotations

import logging
import runpy
import signal
import sys
from pathlib import Path

from typing_extensions import TYPE_CHECKING

from cram_viz.live import hooks
from cram_viz.live.bridge import Bridge
from cram_viz.live.http import DEFAULT_PORT, BridgeHTTPServer, serve

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)


# %% library entry point -----------------------------------------------------------
def start(world: World | None = None, port: int = DEFAULT_PORT) -> BridgeHTTPServer:
    """
    Start the live bridge.

    Call once, ideally at the top of a demo.
    :param world: Bind to this world immediately; without it the bridge attaches to the
        executing world on the first executor tick.
    :param port: Port of the bridge's HTTP endpoints.
    :return: The running HTTP server (a daemon thread).
    :raises HookAlreadyInstalledError: if a bridge is already running in this process.
    """
    bridge = Bridge()
    hooks.install_mesh_hook(bridge)  # before the demo parses its objects
    hooks.install_plan_hooks(bridge)
    if world is not None:
        bridge.world = world
        bridge._bind()
        bridge.snapshot()  # single-threaded here, before execution starts
    hooks.install_tick_hook(bridge)
    server = serve(bridge, port)
    logger.info(
        "bridge on http://localhost:%d (the viewer shows a Live button "
        "while this runs)",
        port,
    )
    return server


# %% cli entry point ---------------------------------------------------------------
def main() -> None:
    """
    ``cram-viz-live path/to/demo.py`` — run a demo with the live bridge.

    The demo's own directory is put on ``sys.path`` so the demo can import its local
    helper modules, exactly as if it were started directly. After the demo finishes the
    bridge stays up for inspection until Ctrl-C.
    """
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        sys.exit("usage: cram-viz-live path/to/demo.py")
    demo = Path(sys.argv[1]).resolve()
    start()
    sys.path.insert(0, str(demo.parent))
    logger.info("running demo: %s", demo)
    runpy.run_path(str(demo), run_name="__main__")
    logger.info("demo finished — bridge stays up for inspection (Ctrl-C to quit)")
    try:
        signal.pause()  # wait for Ctrl-C (SIGINT) instead of a sleep loop
    except KeyboardInterrupt:
        pass
