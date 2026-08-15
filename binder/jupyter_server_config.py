"""
Serve cramera as the page this Binder opens.

jupyter-server-proxy starts the viewer on the first request and mounts it at
``<base_url>/cramera/``, so a launch URL ending in ``?urlpath=cramera/`` puts the
browser on the viewer alone — nothing else is started.
"""

c = get_config()  # noqa: F821  (traitlets provides it when Jupyter loads this file)

VIEWER_STARTUP_SECONDS = 300
"""
How long the proxy waits for the viewer's port.

The server builds the knowledge base before it starts listening, and that scan reads the
whole repository.
"""

c.ServerProxy.servers = {
    "cramera": {
        "command": ["cramera", "{port}", "--no-browser"],
        "timeout": VIEWER_STARTUP_SECONDS,
        "launcher_entry": {"title": "cramera"},
    }
}
