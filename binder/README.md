# Binder: cramera, and nothing else

This branch exists to launch [cramera](../cramera/README.md) in a browser and show
nothing around it. No notebook is opened, no demo is run, no RViz is started — the
launch URL lands directly on the viewer, filling the tab.

## Launch

```
https://binder.intel4coro.de/v2/gh/sunava/cognitive_robot_abstract_machine/claude/binder-camera-fullscreen-startup-1wo4lt?urlpath=cramera/
```

`urlpath=cramera/` is what makes the page full-bleed: it is the route
[jupyter-server-proxy](https://jupyter-server-proxy.readthedocs.io/) mounts the viewer
on (see [jupyter_server_config.py](jupyter_server_config.py)), so the browser never
sees a JupyterLab shell. Drop the parameter to get the ordinary lab, with *cramera* in
its launcher.

The viewer starts on the first request, not at container startup, and the knowledge
base is built before it answers — the first load therefore takes noticeably longer than
the ones after it.

Select a scene with `&scene=<name>`; without one the viewer opens the default of the
[cram-scenes](https://github.com/cram2/cram-scenes) submodule.

## What the image contains

Only the workspace members cramera imports: `random_events`, `probabilistic_model`,
`krrood`, `semantic_digital_twin`, `coraplex` and `giskardpy`. The simulation,
perception and demo packages are left out — nothing in this Binder runs a robot, so
nothing needs them.

Live mode is not reachable here: it attaches to a demo process running next to the
viewer, and this image starts none.

## Running it locally

```bash
docker compose -f ./binder/docker-compose.yml up --build
```

Then open <http://localhost:8888/cramera/>. The repository is mounted into the
container, so edits to the frontend show up on reload.
