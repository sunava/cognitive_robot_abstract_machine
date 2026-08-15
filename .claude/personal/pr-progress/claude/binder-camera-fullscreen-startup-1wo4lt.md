## Binder branch that opens cramera fullscreen

Goal: a Binder branch where nothing runs but cramera, opened straight into
the browser tab. Branched off `cramera-world-visualization` (that is where
`cramera/` lives), not off `main`.

Plan, agreed with the user:
1. `binder/` with a Dockerfile installing only the workspace members cramera
   imports, and a jupyter-server-proxy entry mounting the viewer at
   `<base_url>/cramera/`.
2. Launch URL `?urlpath=cramera/` so the page is the viewer alone, no
   JupyterLab shell.
3. Make the frontend's api urls relative, since the proxy prefix breaks
   root-absolute ones.

Done (commit 8f61f1ec0, pushed):
- `binder/Dockerfile`, `jupyter_server_config.py`, `docker-compose.yml`,
  `README.md`
- `cramera/src/cramera/web/core/api.js` (`ServerApi.urlFor`) and all 15 call
  sites converted
- tests: `test/cramera_test/test_binder.py`,
  `test/cramera_test/js/test_server_api.js`,
  `TestServedUnderAPathPrefix` in `test_web_assets.py`; graph-panel node test
  now binds the real `core/api.js`
- 453 cramera tests pass locally (`test_onboard.py`/`test_palette.py` need
  `physics_simulators`, absent from the local venv only)

First Binder build failed before Docker even started - repo2docker's
`git submodule update --init --recursive` aborted on `cram_viz/scenes`, a
gitlink left behind by the cramera rename, and `defense/scenes`, never
declared. Both directories held nothing else. Fixed in ade4b6bd2 with
`TestEverySubmoduleCanBeCloned` in `test_binder.py` guarding the invariant.

Next: no PR opened yet - the user did not ask for one. Still unverified
until a Binder build gets past the clone: the image contents, and whether
the base image's own `ServerProxy.servers` (if it sets any) get overwritten
by the plain assignment in `jupyter_server_config.py`.
