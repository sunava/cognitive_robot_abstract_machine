# AICOR dual dispenser

Load `model.urdf` independently or copy this complete folder into another project.
All visual files and artwork are included under `assets/`.

The root `laboratory_pipetting_station` is at the platform centre on its support
surface. Coordinates are metres, with Z up and the front toward negative Y.
The visual contains both reservoirs, supply tubes, nozzles and instrument screen.
Separate collision primitives preserve the open receiving area and front grasp
approach. The original laboratory placement is recorded in `metadata.json`.

The model is stationary and has no moving joints. Dosing, reservoir inventory,
liquid transfer and the live screen require the laboratory simulation controller.
