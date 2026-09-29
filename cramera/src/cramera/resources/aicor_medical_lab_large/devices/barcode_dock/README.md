# Barcode scanner dock

Weighted charging cradle for the independent handheld barcode scanner.

Original AICOR concept model. Principal author and design direction: Vanessa Hassouna.

`model.blend` is editable in Blender 4.0 or newer. Its timeline shows closed/rest at frame 1,
an inspection pose at frame 60, and closed/rest again at frame 120.
`model.urdf` loads independently using only local `assets/*.glb` files.
Copy this entire folder, including its notices, for an authorized reuse.

Coordinates are metres, Z up, front toward negative Y. Use a unique URDF parser prefix for each instance.
Collision sleeves retain open bores; casing contacts use convex primitives.
Masses/inertias are approximate model parameters, not manufacturer specifications.

## Moving parts

| Joint | Motion | Inspection position |
| --- | --- | --- |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `barcode_dock_scanner_mount` | `barcode_dock_base` | mount |
| `barcode_dock_body_grasp` | `barcode_dock_base` | grasp |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
