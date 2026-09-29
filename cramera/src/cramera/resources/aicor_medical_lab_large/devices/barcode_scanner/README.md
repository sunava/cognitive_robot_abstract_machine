# Barcode scanner

Handheld identification scanner with a trigger and a separate docking cradle.

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
| `barcode_scanner_trigger_joint` | revolute | 0.22 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `barcode_scanner_body_grasp` | `barcode_scanner_body` | grasp |
| `barcode_scanner_scan_axis` | `barcode_scanner_body` | measurement |
| `barcode_scanner_trigger_control` | `barcode_scanner_trigger` | control |
| `barcode_scanner_dock_contact` | `barcode_scanner_body` | mount |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
