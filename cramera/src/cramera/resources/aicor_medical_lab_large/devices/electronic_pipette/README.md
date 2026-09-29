# Electronic Pipette

Single-channel pipette with separate plunger and tip-ejector travel.

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
| `electronic_pipette_plunger_joint` | prismatic | -0.006 |
| `electronic_pipette_tip_ejector_joint` | prismatic | -0.008 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `electronic_pipette_tip_mount` | `electronic_pipette_body` | tool_attachment |
| `electronic_pipette_grip_center` | `electronic_pipette_body` | grasp |
| `electronic_pipette_plunger_contact` | `electronic_pipette_plunger` | control |
| `electronic_pipette_ejector_contact` | `electronic_pipette_tip_ejector` | control |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
