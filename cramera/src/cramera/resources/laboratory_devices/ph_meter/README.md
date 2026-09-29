# pH meter

Benchtop pH meter with an articulated support arm and a removable probe resource.

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
| `ph_meter_support_height_joint` | prismatic | 0.045 |
| `ph_meter_probe_arm_joint` | revolute | 0.7 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `ph_meter_probe_mount` | `ph_meter_probe_arm` | mount |
| `ph_meter_rinse_station` | `ph_meter_base` | sample |
| `ph_meter_read_control` | `ph_meter_base` | control |
| `ph_meter_body_grasp` | `ph_meter_base` | grasp |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
