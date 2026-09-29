# Benchtop Incubator

Three-shelf incubation chamber with a transparent articulated door and temperature display.

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
| `incubator_door_joint` | revolute | -1.63 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `incubator_shelf_1_sample_1_1` | `incubator_shelf_1` | sample |
| `incubator_shelf_1_sample_1_2` | `incubator_shelf_1` | sample |
| `incubator_shelf_1_sample_2_1` | `incubator_shelf_1` | sample |
| `incubator_shelf_1_sample_2_2` | `incubator_shelf_1` | sample |
| `incubator_shelf_1_sample_3_1` | `incubator_shelf_1` | sample |
| `incubator_shelf_1_sample_3_2` | `incubator_shelf_1` | sample |
| `incubator_shelf_2_sample_1_1` | `incubator_shelf_2` | sample |
| `incubator_shelf_2_sample_1_2` | `incubator_shelf_2` | sample |
| `incubator_shelf_2_sample_2_1` | `incubator_shelf_2` | sample |
| `incubator_shelf_2_sample_2_2` | `incubator_shelf_2` | sample |
| `incubator_shelf_2_sample_3_1` | `incubator_shelf_2` | sample |
| `incubator_shelf_2_sample_3_2` | `incubator_shelf_2` | sample |
| `incubator_shelf_3_sample_1_1` | `incubator_shelf_3` | sample |
| `incubator_shelf_3_sample_1_2` | `incubator_shelf_3` | sample |
| `incubator_shelf_3_sample_2_1` | `incubator_shelf_3` | sample |
| `incubator_shelf_3_sample_2_2` | `incubator_shelf_3` | sample |
| `incubator_shelf_3_sample_3_1` | `incubator_shelf_3` | sample |
| `incubator_shelf_3_sample_3_2` | `incubator_shelf_3` | sample |
| `incubator_door_handle` | `incubator_door` | handle |
| `incubator_door_hinge` | `incubator_door` | joint |
| `incubator_temperature_control` | `incubator_base` | control |
| `incubator_chamber_approach` | `incubator_base` | approach |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
