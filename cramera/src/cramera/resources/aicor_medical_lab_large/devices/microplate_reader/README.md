# Microplate Reader

Enclosed optical-reader model with an articulated drawer and a 96-well plate seat.

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
| `microplate_reader_plate_drawer_joint` | prismatic | 0.195 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `microplate_reader_drawer_handle` | `microplate_reader_plate_drawer` | handle |
| `microplate_reader_plate_seat` | `microplate_reader_plate_drawer` | sample |
| `microplate_reader_load_approach` | `microplate_reader_plate_drawer` | approach |
| `microplate_reader_eject_button` | `microplate_reader_base` | control |
| `microplate_reader_start_button` | `microplate_reader_base` | control |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
