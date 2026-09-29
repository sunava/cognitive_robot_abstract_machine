# Tube Capper / Decapper

Articulated tube holder, opposing clamp jaws and a translating rotary chuck.

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
| `tube_capper_clamp_left_joint` | prismatic | 0 |
| `tube_capper_clamp_right_joint` | prismatic | 0 |
| `tube_capper_head_carriage_joint` | prismatic | 0 |
| `tube_capper_rotary_chuck_joint` | continuous | 1.047 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `tube_capper_left_clamp_contact` | `tube_capper_clamp_left` | contact |
| `tube_capper_right_clamp_contact` | `tube_capper_clamp_right` | contact |
| `tube_capper_sample_seat` | `tube_capper_base` | sample |
| `tube_capper_chuck_contact` | `tube_capper_rotary_chuck` | tool |
| `tube_capper_front_access` | `tube_capper_base` | approach |
| `tube_capper_display` | `tube_capper_base` | control |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
