# Thermomixer

Covered temperature-controlled mixer with a removable 24-well aluminum insert.

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
| `thermomixer_mixing_carrier_joint` | prismatic | 0 |
| `thermomixer_orbital_carrier_joint` | prismatic | 0 |
| `thermomixer_lid_joint` | revolute | 1.65 |
| `thermomixer_cycle_control_joint` | revolute | 1.571 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `thermomixer_slot_A1` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_A2` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_A3` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_A4` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_A5` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_A6` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_B1` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_B2` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_B3` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_B4` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_B5` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_B6` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_C1` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_C2` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_C3` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_C4` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_C5` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_C6` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_D1` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_D2` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_D3` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_D4` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_D5` | `thermomixer_sample_block` | sample_well |
| `thermomixer_slot_D6` | `thermomixer_sample_block` | sample_well |
| `thermomixer_lid_handle` | `thermomixer_lid` | handle |
| `thermomixer_block_grasp` | `thermomixer_sample_block` | grasp |
| `thermomixer_cycle_control_grasp` | `thermomixer_cycle_control` | control |
| `thermomixer_body_grasp` | `thermomixer_body` | grasp |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
