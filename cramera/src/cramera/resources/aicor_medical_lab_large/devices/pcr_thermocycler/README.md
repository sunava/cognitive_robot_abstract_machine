# PCR Thermocycler

Ninety-six-position thermal block with an articulated heated cover and cycle display.

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
| `pcr_thermocycler_heated_lid_joint` | revolute | -1.7 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `pcr_thermocycler_block_A1` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_A2` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_A3` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_A4` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_A5` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_A6` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_A7` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_A8` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_A9` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_A10` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_A11` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_A12` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B1` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B2` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B3` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B4` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B5` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B6` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B7` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B8` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B9` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B10` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B11` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_B12` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C1` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C2` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C3` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C4` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C5` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C6` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C7` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C8` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C9` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C10` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C11` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_C12` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D1` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D2` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D3` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D4` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D5` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D6` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D7` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D8` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D9` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D10` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D11` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_D12` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E1` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E2` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E3` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E4` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E5` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E6` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E7` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E8` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E9` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E10` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E11` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_E12` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F1` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F2` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F3` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F4` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F5` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F6` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F7` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F8` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F9` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F10` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F11` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_F12` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G1` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G2` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G3` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G4` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G5` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G6` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G7` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G8` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G9` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G10` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G11` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_G12` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H1` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H2` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H3` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H4` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H5` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H6` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H7` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H8` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H9` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H10` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H11` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_block_H12` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_lid_handle` | `pcr_thermocycler_heated_lid` | handle |
| `pcr_thermocycler_lid_hinge` | `pcr_thermocycler_heated_lid` | joint |
| `pcr_thermocycler_plate_seat` | `pcr_thermocycler_reaction_block` | sample |
| `pcr_thermocycler_load_approach` | `pcr_thermocycler_reaction_block` | approach |
| `pcr_thermocycler_start_button` | `pcr_thermocycler_base` | control |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
