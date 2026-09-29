# 96-Well Microplate

Separate 127.76 by 85.48 millimetre plate with 9 millimetre well pitch and 96 open wells.

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
| `microplate_96_well_well_A1` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_A2` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_A3` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_A4` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_A5` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_A6` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_A7` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_A8` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_A9` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_A10` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_A11` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_A12` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B1` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B2` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B3` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B4` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B5` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B6` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B7` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B8` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B9` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B10` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B11` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_B12` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C1` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C2` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C3` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C4` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C5` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C6` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C7` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C8` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C9` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C10` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C11` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_C12` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D1` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D2` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D3` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D4` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D5` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D6` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D7` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D8` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D9` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D10` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D11` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_D12` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E1` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E2` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E3` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E4` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E5` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E6` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E7` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E8` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E9` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E10` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E11` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_E12` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F1` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F2` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F3` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F4` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F5` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F6` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F7` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F8` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F9` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F10` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F11` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_F12` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G1` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G2` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G3` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G4` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G5` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G6` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G7` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G8` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G9` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G10` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G11` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_G12` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H1` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H2` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H3` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H4` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H5` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H6` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H7` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H8` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H9` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H10` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H11` | `microplate_96_well_base` | sample |
| `microplate_96_well_well_H12` | `microplate_96_well_base` | sample |
| `microplate_96_well_left_grip` | `microplate_96_well_base` | grasp |
| `microplate_96_well_right_grip` | `microplate_96_well_base` | grasp |
| `microplate_96_well_plate_center` | `microplate_96_well_base` | placement |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
