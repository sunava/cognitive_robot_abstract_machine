# Pipette Tip Rack

Twenty-four open tip sockets in a covered refillable carrier.

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
| `tip_rack_lid_joint` | revolute | 1.7 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `tip_rack_slot_A1` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_A2` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_A3` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_A4` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_A5` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_A6` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_B1` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_B2` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_B3` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_B4` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_B5` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_B6` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_C1` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_C2` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_C3` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_C4` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_C5` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_C6` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_D1` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_D2` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_D3` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_D4` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_D5` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_slot_D6` | `tip_rack_socket_insert` | tip_socket |
| `tip_rack_lid_handle` | `tip_rack_lid` | handle |
| `tip_rack_insert_grasp` | `tip_rack_socket_insert` | grasp |
| `tip_rack_body_grasp` | `tip_rack_body` | grasp |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
