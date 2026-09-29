# Label printer

Desktop specimen-label printer with an opening media cover, a roll spindle and a feed roller.

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
| `label_printer_media_cover_joint` | revolute | 1.35 |
| `label_printer_media_spindle_joint` | continuous | 1.047 |
| `label_printer_feed_roller_joint` | continuous | 1.571 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `label_printer_cover_handle` | `label_printer_media_cover` | grasp |
| `label_printer_media_loading` | `label_printer_base` | sample |
| `label_printer_label_output` | `label_printer_base` | output |
| `label_printer_feed_control` | `label_printer_media_cover` | control |
| `label_printer_body_grasp` | `label_printer_base` | grasp |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
