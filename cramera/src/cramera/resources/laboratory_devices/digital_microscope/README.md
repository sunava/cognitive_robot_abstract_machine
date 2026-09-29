# Digital microscope

Digital microscope with a focus carriage, a two-axis slide stage and a three-position objective turret.

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
| `digital_microscope_monitor_joint` | revolute | 0.2 |
| `digital_microscope_objective_turret_joint` | revolute | 2.094 |
| `digital_microscope_focus_carriage_joint` | prismatic | 0.01 |
| `digital_microscope_slide_x_joint` | prismatic | 0.02 |
| `digital_microscope_slide_y_joint` | prismatic | -0.015 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `digital_microscope_sample_stage` | `digital_microscope_slide_y` | sample |
| `digital_microscope_focus_control` | `digital_microscope_focus_carriage` | control |
| `digital_microscope_body_grasp` | `digital_microscope_base` | grasp |
| `digital_microscope_objective_axis` | `digital_microscope_optical_head` | measurement |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
