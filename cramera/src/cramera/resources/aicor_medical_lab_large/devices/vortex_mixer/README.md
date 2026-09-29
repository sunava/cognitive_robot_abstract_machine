# Vortex Mixer

Bench vortex mixer with a spring-loaded contact cup and speed control.

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
| `vortex_mixer_contact_head_joint` | prismatic | -0.005 |
| `vortex_mixer_speed_control_joint` | revolute | 2.827 |

## Interaction frames

| Frame | Attached to | Purpose |
| --- | --- | --- |
| `vortex_mixer_sample_contact` | `vortex_mixer_contact_head` | sample_contact |
| `vortex_mixer_speed_control_grasp` | `vortex_mixer_speed_control` | control |
| `vortex_mixer_body_grasp` | `vortex_mixer_body` | grasp |

The URDF supplies geometry, articulation and frames. It does not itself execute dosing,
mixing, heating, optical measurements, printing or biological processes.
