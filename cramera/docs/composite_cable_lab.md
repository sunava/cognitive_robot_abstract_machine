# Composite Cable Lab

Run the Cramera server in the CRAM environment and open **Cable Lab** in the top
navigation, or visit `http://localhost:8711/cable-lab.html`:

```bash
.venv/bin/python -m cramera.server --no-browser
```

The lab runs MuJoCo in a headless `MujocoSim` worker. Its browser view displays the
measured cable and gripper geometry in 3D, with a measured 2D projection when
WebGL is unavailable. It needs the local Python server; a static copy of the
frontend cannot run physics. The implementation is tested with MuJoCo 3.9.0.

## Try a friction grasp

1. Select the loose-cable preset, then start the simulation. This disables the
   optional end anchor so a failed grasp can fall freely.
2. Close the parallel gripper with **Pinch** and inspect the measured pad contact
   forces.
3. Use **Lift** to raise the gripper. Cable motion results from contact friction.
4. **Open** opens the pads. The loose cable falls back onto the support surface.
5. Repeat with low sliding friction to observe slip. **Reset** restores the last
   applied configuration; starting again applies edited parameters.

An anchored cable remains attached at its first joint even after the gripper
opens. Contact counts and forces are measurements, not a guarantee that the cable
will remain held.

## Cable model

`CableConfig` specifies element count and length, capsule radius, **total** cable
mass, color, bending/twisting stiffness, anchor pose, and MuJoCo contact settings.
`mass_per_segment`, `strategy`, and `use_composite` are not supported. The first
element's default direction is the parent's local positive x-axis. Anchor offset
and roll/pitch/yaw are evaluated in that parent's frame.

The physical cable uses MuJoCo's native `composite type="cable"`: rigid capsule
elements connected by ball joints. The native `mujoco.elasticity.cable` plugin
provides bend and twist stiffness. The implementation parses the composite MJCF,
attaches its expanded spec to the scene, and recompiles before starting physics.
The sum of the generated element masses equals the configured total mass.
The lab uses a 1 ms timestep with RK4 integration and finger actuators limited to
2 N each. Solver warnings are displayed in the lab. Its fixed anchor frame is
0.35 m above the world origin; configured offsets are relative to that frame.

`MjSpec.add_flex()` describes a different MuJoCo model and does **not** generate
this rigid-body chain. See the official [composite reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-composite)
and the [MuJoCo maintainer's spec attachment example](https://github.com/google-deepmind/mujoco/discussions/2798).

Semantic `Body` points named `cable_segment_i` retain free connections under the
world root. They receive poses from the physical cable. They have no duplicate
collision or inertial bodies in the MuJoCo model. The physical links are named
`cable_segment_Bi`. The optional end anchor uses the native cable root connection.
Gripping never adds welds, equality constraints, or semantic reparenting and never
overwrites a physical cable segment's position.

If compilation is unavailable, the lab exposes its semantic placeholders and a
warning. It explicitly reports `physical: false`; it cannot report a successful
physical grasp in that state.

## Local API

| Method | Route | Payload |
| --- | --- | --- |
| GET | `/api/cable/state` | Current geometry, contacts, configuration and lifecycle |
| POST | `/api/cable/start` | Flat configuration object; `{}` uses defaults |
| POST | `/api/cable/command` | `{"action":"pinch"}`, `release`, `lift`, or `lower` |
| POST | `/api/cable/command` | `{"action":"move","position":[x,y,z]}` in metres |
| POST | `/api/cable/reset` | `{}` |
| POST | `/api/cable/stop` | `{}` |

The lab owns an independent world and a parallel gripper. It is an experiment
for cable contact physics, not a full-arm manipulation plan. Lifecycle changes
are serialized, and stopping or shutting down the server releases the simulation
worker and its world synchronization callbacks.
