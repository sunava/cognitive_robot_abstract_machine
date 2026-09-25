# AICOR Wood Workshop

The workshop is a separate CRAMERA scene, `aicor_workshop`. Its original resources
are in `cramera/src/cramera/resources/aicor_workshop`, inside the `private_meshes`
resource checkout as well as the application package. The medical laboratory
scene and its protected address remain separate.

## Browser controls

Open `index.html?scene=aicor_workshop&layout=scene&offline=1` on the CRAMERA server.
The **Wood workshop** panel starts a dedicated MuJoCo contact world. Drag a box or
part on the bench, or select it and choose **Grab selected**. The movement buttons
change the manipulation target by 2.5 cm; **Tilt / rotate by 15°** changes its
orientation. **Release** removes the manipulation force so gravity and contacts
determine the result. **Pause physics** freezes the scene; **Reset** restores it.

The two fastener boxes, four shelf boxes, individual screws and nails, wooden
workpiece and hand screwdriver are independent rigid bodies. Contents are not
parented to their box: friction and wall contacts carry them, and tilting can make
them slide or spill. The bench, shelf frame, vise and mounted tool display remain
fixtures. Drawer buttons request force-limited motion; obstructions can stop them.
Handle TF frames and measured contacts can be shown through **Layers**.

Under **Screw driving**, use **Position screw**, **Align driver**, then **Drive in**.
**Reverse** withdraws it and **Stop** stops rotation. These alignment controls move
force targets through elevated waypoints; the displayed geometry always follows
measured MuJoCo poses. They do not teleport a blocked object to its destination.

The contact model uses primitive collision shapes, gravity and friction. Screw
driving adds an ideal thread constraint: after alignment and contact engagement,
measured screwdriver rotation advances the screw by the configured pitch, with
reaction on the movable workpiece. This models insertion and withdrawal, not wood
cutting, damage, calibrated tightening torque or detailed thread-flank contact.
The UI reports actual insertion depth and tool engagement.

## Native PR2 plan

`cramera/src/cramera/workshop_demo.py` defines the native `WorkshopDemo`, a
`Context`, modular `PickUpScrewAction` / `PlaceAction`, a `sequential` plan and
execution inside `with simulated_robot_advanced`. It uses Giskard motion planning,
collision geometry and a geometric finger closing limit. It targets the prepared
pilot hole; no material-removal or physical screw-thread simulation is claimed.
The base is fixed for this bench task, and a Giskard pointing constraint makes the
head follow the working gripper. The active arm uses a 20 mm preferred planning
clearance; penetration remains forbidden. The standalone workshop world retains
the mobile base.

Run from the configured CRAM environment:

```bash
.venv/bin/python -m cramera.workshop_demo
```

Only successful runs with checked final pose are exported as `aicor_workshop_pr2`
in the local scene directory. This recorded scene is separate from the interactive
MuJoCo world. Its frames are actual native controller states, not authored robot
keyframes; playback does not control the workshop contact session.
`acceptance_metrics.json` records placement accuracy, measured collision and finger
clearance, release, unchanged neighboring parts and stationary-base checks. The
recording covers pickup, transport, insertion and release. Screwdriver operation
is currently available through the interactive physics controls.

## Blender source and regeneration

```bash
blender --background --python cramera/tools/workshop/build_blender.py -- \
  --output cramera/src/cramera/resources/aicor_workshop \
  --source ~/.cramera/scenes/aicor_workshop
.venv/bin/python cramera/tools/workshop/bundle.py --output cramera/src/cramera/resources/aicor_workshop
```

The authoring script generates `workshop.blend` and overview/detail renders in the
chosen source directory, and reusable GLBs in the resource bundle. Keep the export's
notice files with authorized copies. Asset packaging applies original workshop
attribution without changing the CRAM software or third-party PR2 licence.

## Protected hosting

The dedicated workshop gateway serves `/workshop/` with the existing access
password and an independent secure session cookie. It exposes only workshop
assets and, when explicitly configured, its native recording. Its own contact
session runs in `cramera-workshop-gateway.service`, with read-only state at
`/workshop/api/workshop/state` and authenticated same-origin commands at
`/workshop/api/workshop/command`. It cannot call the laboratory robot API.
The gateway is independent of both the laboratory services and `cramera.service`.
Pin a deployed bundle with `--scene-directory /path/to/release/aicor_workshop` so
older writable recordings cannot shadow its models or physics metadata.
The local viewer exposes the same API without the `/workshop` mount prefix.
All clients observe one shared workshop; it pauses and releases manipulation after
90 seconds without a client request.
