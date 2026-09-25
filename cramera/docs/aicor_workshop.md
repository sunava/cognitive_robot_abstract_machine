# AICOR Wood Workshop

The workshop is a separate CRAMERA scene, `aicor_workshop`. Its original resources
are in `cramera/src/cramera/resources/aicor_workshop`, inside the `private_meshes`
resource checkout as well as the application package. The medical laboratory
scene and its protected address remain separate.

## Browser controls

Open `index.html?scene=aicor_workshop&layout=scene&offline=1` on the CRAMERA server.
The **Wood workshop** panel provides three drawer controls, a selector for all ten
individual fasteners, **Lift selected part**, four movement controls, **Insert in
pilot hole**, **Return to box**, **Reset workshop**, and camera presets.

Lift a screw before inserting it. The pilot hole accepts one screw at a time;
lift the inserted screw and return it to the box before inserting another. A nail
can be lifted and moved independently. Drawer handles also have TF frames, visible
through the **Layers** panel. Collapse the workshop controls for an unobstructed view.

Manual controls are direct positioning with bounded travel, elevated transport
and an occupancy check. They are not contact dynamics or machining simulation.
The drill, tools, shelf stock and vise are currently fixed visual props.

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
in the local scene directory. This recorded scene is separate from manual object
editing. Its frames are actual native controller states, not authored robot keyframes.
`acceptance_metrics.json` records placement accuracy, measured collision and finger
clearance, release, unchanged neighboring parts and stationary-base checks. The
first validated recording contains 44.5 seconds of pickup, transport, insertion and
release; powered screw driving is a later action.

## Blender source and regeneration

```bash
blender --background --python cramera/tools/workshop/build_blender.py -- \
  --output cramera/src/cramera/resources/aicor_workshop \
  --source ~/.cramera/scenes/aicor_workshop
.venv/bin/python cramera/tools/workshop/bundle.py --output cramera/src/cramera/resources/aicor_workshop
```

The authoring script generates `workshop.blend` and overview/detail renders in the
chosen source directory, and six GLBs in the resource bundle. Keep the export's
notice files with authorized copies. Asset packaging applies original workshop
attribution without changing the CRAM software or third-party PR2 licence.

## Protected hosting

The dedicated workshop gateway serves `/workshop/` with the existing access
password and an independent secure session cookie. It exposes only workshop
assets and, when explicitly configured, its native recording. It cannot call the
laboratory robot API. `cramera-workshop-gateway.service` is independent of both the
laboratory services and `cramera.service`.
