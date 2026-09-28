# AICOR Wood Workshop

**Principal author and design direction:** Vanessa Hassouna · **Institution:** AICOR Institute.
Original workshop assets require permission for reuse; see [LICENSE.txt](LICENSE.txt) and [NOTICE.md](NOTICE.md).

A compact wood workshop with a timber workbench, AICOR tool board, drill-bit and
screwdriver holders, cordless driver, hand saw, hammer, vise, timber shelves and
storage bins. The Blender models use embedded wood colour/roughness textures,
metal finishes and transparent teal AICOR lettering.

The complete environment is in `environment.urdf`. Its three drawers have
independent prismatic joints and fixed child handle frames. `scene.json` describes
the assembled viewer scene, and `semantics.json` records the handle and socket
frames. Units are metres with Z up; the workbench front faces negative Y.

Eighteen objects are separate bodies, each with its own URDF, mesh reference and
initial pose: `screw_1`–`screw_5`, `nail_1`–`nail_5`, `screw_box`, `nail_box`,
`storage_box_1`–`storage_box_4`, `workshop_assembly` and `driver`. They share reusable
local meshes under `assets/`. Box walls have open collision geometry, and all six
boxes have child handle frames. The loaded screw box starts on the right shelving;
its `benchPose` in the workshop metadata identifies its destination near the front
worktop edge, within reach from the robot's clear aisle position.
Screw and nail origins are at their tips, with their shafts along local positive Z.
The screwdriver rests horizontally near the front worktop edge on a wooden cradle
that supports its shaft and rear handle, leaving the handle accessible to the robot.
The wooden assembly block has an actual pilot hole in both its visual and
collision geometry. A raised wooden jig supports it with the pilot facing the
aisle, so the robot can insert the screwdriver horizontally. The pedestal, rear
brace and side stops have matching contact geometry. Tools and timber stored on
the shelves are fixed visual props.

The CRAMERA workshop controls run a MuJoCo contact world using these collision
shapes. Boxes and their loose contents move through gravity, friction and contacts;
manipulation and drawer controls apply bounded forces. Screwdriver contact and
measured rotation drive an ideal thread constraint in the pilot hole. This does
not model cutting wood, material damage or detailed thread-flank contact. URDF
loading alone supplies geometry and articulation; physics controllers and robot
plans are separate from the model resources. The native PR2 replay is separate
from the interactive contact world.

Load the full environment in CRAM:

```python
from cramera.workshop_world import WorkshopWorld

world = WorkshopWorld().create_world()
world_with_pr2 = WorkshopWorld().build()
```

For other resource users, load `environment.urdf` with a URDF parser, then instantiate
the loose-body URDFs at the `spawn` poses in `scene.json`. To copy the world, preserve
the entire directory and its relative mesh references and notices. The PR2 robot
model is an external dependency and is not included in this original asset bundle.

The reproducible Blender authoring script is `cramera/tools/workshop/build_blender.py`
in the application repository. Browser controls and native demo instructions are
in `cramera/docs/aicor_workshop.md` there.
