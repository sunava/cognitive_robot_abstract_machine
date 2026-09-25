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

Ten fasteners and two open sorting boxes are separate bodies, each with its own
URDF, mesh reference and initial pose: `screw_1`–`screw_5`, `nail_1`–`nail_5`,
`screw_box` and `nail_box`. They share reusable local meshes under `assets/`.
Screw and nail origins are at their tips, with their shafts along local positive Z.
The wooden assembly block has an actual pilot hole in both its visual and
collision geometry. Tools and timber stored on the shelves are fixed visual props.

The CRAMERA workshop controls open/close each drawer, select and lift one fastener,
move it above the bench, return it to its box and seat a screw in the pilot hole.
These controls position objects directly; they do not simulate force, free falling,
cutting wood or thread friction. URDF loading supplies geometry and articulation;
application controllers and robot plans are separate from the model resources.

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
