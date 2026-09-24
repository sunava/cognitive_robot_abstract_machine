# AICOR Laboratory assets

This portable bundle contains the authored laboratory's environment, movable
objects, collision meshes, materials, initial poses, and rack-slot metadata.
Four independently movable glasses occupy A1, A2, B1, and B3. The empty reserve
glass in B3 shares the clear glass visual mesh and has its own URDF body.
All asset references are relative to this directory. Coordinates use metres,
Z up, and pose quaternions in XYZW order.

The centrifuge has separate housing, rotor, and hinged-lid GLB assets. Its URDF
contains open collision sleeves for four existing tubes, a continuous rotor
joint, and a bounded lid joint. `scene.json` records the opposing holder pairs
and rotor-local sample positions. The included initial viewer frame opens the
lid; a newly loaded planning world exposes both joints for ordinary CRAM control.

The compact sample refrigerator stands beside the bench. Its glass door is a
separate articulated body, and two shelves provide six numbered sample holders.
The interactive physics session controls the door, observes supported samples,
and models cabinet temperature approaching the cooling target or room temperature
according to the door state. This is a reduced temperature simulation.

In the physics controls, select **Sample refrigerator → View refrigerator**, then
**Open door**. Select a glass and a free storage position R1–R6, and choose
**Store selected glass**. **Retrieve sample to rack** returns the selected stored
sample to a free rack position. These manual transfers use physical force targets;
the PR2 does not execute a refrigerator plan. Close the door to cool toward 4 °C.
Retrieve stored samples before starting an existing PR2 demonstration.

Click the centrifuge, dual dispenser, scale, tube rack or refrigerator and choose
**Open manual**. **Layers → Manual mode** opens the currently selected device;
the **Manual instrument** selector switches between instruments.
Numbered titles sit close to their parts with short leader lines. Select a title
or number to expand its explanation. The lines follow camera movement and
articulation. Live observations add the relevant instrument's measurements and
state; missing observations remain unknown. In a live physics scene, open the
centrifuge lid to show its interior annotations. **Show controls** restores the
instrument controls, and **Exit manual** returns to the ordinary scene.

The adjacent `resources/laboratory_devices/` catalog provides independently
loadable URDF folders, each with its own local meshes and documentation. These
portable models preserve the geometry and joints used in this assembled scene.

The runtime bundle ships with the `cramera` Python package and requires no
Blender installation, local scene cache, browser, server, or PR2 recording.
`trajectory.json` contains one initial-state frame for scene-viewer compatibility;
it is not an executed robot plan.

The scene was generated with `cramera/tools/laboratory/build_blender.py` and
`cramera/tools/laboratory/bundle.py` from
`cramera/tools/laboratory/specification.json`. The source Blender project and
preview renders are authoring outputs and are excluded from this runtime bundle.
The assets are distributed under the repository's GPL-3.0-only license.

The GLB files retain the authored visual materials. URDF collision geometry is
separate from those visuals: rack openings use square approximations, glass
walls use compound boxes, and the rounded glass bottom uses a shell mesh.
The URDF models include estimated inertias. Loading the world provides geometry,
kinematics, and collision queries; contact dynamics, liquid dynamics, and glass
fracture require a separately configured simulation backend. The packaged
validation metadata describes the source scene, not a robot execution result.
