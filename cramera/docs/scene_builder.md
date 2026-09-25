# Scene Builder

Open `scene_builder.html` from the **Scene Builder** tab beside Plan Builder, or
use **Arrange a scene in Scene Builder** in the workshop panel. The protected
workshop also serves this page under its existing password.

The editor arranges independent copies of existing models. No robot plan or
physics simulation runs in this page, and editing a draft never changes an
original URDF, mesh, recorded scene or running workshop.

## Arrange a scene

1. Choose an existing scene and **Open scene copy**, or begin with **New scene**.
2. Search the asset library and add a URDF, GLB/GLTF, STL, OBJ or DAE model.
3. Select an object and use **Move** or **Rotate** handles, or enter its position,
   rotation and scale. A robot's **Whole object / robot spawn** moves its full model.
4. For URDF furniture, choose a link in **Edit part** or Alt-click the part in 3D.
   Its descendants move together; joints remain independently adjustable.
5. **Save draft** persists the layout on the serving machine. **Save a copy**
   creates a new draft. **Export JSON** and **Import JSON** transfer a layout
   between installations containing the same resource IDs.

Undo/redo covers placement, duplication, deletion, parts and joints. The browser
also keeps a recovery copy. If WebGL is unavailable, whole-object numerical editing,
saving and export remain available; new URDF part discovery requires the 3D loader.

Meshes are selectable model instances. A single combined mesh is not automatically
segmented into furniture. URDFs expose their authored link hierarchy. Missing model
dependencies produce a visible load error.

## Resource library and documents

Local discovery uses the package's `resources`, configured `scenes`, writable
`local_scenes`, and `coraplex/resources` roots. Resource IDs preserve the root name
and relative file path. Refresh the catalog after placing new assets in those roots.
Dependencies and attribution notices remain confined to these roots.

A version 1 document contains named instances with asset IDs, world position,
roll/pitch/yaw in radians, positive scale, joint values and optional link offsets.
Rotation follows URDF fixed-axis roll/pitch/yaw (`Rz * Ry * Rx`). A link offset is
expressed in its parent frame and composed before its original local transform.
Removing that offset restores its authored placement. Documents are editor layouts;
they are not executable CRAM plans or newly generated URDF files.

Drafts are written atomically under `CRAMERA_DATA/scene-builder` by default.
The API namespace is `/api/scene-builder`, relative to the current protected mount:
`catalog`, `template?asset=...`, `assets/<root>/<path>`, and `drafts[/<id>]`.
Only `POST drafts` writes data; it accepts a document and an optional existing ID.

## Protected deployment

The workshop gateway accepts `--builder-directory` containing named resource-root
directories and `--builder-drafts-directory` pointing to a separate writable store.
It exposes those configured roots only. Its password, same-origin write checks and
document size limits apply to the editor as well. Keep the original source files
and notices in each deployed library; copy model files into matching named roots
to preserve draft portability.
