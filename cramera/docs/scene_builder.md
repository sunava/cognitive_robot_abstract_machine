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

## Scale objects

Select a whole object and choose **Scale** to drag its size handles. The inspector
also accepts independent X/Y/Z multipliers. **Keep proportions** changes all three
axes by the same ratio when editing one axis. Scaling applies to the entire model,
including its articulated children; individual URDF link offsets remain rigid
transforms. Placement and scale are saved together and undoable.

## Arrange models inside a scanned room

1. Export your scan as a vertex-only PLY or an uncompressed PCD file.
2. Choose **Source units** (metres, centimetres or millimetres), then **Load point
   cloud**. The original file is stored with your drafts on this server.
3. Select the reference, turn off **Lock reference**, and use position, rotation
   and scale to align the scan with the editor's Z-up coordinate system. Original
   scan coordinates are retained; the editor does not recenter it automatically.
4. Enable **Lock reference** again. The scan remains visible while clicks select
   the furniture and robot models you add over it.
5. Adjust **Show reference**, opacity and point size as needed, then **Save draft**.

Imports accept up to 64 MiB and two million source points. The renderer displays
a deterministic sample of at most 250,000 finite points, retaining the full source
file on the server. PLY supports ASCII and little-/big-endian binary scalar vertex
properties. PCD supports ASCII and uncompressed binary, including packed RGB.
Compressed PCD and polygon meshes are not accepted as scan references; export a
vertex-only PLY or uncompressed PCD first.

Point clouds are visual references. They do not automatically create furniture,
collision shapes, semantic labels or a physics world. Saved layouts retain their
visibility, lock, size, opacity and alignment. JSON exports reference the uploaded
files by stable asset ID; moving a layout to another server requires copying those
scan files as well.

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
`POST drafts` accepts a document and an optional existing ID. `POST point-clouds/ply`
and `POST point-clouds/pcd` accept the original bytes with media type
`application/octet-stream`. Validated scans are content-addressed under the draft
store's `point-clouds` subdirectory and catalogued in the reserved `point_clouds`
resource root. Reimporting the same file reuses its stored asset.

## Protected deployment

The workshop gateway accepts `--builder-directory` containing named resource-root
directories and `--builder-drafts-directory` pointing to a separate writable store.
It exposes those configured roots only. Its password, same-origin write checks and
document size limits apply to the editor as well. Keep the original source files
and notices in each deployed library; copy model files into matching named roots
to preserve draft portability.

The gateway needs write access only to its draft store, including the scan
subdirectory. Configure the reverse proxy to allow a 64 MiB request body for the
point-cloud upload routes; other document endpoints keep their smaller limit.
