# Laboratory handle frames

Each visible handle has a named URDF link attached by a fixed joint to the body
carrying it, following the apartment cabinet model's frame structure. The frame
origin is at the centre of the grip bar, in metres; its axes match the parent
body's axes. Drawer, lid and door movement therefore move their handle frames.

| Handle frame | Parent body |
| --- | --- |
| `laboratory_drawer_handle` | `laboratory_drawer` |
| `laboratory_cabinet_lower_handle` | `laboratory_room` |
| `laboratory_cabinet_middle_handle` | `laboratory_room` |
| `laboratory_cabinet_right_handle` | `laboratory_room` |
| `laboratory_centrifuge_lid_handle` | `laboratory_centrifuge_lid` |
| `laboratory_refrigerator_door_handle` | `laboratory_refrigerator_door` |

The three cabinet fronts are still stationary geometry. Their handle frames do
not add opening joints. Existing visual and collision geometry stays on the
parent bodies, so the added frames do not duplicate meshes or contact surfaces.
The MuJoCo contact model represents these empty fixed frames as massless sites;
the instrument and robot masses, inertias and actuator behavior remain unchanged.

In CRAMERA, enable **Layers → TF frames**. Use its gear menu to show frame names
or select individual frames. The axes follow the measured instrument joints.

`LaboratoryEnvironment.create_world()` also registers the named bodies as CRAM
`Handle` annotations. They can be queried directly or looked up by frame name:

```python
from cramera.laboratory_handles import LaboratoryHandle
from cramera.laboratory_world import LaboratoryEnvironment
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle

world = LaboratoryEnvironment().create_world()
handles = world.get_semantic_annotations_by_type(Handle)
grip = world.get_body_by_name(LaboratoryHandle.CENTRIFUGE_LID)
world_T_grip = world.compute_forward_kinematics(world.root, grip)
```

The independent centrifuge and refrigerator resources retain their handle links
in `resources/laboratory_devices/<device>/model.urdf`. A generic URDF load exposes
these named frames; the assembled laboratory loader adds the semantic annotations.
These are model frames available to the viewer and CRAM world, not a separately
started ROS TF broadcaster.
