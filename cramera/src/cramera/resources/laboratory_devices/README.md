# AICOR laboratory devices

Each subdirectory is a self-contained device model. Copy the entire device folder
to reuse it: `model.urdf` refers only to relative files under `assets/`, and the GLB
files embed their material images. Coordinates are metres with Z up.

| Directory | Model | Root frame |
| --- | --- | --- |
| `centrifuge` | Housing, articulated lid, continuous rotor and four open sample sleeves | Base centre at the support surface |
| `dual_dispenser` | Dual reservoirs, nozzles, display and receiving cradle | Platform centre at the support surface |
| `scale` | Housing, weighing pan and open sample support collisions | Centre of the weighing pan |
| `tube_rack` | Two rows of three open positions | Base centre at the support surface |
| `refrigerator` | Compact cabinet, articulated glass door and six sample holders | Base centre at floor level |

The portable models retain the laboratory's link names, contact geometry and
joint limits. `metadata.json` records each original placement in the assembled
laboratory. The room, workbench and robot are not included.

```python
from pathlib import Path

from cramera.laboratory_devices import LaboratoryDevice, LaboratoryDeviceResource
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.spatial_types import Pose

model = LaboratoryDeviceResource(LaboratoryDevice.CENTRIFUGE)
device_world = URDFParser.from_file(str(model.urdf), prefix="centrifuge_one_").parse()

# Merge into an existing CRAM semantic world at any desired placement.
world.merge_world_at_pose(
    device_world,
    Pose.from_xyz_quaternion(0.3, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0).to_homogeneous_matrix(),
)

# Or copy the model for another machine or project.
model.copy_to(Path("my_devices"))
```

Choose a different parser prefix for each instance to avoid duplicate body and
joint names. Load every available model by iterating over `LaboratoryDevice`.

These URDFs supply appearance, articulation and collision geometry. Instrument
controllers, weighing measurements, dispensing liquid, centrifuge interlocks,
simulated refrigeration temperatures and
the existing browser controls are implemented by the laboratory simulation; a
generic URDF loader does not execute those behaviors automatically. Collision
apertures use the same compound approximations as the assembled laboratory.

Regenerate the folders after authoring asset changes:

```sh
.venv/bin/python cramera/tools/laboratory/device_resources.py
```
