# private_meshes

**Principal author and design direction:** Vanessa Hassouna.
**Institution:** AICOR Institute.

Original laboratory assets require prior written permission for reuse, subject
to the scope, existing permissions and third-party exclusions in
[LICENSE.txt](LICENSE.txt) and [NOTICE.md](NOTICE.md).

This repository contains the assembled laboratory, its independent equipment,
and a separate wood workshop. Each bundle carries its own scoped asset terms:

- [`aicor_medical_lab`](aicor_medical_lab/README.md): the room, workbench, articulated
  drawer, instruments, rack and separate sample vessels. `scene.json` describes
  the assembled scene; `environment.urdf` describes the environment and
  articulated equipment. Meshes, materials and collision data are included.
- [`aicor_workshop`](aicor_workshop/README.md): a timber workbench, tool board,
  shelves, three articulated drawers, separate sorting bins, screws and nails,
  and a wooden assembly block. Its [asset terms](aicor_workshop/LICENSE.txt)
  cover original workshop assets.
- [`laboratory_devices`](laboratory_devices/README.md): five independently
  loadable instrument folders, listed below.
- [`centrifuge_manuals`](centrifuge_manuals/): the two operating-manual editions
  used by the laboratory knowledge demonstration.

The independent device models are:

| Device | URDF |
| --- | --- |
| Centrifuge with articulated lid and rotor | [`centrifuge/model.urdf`](laboratory_devices/centrifuge/model.urdf) |
| Dual dispenser | [`dual_dispenser/model.urdf`](laboratory_devices/dual_dispenser/model.urdf) |
| Laboratory scale | [`scale/model.urdf`](laboratory_devices/scale/model.urdf) |
| Tube rack | [`tube_rack/model.urdf`](laboratory_devices/tube_rack/model.urdf) |
| Sample refrigerator with articulated glass door | [`refrigerator/model.urdf`](laboratory_devices/refrigerator/model.urdf) |

Each device directory includes its URDF, visual meshes with embedded materials,
collision geometry, frame metadata, documentation and permission notices. For
authorized reuse, copy the complete device directory, including its notices;
its mesh references remain relative. Units are
metres, with Z pointing up.

Load a model directly with CRAM's semantic digital twin:

```python
from semantic_digital_twin.adapters.urdf import URDFParser

world = URDFParser.from_file(
    "/path/to/private_meshes/laboratory_devices/centrifuge/model.urdf",
    prefix="centrifuge_one_",
).parse()
```

Use a distinct prefix for each instance. Loading these models does not require
the CRAMERA viewer. Instrument controllers and fluid simulation belong to the
laboratory application; loading a URDF supplies the geometry and articulation.

In the CRAMERA workspace this repository is checked out directly at
`cramera/src/cramera/resources`. Its `laboratory_devices` catalog can therefore
also be accessed through `cramera.laboratory_devices.LaboratoryDeviceResource`.
The `aicor_medical_lab`, `aicor_workshop`, `laboratory_devices` and `centrifuge_manuals` directories
are all versioned in this resource repository. The surrounding CRAMERA
repository currently also tracks these resource files for application packaging;
a commit in one repository does not automatically create a commit in the other.

The PR2 robot description and the CRAM/CRAMERA application source are separate
dependencies and are not included here.
