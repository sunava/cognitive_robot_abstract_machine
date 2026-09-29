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
- [`laboratory_devices`](laboratory_devices/README.md): twenty independently
  loadable instrument and accessory folders, listed below. Fifteen new models
  include their own editable Blender source and rest/inspection previews.
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
| Vortex mixer with moving contact cup | [`vortex_mixer/model.urdf`](laboratory_devices/vortex_mixer/model.urdf) |
| Electronic pipette with plunger and tip ejector | [`electronic_pipette/model.urdf`](laboratory_devices/electronic_pipette/model.urdf) |
| Opening pipette-tip rack | [`tip_rack/model.urdf`](laboratory_devices/tip_rack/model.urdf) |
| Tube capper / decapper | [`tube_capper/model.urdf`](laboratory_devices/tube_capper/model.urdf) |
| Thermomixer with opening cover | [`thermomixer/model.urdf`](laboratory_devices/thermomixer/model.urdf) |
| Microplate reader with sample drawer | [`microplate_reader/model.urdf`](laboratory_devices/microplate_reader/model.urdf) |
| Separate 96-well microplate | [`microplate_96_well/model.urdf`](laboratory_devices/microplate_96_well/model.urdf) |
| Incubator with glass door and three shelves | [`incubator/model.urdf`](laboratory_devices/incubator/model.urdf) |
| Digital microscope with focus, stage and objective motion | [`digital_microscope/model.urdf`](laboratory_devices/digital_microscope/model.urdf) |
| pH meter with articulated support arm | [`ph_meter/model.urdf`](laboratory_devices/ph_meter/model.urdf) |
| Separate pH electrode | [`ph_probe/model.urdf`](laboratory_devices/ph_probe/model.urdf) |
| PCR thermocycler with opening heated cover | [`pcr_thermocycler/model.urdf`](laboratory_devices/pcr_thermocycler/model.urdf) |
| Handheld barcode scanner with movable trigger | [`barcode_scanner/model.urdf`](laboratory_devices/barcode_scanner/model.urdf) |
| Separate barcode-scanner dock | [`barcode_dock/model.urdf`](laboratory_devices/barcode_dock/model.urdf) |
| Label printer with opening cover and media rollers | [`label_printer/model.urdf`](laboratory_devices/label_printer/model.urdf) |

Each device directory includes its URDF, visual meshes with embedded materials,
collision geometry, frame metadata, documentation and permission notices. For
authorized reuse, copy the complete device directory, including its notices;
its mesh references remain relative. Units are
metres, with Z pointing up.

The new instruments are original AICOR concept designs. Each has a separate
`model.blend`, `definition.json`, and rest/inspection images. Blender frames
1, 60 and 120 demonstrate rest, inspection and rest respectively. These
resources do not change the assembled laboratory until placed into a scene.
Measurements, thermal processes, pipetting and printing require a controller;
the instrument files contain geometry, joints, collision shapes and semantic
interaction frames. Mass and inertia values are approximate model parameters.

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
Use `LaboratoryDevice` for the five extracted lab devices and
`LaboratoryInstrument` for the fifteen new instrument/accessory models.
The `aicor_medical_lab`, `aicor_workshop`, `laboratory_devices` and `centrifuge_manuals` directories
are all versioned in this resource repository. The surrounding CRAMERA
repository currently also tracks these resource files for application packaging;
a commit in one repository does not automatically create a commit in the other.

The PR2 robot description and the CRAM/CRAMERA application source are separate
dependencies and are not included here.

From the CRAM workspace root, regenerate the new Blender-authored resources
with Blender 4.0 (the tested authoring version):

```sh
blender -b -t 8 --python cramera/tools/laboratory/instruments/build_blender.py -- \
  --output cramera/src/cramera/resources/laboratory_devices
```

The [device catalog](laboratory_devices/README.md) documents selection flags,
the older assembled-laboratory extraction command and loading examples.
