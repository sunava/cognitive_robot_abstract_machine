# AICOR laboratory devices

**Principal author:** Vanessa Hassouna · **Institution:** AICOR Institute.
Original laboratory assets require written permission for reuse. Every device
folder includes `LICENSE.txt`, `NOTICE.md` and `rights.json`; keep them with
authorized copies. Existing permissions and third-party rights are preserved.

Each subdirectory is a self-contained device model. For an authorized reuse, copy
the entire device folder: `model.urdf` refers only to relative files under
`assets/`, and the GLB files embed their material images. Coordinates are metres with Z up.

Open [the instrument gallery](catalog/index.html) for rest and inspection previews
and individual Blender downloads. The same gallery is available behind the
laboratory login at `/laboratory/instruments/catalog/index.html`.

The five instruments extracted from the assembled laboratory are:

| Directory | Model | Root frame |
| --- | --- | --- |
| `centrifuge` | Housing, articulated lid, continuous rotor and four open sample sleeves | Base centre at the support surface |
| `dual_dispenser` | Dual reservoirs, nozzles, display and receiving cradle | Platform centre at the support surface |
| `scale` | Housing, weighing pan and open sample support collisions | Centre of the weighing pan |
| `tube_rack` | Two rows of three open positions | Base centre at the support surface |
| `refrigerator` | Compact cabinet, articulated glass door and six sample holders | Base centre at floor level |

These five models retain the laboratory's link names, contact geometry and
joint limits. Their `metadata.json` records the original placement in the
assembled laboratory. The room, workbench and robot are not included.

The additional instruments and accessories are original AICOR concept models,
each authored as an independent Blender scene:

| Directory | Instrument or accessory | Articulation and interaction |
| --- | --- | --- |
| `vortex_mixer` | Vortex mixer | Moving contact cup and speed control |
| `electronic_pipette` | Electronic pipette | Independent plunger and tip ejector |
| `tip_rack` | Pipette-tip rack | Opening lid and twenty-four open tip sockets |
| `tube_capper` | Tube capper / decapper | Opposing jaws, translating chuck and rotary head |
| `thermomixer` | Thermomixer | Opening cover and moving sample block |
| `microplate_reader` | Microplate reader | Sliding sample drawer |
| `microplate_96_well` | Separate microplate | Ninety-six open wells and robot grasp frames |
| `incubator` | Incubator | Opening glass door and three separate shelf bodies |
| `digital_microscope` | Digital microscope | Focus carriage, X/Y slide stage, objective turret and tilting monitor |
| `ph_meter` | pH meter | Adjustable-height and swivelling probe support |
| `ph_probe` | Separate pH electrode | Sensing bulb, protective cage and grasp frame |
| `pcr_thermocycler` | PCR thermocycler | Opening heated cover and ninety-six-position thermal block |
| `barcode_scanner` | Handheld barcode scanner | Moving trigger and scan/grasp frames |
| `barcode_dock` | Separate scanner cradle | Open holder and mounting frame |
| `label_printer` | Label printer | Opening media cover, rotating media spindle and feed roller |

Each new folder includes `model.blend`, `model.urdf`, per-link `assets/*.glb`,
`definition.json`, `metadata.json`, `preview.png` and `preview_open.png`.
The Blender timeline shows rest at frame 1, an inspection pose at frame 60 and
rest again at frame 120. The URDF exports the same body hierarchy and
interaction frames. Coordinates are metres, Z up, with the front facing -Y.
Their root frames are device-local; they have no placement in the existing
laboratory until explicitly added. Companion accessories are listed in
`metadata.json` and can be loaded or moved separately.

```python
from pathlib import Path

from cramera.laboratory_devices import (
    LaboratoryDevice,
    LaboratoryDeviceResource,
    LaboratoryInstrument,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.spatial_types import Pose

model = LaboratoryDeviceResource(LaboratoryDevice.CENTRIFUGE)
device_world = URDFParser.from_file(str(model.urdf), prefix="centrifuge_one_").parse()

# Merge into an existing CRAM semantic world at any desired placement.
world.merge_world_at_pose(
    device_world,
    Pose.from_xyz_quaternion(0.3, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0).to_homogeneous_matrix(),
)

# Copy the model for an authorized use on another machine or project.
model.copy_to(Path("my_devices"))

# The same resource helper locates the separately authored instruments.
microscope = LaboratoryDeviceResource(LaboratoryInstrument.DIGITAL_MICROSCOPE)
microscope_world = URDFParser.from_file(
    str(microscope.urdf), prefix="microscope_one_"
).parse()
```

Choose a different parser prefix for each instance to avoid duplicate body and
joint names. `LaboratoryDevice` lists the five extracted lab devices;
`LaboratoryInstrument` lists the fifteen new independent instrument and
accessory models. `LaboratoryDeviceResource` accepts either catalog.

These URDFs supply appearance, articulation and collision geometry. Instrument
controllers, weighing measurements, dispensing liquid, centrifuge interlocks,
simulated refrigeration temperatures and
the existing browser controls are implemented by the laboratory simulation; a
generic URDF loader does not execute those behaviors automatically. The new
concept models likewise need controllers for dosing, heating, mixing, optical
measurement or printing. Their nominal masses and compound-bounds inertias
are model parameters, not calibrated manufacturer data. Open sleeves use
segmented wall colliders so their insertion axes remain accessible.

Regenerate the fifteen Blender-authored instrument and accessory folders from
the CRAM workspace root using Blender 4.0 (the tested authoring version):

```sh
blender -b -t 8 --python cramera/tools/laboratory/instruments/build_blender.py -- \
  --output cramera/src/cramera/resources/laboratory_devices
```

Use `--device digital_microscope` to rebuild one device, repeat `--device` for a
subset, or add `--no-render` to skip preview images. The editable Blender file,
URDF, visual assets, metadata and asset notices are still produced.

Regenerate the static overview and validate every self-contained package:

```sh
.venv/bin/python cramera/tools/laboratory/instruments/gallery.py \
  --root cramera/src/cramera/resources/laboratory_devices
.venv/bin/python cramera/tools/laboratory/instruments/validate_packages.py \
  --root cramera/src/cramera/resources/laboratory_devices
```

Regenerate the five extracted devices after changes to the assembled lab:

```sh
.venv/bin/python cramera/tools/laboratory/device_resources.py --prepare-rack
```

The preparation step separates the rack from older room assets, adds its carry
handle and hollow lower sample guides, and clears the drawer floor. It preserves
the embedded materials and can be repeated after regenerating the laboratory
URDF without rendering the room again.
