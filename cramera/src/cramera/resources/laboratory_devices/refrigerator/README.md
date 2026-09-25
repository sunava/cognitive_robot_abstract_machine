# AICOR sample refrigerator

**Principal author:** Vanessa Hassouna · **Institution:** AICOR Institute.
Original laboratory assets require prior written permission for reuse; see
[LICENSE.txt](LICENSE.txt) and [NOTICE.md](NOTICE.md). Keep these notices with
authorized copies. Existing permissions and third-party rights are preserved.

For authorized reuse, load `model.urdf` independently or copy the complete folder
into another project.
All visual meshes and materials are included under `assets/`.

The root `laboratory_refrigerator_base` is at the cabinet's centre at floor level.
Coordinates are metres, with Z up and the front toward negative Y. The cabinet
is 0.54 m wide, 0.58 m deep and 0.84 m high. Its glass door pivots outward about
the front-left vertical hinge; the URDF defines the joint axis and limits.

The fixed child link `laboratory_refrigerator_door_handle` identifies the centre
of the grip bar on `laboratory_refrigerator_door`. Its axes match the door's axes
and it moves with the hinge. Its geometry remains on the door.

Six open sample holders, R1–R6, stand on two shelves. Cabinet walls, shelves,
door glass and holders have collision geometry. The room, workbench and robot
are not included. `metadata.json` records the original laboratory placement.

This model supplies appearance, articulation and contact shapes. Door controls,
observed sample occupancy and simulated temperature require the laboratory
controller. A generic URDF loader does not simulate refrigeration.
